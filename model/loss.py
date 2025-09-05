import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from einops import rearrange
from torch.linalg import vector_norm

import math
import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram
import typing as tp
from model.balancer import Balancer

def mse_loss():
    return torch.nn.MSELoss()


def l1_loss():
    return torch.nn.L1Loss()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# Discriminator hinge losses
def hinge_discriminator_real_loss(real_outputs):
    """Hinge loss for discriminator on real data: max(0, 1 - real)"""
    return F.relu(1.0 - real_outputs).mean()


def hinge_discriminator_fake_loss(fake_outputs):
    """Hinge loss for discriminator on fake data: max(0, 1 + fake)"""
    return F.relu(1.0 + fake_outputs).mean()


# Generator hinge loss  
def hinge_generator_loss(fake_outputs):
    """Hinge loss for generator: -fake"""
    return F.relu(1.0 - fake_outputs).mean()


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def gradient_penalty(wave, output, weight=10, center=0.):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs=output,
        inputs=wave,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim=1) - center) ** 2).mean()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))

class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.
    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(torch.nn.Module):
    """L1 Loss on MelSpectrogram.
    Args:
        sample_rate (int): Sample rate.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                             log=log, normalized=normalized, floor_level=floor_level)

    def forward(self, x, y):
        self.melspec.to(x.device)
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return self.l1(s_x, s_y)


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).
    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss

def _stft(x: torch.Tensor, fft_size: int, hop_length: int, win_length: int,
          window: tp.Optional[torch.Tensor], normalized: bool) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x: Input signal tensor (B, C, T).
        fft_size (int): FFT size.
        hop_length (int): Hop size.
        win_length (int): Window length.
        window (torch.Tensor or None): Window function type.
        normalized (bool): Whether to normalize the STFT or not.
    Returns:
        torch.Tensor: Magnitude spectrogram (B, C, #frames, fft_size // 2 + 1).
    """
    B, C, T = x.shape
    x_stft = torch.stft(
        x.view(-1, T), fft_size, hop_length, win_length, window,
        normalized=normalized, return_complex=True,
    )
    x_stft = x_stft.view(B, C, *x_stft.shape[1:])
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss."""
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """Calculate forward propagation.
        Args:
            x_mag: Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag: Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            torch.Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.epsilon)


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss.
    Args:
        epsilon (float): Epsilon value for numerical stability.
    """
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """Calculate forward propagation.
        Args:
            x_mag (torch.Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (torch.Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            torch.Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(self.epsilon + y_mag), torch.log(self.epsilon + x_mag))


class STFTLosses(nn.Module):
    """STFT losses.
    Args:
        n_fft (int): Size of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (str): Window function type.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss(epsilon)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(epsilon)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (torch.Tensor): Predicted signal (B, T).
            y (torch.Tensor): Groundtruth signal (B, T).
        Returns:
            torch.Tensor: Spectral convergence loss value.
            torch.Tensor: Log STFT magnitude loss value.
        """
        # Ensure window is on the same device as input
        window = self.window.to(x.device)
        
        x_mag = _stft(x, self.n_fft, self.hop_length,
                      self.win_length, window, self.normalized)  # type: ignore
        y_mag = _stft(y, self.n_fft, self.hop_length,
                      self.win_length, window, self.normalized)  # type: ignore
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss(nn.Module):
    """Multi-resolution STFT loss for SoundStream reconstruction.
    
    Combines spectral convergence loss and log magnitude loss across multiple scales.
    This is more comprehensive than simple L1 loss in STFT domain.
    """
    
    def __init__(self, 
                 n_ffts: tp.Sequence[int] = [2048, 1024, 512, 256, 128],
                 hop_lengths: tp.Optional[tp.Sequence[int]] = None, 
                 win_lengths: tp.Optional[tp.Sequence[int]] = None,
                 window: str = "hann_window",
                 factor_sc: float = 0.1,
                 factor_mag: float = 0.1,
                 normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        
        # Default hop and win lengths if not provided
        if hop_lengths is None:
            hop_lengths = [f // 4 for f in n_ffts]
        if win_lengths is None:
            win_lengths = n_ffts
            
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses.append(STFTLosses(fs, ss, wl, window, normalized, epsilon))
        
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        
    def forward(self, x_real, x_fake):
        """
        Args:
            x_real: Target audio [B, T] or [B, C, T]
            x_fake: Generated audio [B, T] or [B, C, T]
        Returns:
            Multi-resolution STFT loss (spectral convergence + log magnitude)
        """
        # Handle different input dimensions
        if x_real.dim() == 3 and x_real.shape[1] > 1:  # Multi-channel audio [B, C, T]
            # Process each channel separately and average the losses
            batch_size, num_channels, seq_len = x_real.shape
            
            sc_loss = torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)
            mag_loss = torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)
            
            # Loop through each channel
            for c in range(num_channels):
                x_real_ch = x_real[:, c:c+1, :]  # [B, 1, T] - keep channel dimension
                x_fake_ch = x_fake[:, c:c+1, :]  # [B, 1, T] - keep channel dimension
                
                # Compute STFT losses for this channel
                for stft_loss in self.stft_losses:
                    sc_l, mag_l = stft_loss(x_real_ch, x_fake_ch)
                    sc_loss += sc_l
                    mag_loss += mag_l
            
            # Average over channels and STFT scales
            sc_loss /= (num_channels * len(self.stft_losses))
            mag_loss /= (num_channels * len(self.stft_losses))
            
        else:
            # Single channel or 2D input
            if x_real.dim() == 3:
                # Single channel [B, 1, T] - already correct format
                pass
            elif x_real.dim() == 2:
                # Add channel dimension [B, T] -> [B, 1, T]
                x_real = x_real.unsqueeze(1)
                x_fake = x_fake.unsqueeze(1)
            elif x_real.dim() > 3:
                # Flatten extra dimensions
                x_real = x_real.view(x_real.shape[0], 1, -1)
                x_fake = x_fake.view(x_fake.shape[0], 1, -1)
                
            sc_loss = torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)
            mag_loss = torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)
            
            for stft_loss in self.stft_losses:
                sc_l, mag_l = stft_loss(x_real, x_fake)
                sc_loss += sc_l
                mag_loss += mag_l
                
            sc_loss /= len(self.stft_losses)
            mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class SoundStreamLoss(nn.Module):
    """
    SoundStream neural audio codec loss function.
    
    Implements the loss components from SoundStream paper:
    1. Reconstruction Loss (Multi-resolution STFT loss) - Eq. (2)
    2. Adversarial Loss (Generator hinge loss) - Eq. (3) 
    3. Feature Matching Loss - Eq. (4)
    4. Commitment Loss (VQ regularization)
    
    Total loss: L = λ_rec * L_rec + λ_adv * L_adv + λ_fm * L_fm + λ_commit * L_commit
    """
    
    def __init__(self, 
                 lambda_rec=1.0,      # Reconstruction loss weight
                 lambda_adv=1.0,      # Adversarial loss weight  
                 lambda_fm=100.0,     # Feature matching loss weight
                 lambda_commit=0.02,  # Commitment loss weight
                 n_ffts: tp.Sequence[int] = [2048, 1024, 512, 256, 128],
                 hop_lengths: tp.Optional[tp.Sequence[int]] = None,
                 win_lengths: tp.Optional[tp.Sequence[int]] = None,
                 factor_sc: float = 0.1,
                 factor_mag: float = 0.1,
                 balance_losses: bool = True,
                 balance_ema_decay: float = 0.999,
                 # parameters for reconstruction loss weighting
                 stft_loss_weight: float = 0.0,      # Weight for original STFT loss
                 soundstream_loss_weight: float = 1.0, # Weight for SoundStream mel loss
                 soundstream_scales: tp.Sequence[int] = (5, 6, 7, 8, 9, 10, 11),
                 soundstream_n_mels: int = 64):
        super().__init__()
        
        # Loss weights (used when balance_losses=False)
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_commit = lambda_commit
        
        # Reconstruction loss component weights
        self.stft_loss_weight = stft_loss_weight
        self.soundstream_loss_weight = soundstream_loss_weight
        
        # multi-resolution STFT loss linear scale
        self.stft_loss = STFTLoss(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            win_lengths=win_lengths,
            factor_sc=factor_sc,
            factor_mag=factor_mag
        )
        
        # reconstruction loss with multi-scale mel spectrogram
        self.soundstream_stft_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=24000,
            range_start=5,  # Start from 2^5 = 32
            range_end=12,   # End at 2^11 = 2048 (range_end is exclusive)
            n_mels=soundstream_n_mels,
            f_min=0.0,
            f_max=None,
            normalized=True,
            alphas=True,  # Use alpha coefficients as in SoundStream
            floor_level=1e-5
        )
        
        # Gradient balancer for stable multi-loss training
        self.balance_losses = balance_losses
        if balance_losses:
            # Balancer weights - these represent desired relative importance
            self.balancer = Balancer(
                weights={
                    'reconstruction': lambda_rec,
                    'adversarial': lambda_adv, 
                    'feature_matching': lambda_fm,
                    'commitment': lambda_commit
                },
                balance_grads=True,
                total_norm=1.0,
                ema_decay=balance_ema_decay,
                per_batch_item=True,
                monitor=True  # Enable monitoring for debugging
            )
    
    def reconstruction_loss(self, x_real, x_fake):
        """
        Multi-resolution STFT reconstruction loss combining both approaches.
        
        Uses both:
        1. Multi-resolution STFT loss with spectral convergence and log magnitude
        2. SoundStream-style mel-spectrogram loss with L1 + L2 components
        
        Args:
            x_real: Ground truth audio [B, C, T] or [B, T]
            x_fake: Reconstructed audio [B, C, T] or [B, T]
        """
        total_reconstruction_loss = torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)
        
        # Original multi-resolution STFT loss (if weight > 0)
        if self.stft_loss_weight > 0:
            stft_loss = self.stft_loss(x_real, x_fake)
            total_reconstruction_loss += self.stft_loss_weight * stft_loss
        
        # SoundStream-style reconstruction loss with mel spectrograms (if weight > 0)
        if self.soundstream_loss_weight > 0:
            soundstream_loss = self.soundstream_stft_loss(x_real, x_fake)
            total_reconstruction_loss += self.soundstream_loss_weight * soundstream_loss
        
        return total_reconstruction_loss
    
    def adversarial_loss(self, disc_fake_outputs):
        """
        Generator adversarial loss (Eq. 2 in SoundStream paper).
        
        L_adv = -D(x̂)  (for single STFT discriminator)
        
        Args:
            disc_fake_outputs: STFT discriminator outputs on fake data (list of features)
        """
        if isinstance(disc_fake_outputs, list) and len(disc_fake_outputs) > 0:
            # STFT discriminator returns list of features, final element is the score
            final_score = disc_fake_outputs[-1]
            return hinge_generator_loss(final_score)
        else:
            # Direct discriminator score
            return hinge_generator_loss(disc_fake_outputs)
    
    def feature_matching_loss(self, disc_real_outputs, disc_fake_outputs):
        """
        Feature matching loss (Eq. 3 in SoundStream paper).
        
        L_fm = Σ_i ||D^(i)(x) - D^(i)(x̂)||_1  (for single STFT discriminator)
        
        Args:
            disc_real_outputs: STFT discriminator features on real data (list)
            disc_fake_outputs: STFT discriminator features on fake data (list)
        """
        if not isinstance(disc_real_outputs, list) or not isinstance(disc_fake_outputs, list):
            return torch.tensor(0.0, device=disc_real_outputs.device if hasattr(disc_real_outputs, 'device') else 'cpu')
        
        fm_loss = 0.0
        # Exclude final layer (discriminator score) from feature matching
        for real_feat, fake_feat in zip(disc_real_outputs[:-1], disc_fake_outputs[:-1]):
            fm_loss += F.l1_loss(fake_feat, real_feat.detach()) / torch.abs(real_feat.detach()).mean()
        # Average over number of feature layers
        num_layers = len(disc_real_outputs) - 1
        return fm_loss / max(num_layers, 1)
    
    def commitment_loss(self, quantized, encodings):
        """
        Vector quantization commitment loss.
        
        L_commit = ||sg[z_e] - e||²₂
        
        Args:
            quantized: Quantized vectors from VQ
            encodings: Original encoder outputs before quantization
        """
        if quantized is None or encodings is None:
            return torch.tensor(0.0, device=quantized.device if quantized is not None else 'cpu')
        
        # Stop gradient on quantized vectors (sg[z_e])
        return F.mse_loss(quantized.detach(), encodings)
    
    def forward(self, 
                x_real, 
                x_fake, 
                disc_real_outputs=None,
                disc_fake_outputs=None, 
                quantized=None,
                encodings=None):
        """
        Compute total generator loss with optional gradient balancing.
        
        Args:
            x_real: Ground truth audio [B, C, T] or [B, T]
            x_fake: Reconstructed audio [B, C, T] or [B, T] 
            disc_real_outputs: Discriminator outputs/features on real audio
            disc_fake_outputs: Discriminator outputs/features on fake audio
            quantized: Quantized representations from VQ
            encodings: Original encodings before quantization
            
        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary with individual loss values
        """
        losses = {}
        
        # 1.1 Reconstruction Loss (Time domain)
        rec_loss_t = l1_loss()(x_real, x_fake)
        losses['reconstruction_t'] = rec_loss_t

        # 1.2 Reconstruction Loss (STFT domain)
        rec_loss_f = self.reconstruction_loss(x_real, x_fake)
        losses['reconstruction_f'] = rec_loss_f
        
        # 2. Adversarial Loss
        adv_loss = torch.tensor(0.0, device=x_real.device)
        if disc_fake_outputs is not None:
            adv_loss = self.adversarial_loss(disc_fake_outputs)
        losses['adversarial'] = adv_loss
        
        # 3. Feature Matching Loss
        fm_loss = torch.tensor(0.0, device=x_real.device)
        if disc_real_outputs is not None and disc_fake_outputs is not None:
            fm_loss = self.feature_matching_loss(disc_real_outputs, disc_fake_outputs)
        losses['feature_matching'] = fm_loss
        
        # # 4. Commitment Loss
        # commit_loss = self.commitment_loss(quantized, encodings)
        # print("commitment", commit_loss)
        # losses['commitment'] = commit_loss
        
        # Apply gradient balancing if enabled
        if self.balance_losses and x_fake.requires_grad:
            # Filter out zero losses to avoid balancer issues
            active_losses = {k: v for k, v in losses.items() 
                           if k in self.balancer.weights and v.item() >= 0}

            if len(active_losses) > 1:  # Only balance if we have multiple active losses
                # Use balancer for gradient balancing (implements Equation 5)
                total_loss = self.balancer.backward(active_losses, x_fake)
                
                # Store balancer metrics for monitoring
                balancer_metrics = self.balancer.metrics
                losses.update({f'balance_{k}': v for k, v in balancer_metrics.items()})
            else:
                # Fallback to manual weighting
                total_loss = (self.lambda_rec * (rec_loss_t + rec_loss_f) + 
                             self.lambda_adv * adv_loss +
                             self.lambda_fm * fm_loss) #+ 
                            #  self.lambda_commit * commit_loss)
        else:
            # Manual weighted loss (original approach)
            total_loss = (self.lambda_rec * rec_loss_f + 
                         self.lambda_adv * adv_loss +
                         self.lambda_fm * fm_loss) # + 
                        #  self.lambda_commit * commit_loss)

        losses['total'] = total_loss
        return total_loss, losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for SoundStream training with single STFT discriminator.
    
    Implements hinge loss for discriminator training:
    L_D = max(0, 1 - D(x)) + max(0, 1 + D(x̂))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, disc_real_outputs, disc_fake_outputs):
        """
        Compute discriminator loss for single STFT discriminator.
        
        Args:
            disc_real_outputs: STFT discriminator outputs on real audio (list of features)
            disc_fake_outputs: STFT discriminator outputs on fake audio (list of features)
            
        Returns:
            Total discriminator loss
        """
        # Extract final discriminator scores (last element in feature list)
        if isinstance(disc_real_outputs, list) and len(disc_real_outputs) > 0:
            real_score = disc_real_outputs[-1]
        else:
            real_score = disc_real_outputs
            
        if isinstance(disc_fake_outputs, list) and len(disc_fake_outputs) > 0:
            fake_score = disc_fake_outputs[-1]
        else:
            fake_score = disc_fake_outputs
        
        # Compute hinge loss
        real_loss = hinge_discriminator_real_loss(real_score)
        fake_loss = hinge_discriminator_fake_loss(fake_score)
        
        return real_loss + fake_loss


def soundstream_loss(**kwargs):
    """Factory function for SoundStream generator loss.
    
    Args:
        **kwargs: Parameters for SoundStreamLoss including:
            - lambda_rec, lambda_adv, lambda_fm, lambda_commit: Loss weights
            - n_ffts, hop_lengths, win_lengths: STFT parameters
            - factor_sc, factor_mag: Spectral loss weighting factors
            - balance_losses: Whether to use gradient balancing (default: True)
            - balance_ema_decay: EMA decay for balancer (default: 0.999)
    """ 
    return SoundStreamLoss(**kwargs)


def discriminator_loss():
    """Factory function for SoundStream discriminator loss."""
    return DiscriminatorLoss()
