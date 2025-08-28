import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from einops import rearrange
from torch.linalg import vector_norm
import torchaudio


def mse_loss():
    return torch.nn.MSELoss()


def l1_loss():
    return torch.nn.L1Loss()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


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


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss for time-frequency domain reconstruction."""
    
    def __init__(self, 
                 fft_sizes=[2048, 1024, 512, 256, 128], 
                 hop_lengths=None, 
                 win_lengths=None,
                 alpha=1.0, 
                 beta=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths or [f // 4 for f in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        self.alpha = alpha  # Weight for spectral convergence
        self.beta = beta    # Weight for log magnitude loss
        
    def stft(self, x, fft_size, hop_length, win_length):
        """Compute STFT."""
        return torch.stft(
            x, 
            n_fft=fft_size, 
            hop_length=hop_length, 
            win_length=win_length,
            return_complex=True,
            center=True
        )
    
    def spectral_convergence_loss(self, x_mag, y_mag):
        """Spectral convergence loss."""
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(x_mag, p='fro')
    
    def log_magnitude_loss(self, x_mag, y_mag, eps=1e-7):
        """Log magnitude loss."""
        return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))
    
    def forward(self, x, y):
        """
        Args:
            x: Real audio [B, T]
            y: Generated audio [B, T]
        """
        if x.dim() > 2:
            x = x.squeeze(1)  # Remove channel dim if present
        if y.dim() > 2:
            y = y.squeeze(1)
            
        total_loss = 0.0
        
        for fft_size, hop_length, win_length in zip(self.fft_sizes, self.hop_lengths, self.win_lengths):
            x_stft = self.stft(x, fft_size, hop_length, win_length)
            y_stft = self.stft(y, fft_size, hop_length, win_length)
            
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_magnitude_loss(x_mag, y_mag)
            
            total_loss += self.alpha * sc_loss + self.beta * mag_loss
            
        return total_loss / len(self.fft_sizes)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for discriminator features."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, real_features, fake_features):
        """
        Args:
            real_features: List of feature maps from discriminator on real data
            fake_features: List of feature maps from discriminator on fake data
        """
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)


class SoundStreamLoss(nn.Module):
    """
    SoundStream loss function combining multiple loss components:
    - Reconstruction loss (time domain)
    - Spectral loss (frequency domain) 
    - Adversarial loss (generator)
    - Feature matching loss
    - Commitment loss (for VQ)
    """
    
    def __init__(self, 
                 lambda_rec=1.0,
                 lambda_spec=1.0, 
                 lambda_adv=1.0,
                 lambda_feat=1.0,
                 lambda_commit=0.02,
                 use_spectral_loss=True,
                 use_feature_matching=True):
        super().__init__()
        
        # Loss weights
        self.lambda_rec = lambda_rec
        self.lambda_spec = lambda_spec
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_commit = lambda_commit
        
        # Loss components
        self.reconstruction_loss = nn.L1Loss()
        self.spectral_loss = SpectralLoss() if use_spectral_loss else None
        self.feature_matching_loss = FeatureMatchingLoss() if use_feature_matching else None
        
        self.use_spectral_loss = use_spectral_loss
        self.use_feature_matching = use_feature_matching
    
    def generator_loss(self, discriminator_outputs):
        """Compute generator adversarial loss."""
        adv_loss = 0.0
        for disc_output in discriminator_outputs:
            if isinstance(disc_output, dict):
                # Multiple discriminators
                for key, output in disc_output.items():
                    # Use the final output (last element in feature list)
                    final_output = output[-1] if isinstance(output, list) else output
                    adv_loss += hinge_gen_loss(final_output)
            else:
                # Single discriminator output
                final_output = disc_output[-1] if isinstance(disc_output, list) else disc_output
                adv_loss += hinge_gen_loss(final_output)
        return adv_loss
    
    def commitment_loss(self, quantized, encodings):
        """VQ commitment loss."""
        if quantized is None or encodings is None:
            return torch.tensor(0.0, device=quantized.device if quantized is not None else 'cpu')
        return F.mse_loss(quantized.detach(), encodings)
    
    def forward(self, 
                real_audio, 
                fake_audio, 
                discriminator_real_outputs=None,
                discriminator_fake_outputs=None,
                quantized=None,
                encodings=None,
                mode='generator'):
        """
        Args:
            real_audio: Original audio [B, C, T] or [B, T]
            fake_audio: Reconstructed audio [B, C, T] or [B, T]
            discriminator_real_outputs: Discriminator outputs on real audio
            discriminator_fake_outputs: Discriminator outputs on fake audio
            quantized: Quantized representations from VQ
            encodings: Original encodings before quantization
            mode: 'generator' or 'discriminator'
        """
        losses = {}
        total_loss = 0.0
        
        # Ensure audio tensors have the right shape for loss computation
        if real_audio.dim() == 3 and real_audio.shape[1] == 4:
            # Convert from [B, 4, T] to [B, T] by taking mean across channels
            real_audio_for_loss = real_audio.mean(dim=1)
            fake_audio_for_loss = fake_audio.mean(dim=1)
        else:
            real_audio_for_loss = real_audio.squeeze(1) if real_audio.dim() == 3 else real_audio
            fake_audio_for_loss = fake_audio.squeeze(1) if fake_audio.dim() == 3 else fake_audio
        
        if mode == 'generator':
            # 1. Reconstruction Loss (Time Domain)
            rec_loss = self.reconstruction_loss(fake_audio, real_audio)
            losses['reconstruction'] = rec_loss
            total_loss += self.lambda_rec * rec_loss
            
            # 2. Spectral Loss (Frequency Domain)
            if self.use_spectral_loss and self.spectral_loss is not None:
                spec_loss = self.spectral_loss(real_audio_for_loss, fake_audio_for_loss)
                losses['spectral'] = spec_loss
                total_loss += self.lambda_spec * spec_loss
            
            # 3. Adversarial Loss
            if discriminator_fake_outputs is not None:
                adv_loss = self.generator_loss(discriminator_fake_outputs)
                losses['adversarial'] = adv_loss
                total_loss += self.lambda_adv * adv_loss
            
            # 4. Feature Matching Loss
            if (self.use_feature_matching and 
                self.feature_matching_loss is not None and 
                discriminator_real_outputs is not None and 
                discriminator_fake_outputs is not None):
                
                feat_loss = 0.0
                # Handle multiple discriminators
                if isinstance(discriminator_real_outputs, dict):
                    for key in discriminator_real_outputs.keys():
                        if key in discriminator_fake_outputs:
                            real_feats = discriminator_real_outputs[key]
                            fake_feats = discriminator_fake_outputs[key]
                            feat_loss += self.feature_matching_loss(real_feats, fake_feats)
                elif isinstance(discriminator_real_outputs, list):
                    feat_loss = self.feature_matching_loss(discriminator_real_outputs, discriminator_fake_outputs)
                
                losses['feature_matching'] = feat_loss
                total_loss += self.lambda_feat * feat_loss
            
            # 5. Commitment Loss (VQ)
            if quantized is not None and encodings is not None:
                commit_loss = self.commitment_loss(quantized, encodings)
                losses['commitment'] = commit_loss
                total_loss += self.lambda_commit * commit_loss
        
        elif mode == 'discriminator':
            # Discriminator loss is handled separately in the training loop
            pass
        
        losses['total'] = total_loss
        return total_loss, losses


def soundstream_loss(**kwargs):
    """Factory function for SoundStream loss."""
    return SoundStreamLoss(**kwargs)
