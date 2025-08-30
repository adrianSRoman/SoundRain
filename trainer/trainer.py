import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from trainer.base_trainer import BaseTrainer
from model.soundrain import STFTDiscriminator
from model.loss import DiscriminatorLoss
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        
        # Initialize discriminators if using SoundStream loss
        self.use_discriminators = hasattr(loss_function, 'lambda_adv') and loss_function.lambda_adv > 0
        
        if self.use_discriminators:
            self._setup_discriminators(config)
            # Initialize discriminator loss function
            self.discriminator_loss_fn = DiscriminatorLoss()
            # Set discriminator training frequency from config
            self.disc_train_freq = config.get("discriminator_train_freq", 1)
    
    def _setup_discriminators(self, config):
        """Setup discriminators and their optimizers."""
        # Get discriminator config or use defaults
        disc_config = config.get("discriminator", {})
        
        # STFT discriminator  
        self.stft_discriminator = STFTDiscriminator(
            C=disc_config.get("stft_C", 32),
            F_bins=disc_config.get("stft_F_bins", 1024)
        ).to(self.device)
        
        # Multi-GPU support
        if self.n_gpu > 1:
            self.stft_discriminator = torch.nn.DataParallel(self.stft_discriminator, device_ids=list(range(self.n_gpu)))
        
        # Discriminator optimizer
        disc_lr = config.get("discriminator_optimizer", {}).get("lr", 2e-4)
        disc_betas = config.get("discriminator_optimizer", {}).get("betas", (0.5, 0.9))
        
        self.stft_disc_optimizer = torch.optim.Adam(
            self.stft_discriminator.parameters(), 
            lr=disc_lr, 
            betas=disc_betas
        )
    
    def _compute_stft(self, audio, n_fft=1024, hop_length=256, win_length=1024):
        """Compute STFT for STFT discriminator."""
         # Ensure audio has channel dimension
        if audio.dim() == 2:  # [B, T]
            audio = audio.unsqueeze(1)  # -> [B, 1, T]

        B, C, T = audio.shape

        # Compute STFT channel-wise
        stft = torch.stft(
            audio.view(-1, T),     # [B*C, T]
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(audio.device),
            normalized=False,
            return_complex=True,
        )  # -> [B*C, F, T]

        # Reshape back to [B, C, F, T]
        stft = stft.view(B, C, stft.size(-2), stft.size(-1))

        # Separate real/imag as extra dim instead of concatenating
        stft_features = torch.cat([stft.real, stft.imag], dim=1)  # -> [B, C * 2, F, T]

        return stft_features
    
    def _discriminator_step(self, real_audio, fake_audio):
        """Perform discriminator training step."""
        self.stft_disc_optimizer.zero_grad()

        # STFT discriminator
        real_stft = self._compute_stft(real_audio)
        fake_stft = self._compute_stft(fake_audio.detach())
        
        real_stft_outputs = self.stft_discriminator(real_stft)
        fake_stft_outputs = self.stft_discriminator(fake_stft)
        
        # Combine discriminator outputs for loss computation
        disc_real_outputs = real_stft_outputs
        disc_fake_outputs = fake_stft_outputs
        
        # Compute discriminator loss using the DiscriminatorLoss class
        total_disc_loss = self.discriminator_loss_fn(disc_real_outputs, disc_fake_outputs)
        
        total_disc_loss.backward()
        self.stft_disc_optimizer.step()
        
        return {
            'stft_disc_loss': total_disc_loss.item(),
            'total_disc_loss': total_disc_loss.item()
        }
    
    def _generator_step(self, real_audio, fake_audio):
        """Perform generator training step with discriminator feedback."""
        if not self.use_discriminators:
            return {}, {}, {}
        
        # Get discriminator outputs for generator training
        # Real outputs: no gradients needed (only used for feature matching)
        with torch.no_grad():
            real_stft = self._compute_stft(real_audio)
            real_stft_outputs = self.stft_discriminator(real_stft)
        
        # Fake outputs: gradients needed for adversarial loss
        fake_stft = self._compute_stft(fake_audio)
        fake_stft_outputs = self.stft_discriminator(fake_stft)
        
        # return discriminator outputs
        discriminator_real_outputs = real_stft_outputs
        discriminator_fake_outputs = fake_stft_outputs
        
        return discriminator_real_outputs, discriminator_fake_outputs, {}

    def _train_epoch(self, epoch):
        loss_total = 0.0
        disc_loss_total = 0.0
        
        # Discriminator training frequency (train discriminator every N generator steps)
        disc_train_freq = getattr(self, 'disc_train_freq', 1)

        for i, (audio_sig, cov_matrix, name) in enumerate(self.train_data_loader):
            # For now, we only use audio_sig (cov_matrix will be used in future updates)
            audio_sig = audio_sig.to(self.device)  # [B, 4, T]
            
            # Generate reconstructed audio
            self.optimizer.zero_grad()
            reconstructed = self.model(audio_sig)
            
            # Get discriminator outputs for loss computation
            if self.use_discriminators:
                disc_real_outputs, disc_fake_outputs, _ = self._generator_step(audio_sig, reconstructed)
            else:
                disc_real_outputs, disc_fake_outputs = None, None
            
            # Compute generator loss using SoundStream loss
            gen_loss, loss_components = self.loss_function(
                x_real=audio_sig,
                x_fake=reconstructed,
                disc_real_outputs=disc_real_outputs,
                disc_fake_outputs=disc_fake_outputs
            )

            gen_loss.backward()
            self.optimizer.step()
            
            # Train discriminators
            if self.use_discriminators and i % disc_train_freq == 0:
                disc_losses = self._discriminator_step(audio_sig, reconstructed.detach())
                disc_loss_total += disc_losses['total_disc_loss']
                
                # Log discriminator losses
                if i % 100 == 0:  # Log every 100 steps
                    for key, value in disc_losses.items():
                        wandb.log({f"Loss/train_{key}": value}, step=epoch * len(self.train_data_loader) + i)

            loss_total += gen_loss.item()
            
            # Log detailed losses
            if i % 100 == 0 and hasattr(self.loss_function, 'forward'):
                for key, value in loss_components.items():
                    if torch.is_tensor(value):
                        wandb.log({f"Loss/train_{key}": value.item()}, step=epoch * len(self.train_data_loader) + i)
                    else:
                        wandb.log({f"Loss/train_{key}": value}, step=epoch * len(self.train_data_loader) + i)
            
            if i % 100 == 0:
                print(f"Batch {i}, Generator Loss: {gen_loss.item():.6f}")

        dl_len = len(self.train_data_loader)
        avg_gen_loss = loss_total / dl_len
        avg_disc_loss = disc_loss_total / max(1, dl_len // disc_train_freq)
        
        wandb.log({
            "Loss/train": avg_gen_loss,
            "Loss/train_discriminator": avg_disc_loss
        }, step=epoch)
        
        print(f"Epoch {epoch} - Avg Generator Loss: {avg_gen_loss:.6f}, Avg Discriminator Loss: {avg_disc_loss:.6f}")

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0  # total loss for validation
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        for i, (audio_sig, cov_matrix, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            # For now, we only use audio_sig (cov_matrix will be used in future updates)
            audio_sig = audio_sig.to(self.device)  # [1, 4, T]
            # Use same signal as target for current compatibility
            mixture = audio_sig
            clean = audio_sig

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 4, padded_length, device=self.device)], dim=-1)
                clean = torch.cat([clean, torch.zeros(1, 4, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
            clean_chunks = list(torch.split(clean, sample_length, dim=-1))

            enhanced_chunks = []
            for mix_chunk, clean_chunk in zip(mixture_chunks, clean_chunks):
                enhanced_chunk = self.model(mix_chunk)
                
                # Compute loss
                if hasattr(self.loss_function, 'forward'):
                    # SoundStream loss (without discriminator in validation)
                    chunk_loss, _ = self.loss_function(
                        x_real=clean_chunk,
                        x_fake=enhanced_chunk,
                        disc_real_outputs=None,
                        disc_fake_outputs=None
                    )
                else:
                    # Legacy loss function
                    chunk_loss = self.loss_function(clean_chunk, enhanced_chunk)
                
                loss_total += chunk_loss.item()

                enhanced = enhanced_chunk.detach().cpu()
                enhanced_chunks.append(enhanced)

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 4, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            enhanced = enhanced.reshape(-1).numpy()
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced)

        dl_len = len(self.validation_data_loader)
        val_loss_avg = loss_total / dl_len
        print("Loss validation", val_loss_avg)
        wandb.log({"Loss/val": val_loss_avg}, step=epoch)

        return val_loss_avg
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Override to save discriminator checkpoints."""
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()
        
        # Save discriminator states if they exist
        if self.use_discriminators:
            if isinstance(self.stft_discriminator, torch.nn.DataParallel):
                state_dict["stft_discriminator"] = self.stft_discriminator.module.cpu().state_dict()
            else:
                state_dict["stft_discriminator"] = self.stft_discriminator.cpu().state_dict()
            
            state_dict["stft_disc_optimizer"] = self.stft_disc_optimizer.state_dict()

        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        # Move models back to device
        self.model.to(self.device)
        if self.use_discriminators:
            self.stft_discriminator.to(self.device)
    
    def _resume_checkpoint(self):
        """Override to resume discriminator checkpoints."""
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        # Load discriminator states if they exist
        if self.use_discriminators and "stft_discriminator" in checkpoint:
            if isinstance(self.stft_discriminator, torch.nn.DataParallel):
                self.stft_discriminator.module.load_state_dict(checkpoint["stft_discriminator"])
            else:
                self.stft_discriminator.load_state_dict(checkpoint["stft_discriminator"])
            
            self.stft_disc_optimizer.load_state_dict(checkpoint["stft_disc_optimizer"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")
    
    def _set_models_to_train_mode(self):
        """Override to set discriminators to train mode."""
        self.model.train()
        if self.use_discriminators:
            self.stft_discriminator.train()

    def _set_models_to_eval_mode(self):
        """Override to set discriminators to eval mode."""
        self.model.eval()
        if self.use_discriminators:
            self.stft_discriminator.eval()
