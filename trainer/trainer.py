import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from trainer.base_trainer import BaseTrainer
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

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (audio_sig, cov_matrix, name) in enumerate(self.train_data_loader):
            # For now, we only use audio_sig (cov_matrix will be used in future updates)
            # Treat audio_sig as both mixture and clean for current compatibility
            audio_sig = audio_sig.to(self.device) # [1, 4, T]

            self.optimizer.zero_grad()
            enhanced = self.model(audio_sig)
            # For now, use the same audio_sig as target (this will need to be updated when you have proper targets)
            loss = self.loss_function(audio_sig, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            print("Loss ", loss_total)

        dl_len = len(self.train_data_loader)
        wandb.log({"Loss/train": loss_total / dl_len}, step=epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0 # total loss for validation
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
            # Use same signal as both mixture and clean for current compatibility
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
                loss = self.loss_function(clean_chunk, enhanced_chunk)
                loss_total += loss.item()

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
