import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
#from util.utils import compute_STOI, compute_PESQ
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

        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture)
            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            print("Loss ", loss_total)

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 4, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 4, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

        score = 0

        return score
