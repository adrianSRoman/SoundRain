import os

import librosa
import h5py
import numpy as np
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): Path to dataset. For HDF5: path to .hdf5 file. For legacy: *.txt list file.
            limit (int): Return at most limit files/frames in the list. If None, all files/frames are returned.
            offset (int): Return files/frames starting at an offset within the list. Use negative values to offset from the end.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            HDF5 format contains:
            - audio_frames: (n_frames, frame_samples, n_channels) framed audio data
            - visibility_matrices: (n_frames, nbands, n_channels, n_channels) visibility matrices

        Return:
            For HDF5 format: (framed_audio, visibility_matrices, frame_idx)
        """
        super(Dataset, self).__init__()
        
        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."
        
        self.mode = mode

        # HDF5 format
        self.hdf5_path = os.path.abspath(os.path.expanduser(dataset))
        if not os.path.exists(self.hdf5_path):
            raise ValueError(f"HDF5 file not found: {self.hdf5_path}")
        
        # Load HDF5 metadata to get number of frames
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_frames = f.attrs["n_frames"]
            self.frame_samples = f.attrs["frame_samples"]
            self.n_channels = f.attrs["n_channels"]
            self.n_bands = f.attrs["n_bands"]
            self.sample_rate = f.attrs["sample_rate"]
            
        # Apply offset and limit to frame indices
        self.frame_indices = list(range(self.total_frames))
        self.frame_indices = self.frame_indices[offset:]
        if limit:
            self.frame_indices = self.frame_indices[:limit]
            
        self.length = len(self.frame_indices)
            

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # HDF5 format (simplified - only supports HDF5 now)
        frame_idx = self.frame_indices[item]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load the specific frame
            framed_audio = f["audio_frames"][frame_idx]  # Shape: (frame_samples, n_channels)
            visibility_matrix = f["visibility_matrices"][frame_idx]  # Shape: (nbands, n_channels, n_channels)
            
        # Convert to float32 for consistency
        framed_audio = framed_audio.astype(np.float32)
        visibility_matrix = visibility_matrix.astype(np.complex64)
        
        # Transpose to match expected format (n_channels, frame_samples)
        framed_audio = framed_audio.T

        return framed_audio, visibility_matrix, f"frame_{frame_idx}"
                


class HDF5Dataset(Dataset):
    """
    Convenience class for HDF5 datasets.
    """
    def __init__(self, hdf5_path, limit=None, offset=0, mode="train"):
        super().__init__(
            dataset=hdf5_path,
            limit=limit,
            offset=offset,
            mode=mode
        )
