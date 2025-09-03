import os
import glob
import h5py
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample_poly
from typing import List, Tuple, Optional
import logging

from util.utils import get_visibility_matrix

import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioFramer:
    """
    Utility class for framing audio into fixed-length windows.
    """
    def __init__(self, frame_duration_ms: float = 100.0):
        """
        Initialize the audio framer.
        
        Args:
            frame_duration_ms: Frame duration in milliseconds
        """
        self.frame_duration_ms = frame_duration_ms
        
    def frame_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Frame audio data into fixed-length windows.
        
        Args:
            audio_data: Audio data of shape (n_samples, n_channels)
            sample_rate: Sample rate in Hz
            
        Returns:
            Framed audio data of shape (n_frames, frame_samples, n_channels)
        """
        frame_samples = int(self.frame_duration_ms * sample_rate / 1000.0)
        n_samples, n_channels = audio_data.shape
        
        # Calculate number of complete frames (including potential partial frame)
        n_complete_frames = n_samples // frame_samples
        remainder_samples = n_samples % frame_samples
        
        if n_complete_frames == 0:
            logging.warning(f"Audio too short for framing. Length: {n_samples} samples, "
                          f"Required: {frame_samples} samples per frame")
            return np.array([]).reshape(0, frame_samples, n_channels)
        
        # Add padding if there are remaining samples to create a complete frame
        if remainder_samples > 0:
            n_frames = n_complete_frames + 1
            padding_samples = frame_samples - remainder_samples
            # Pad with zeros at the end
            padding = np.zeros((padding_samples, n_channels), dtype=audio_data.dtype)
            padded_audio = np.concatenate([audio_data, padding], axis=0)
            logging.info(f"Added {padding_samples} zero-padding samples to create complete frame")
        else:
            n_frames = n_complete_frames
            padded_audio = audio_data
        
        # Reshape into frames
        framed_audio = padded_audio.reshape(n_frames, frame_samples, n_channels)
        
        if remainder_samples > 0:
            logging.info(f"Framed audio: {n_samples} samples -> {n_frames} frames of {frame_samples} samples each "
                        f"(last frame padded with {padding_samples} zeros)")
        else:
            logging.info(f"Framed audio: {n_samples} samples -> {n_frames} frames of {frame_samples} samples each")
        return framed_audio


class VisibilityMatrixProcessor:
    """
    Processor for computing visibility matrices from framed audio.
    """
    def __init__(self, frame_duration_ms: float = 100.0, scale: str = "log", nbands: int = 16):
        """
        Initialize the visibility matrix processor.
        
        Args:
            frame_duration_ms: Frame duration in milliseconds
            scale: Scale for frequency bands ("log" or "linear")
            nbands: Number of frequency bands
        """
        self.frame_duration_ms = frame_duration_ms
        self.scale = scale
        self.nbands = nbands
        
    def process_audio_file(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process audio file to extract visibility matrices and corresponding audio frames.
        
        Args:
            audio_data: Audio data of shape (n_samples, n_channels)
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (framed_audio, visibility_matrices)
            - framed_audio: shape (n_frames, frame_samples, n_channels)
            - visibility_matrices: shape (n_frames, nbands, n_channels, n_channels)
        """
        T_sti = self.frame_duration_ms / 1000.0  # Convert to seconds
        
        # Get visibility matrices using the existing function
        # Note: get_visibility_matrix returns shape (nbands, n_frames, n_channels, n_channels)
        visibility_matrices = get_visibility_matrix(
            audio_data, sample_rate, T_sti=T_sti, scale=self.scale, nbands=self.nbands
        )
        
        # Frame the audio data
        framer = AudioFramer(self.frame_duration_ms)
        framed_audio = framer.frame_audio(audio_data, sample_rate)
        
        # Transpose visibility matrices to match framed audio: (n_frames, nbands, n_channels, n_channels)
        visibility_matrices = visibility_matrices.transpose(1, 0, 2, 3)
        
        # Ensure frame counts match
        min_frames = min(framed_audio.shape[0], visibility_matrices.shape[0])
        framed_audio = framed_audio[:min_frames]
        visibility_matrices = visibility_matrices[:min_frames]
        
        logging.info(f"Processed audio: {framed_audio.shape[0]} frames, "
                    f"Visibility matrices: {visibility_matrices.shape}")
        
        return framed_audio, visibility_matrices


class HDF5DatasetGenerator:
    """
    Main class for generating HDF5 datasets from audio files.
    """
    def __init__(self, frame_duration_ms: float = 100.0, scale: str = "log", nbands: int = 16):
        """
        Initialize the HDF5 dataset generator.
        
        Args:
            frame_duration_ms: Frame duration in milliseconds
            scale: Scale for frequency bands ("log" or "linear")
            nbands: Number of frequency bands
        """
        self.processor = VisibilityMatrixProcessor(frame_duration_ms, scale, nbands)
        
    def _get_audio_files(self, data_src: str, split: str) -> List[str]:
        """
        Get list of valid audio files from source directory.
        
        Args:
            data_src: Source directory path
            split: Data split (e.g., "train", "val", "test")

        Returns:
            List of audio file paths
        """
        if not os.path.exists(data_src):
            raise ValueError(f"Source directory does not exist: {data_src}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")

        # audio_files = [
        #     os.path.join(data_src, file) 
        #     for file in os.listdir(data_src) 
        #     if file.endswith('.wav') and not file.startswith('._')
        # ]

        # convert string path to Path object and use glob
        data_path = Path(data_src)
        audio_files = [str(file) for file in data_path.glob('mic_dev/**/*.wav') if split in str(file)]

        if not audio_files:
            raise ValueError(f"No valid audio files found in {data_src}")
            
        logging.info(f"Found {len(audio_files)} audio files in {data_src}")
        return audio_files
        
    def _process_single_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (framed_audio, visibility_matrices, sample_rate)
        """
        try:
            audio_data, sample_rate = sf.read(file_path)
            # resample to 48kHz with librosa
            if sample_rate != 24000:
                audio_data = librosa.resample(y=audio_data.T, orig_sr=sample_rate, target_sr=24000).T
                sample_rate = 24000
            sf.write(f"./{os.path.basename(file_path)}", audio_data, 24000)

            if audio_data.shape[-1] == 32:
                audio_data = audio_data[:, [5,9,25,21]] # 4 ch raw MIC
            
            # Ensure audio is 2D (samples, channels)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
                
            framed_audio, visibility_matrices = self.processor.process_audio_file(
                audio_data, sample_rate
            )
            
            logging.info(f"Processed {file_path}: {framed_audio.shape[0]} frames, {framed_audio.shape[-1]} channels")
            return framed_audio, visibility_matrices, sample_rate
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            raise

    def generate_dataset(self, dataset_name: str, data_src: str, split: str, save_path: str) -> None:
        """
        Generate HDF5 dataset from audio files.
        
        Args:
            dataset_name: Name of the dataset
            data_src: Source directory containing audio files
            save_path: Directory to save the HDF5 file
        """
        if not data_src or not save_path:
            raise ValueError("Both data_src and save_path must be provided")
            
        # Get audio files
        audio_files = self._get_audio_files(data_src, split)

        # Initialize data containers
        all_framed_audio = []
        all_visibility_matrices = []
        sample_rates = []
        
        # Process each file
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            framed_audio, visibility_matrices, sample_rate = self._process_single_file(file_path)

            print("Shapes:")
            print(f"Framed audio: {framed_audio.shape}")
            print(f"Visibility matrices: {visibility_matrices.shape}")

            if framed_audio.shape[0] == visibility_matrices.shape[0]:  # Only add if frames were created
                all_framed_audio.append(framed_audio)
                all_visibility_matrices.append(visibility_matrices)
                sample_rates.append(sample_rate)
        
        if not all_framed_audio:
            raise ValueError("No valid frames were generated from any audio files")

        # Concatenate all data
        final_audio = np.concatenate(all_framed_audio, axis=0)
        final_visibility = np.concatenate(all_visibility_matrices, axis=0)
        
        # Verify sample rates are consistent
        unique_rates = list(set(sample_rates))
        if len(unique_rates) > 1:
            logging.warning(f"Multiple sample rates found: {unique_rates}. Using first: {unique_rates[0]}")
        final_sample_rate = unique_rates[0]
        
        # Save to HDF5
        self._save_to_hdf5(dataset_name, save_path, final_audio, final_visibility, final_sample_rate)
        
        logging.info(f"Dataset generation complete. Final shapes:")
        logging.info(f"Audio: {final_audio.shape}")
        logging.info(f"Visibility matrices: {final_visibility.shape}")
        
    def _save_to_hdf5(self, dataset_name: str, save_path: str, 
                      framed_audio: np.ndarray, visibility_matrices: np.ndarray, 
                      sample_rate: int) -> None:
        """
        Save data to HDF5 file.
        
        Args:
            dataset_name: Name of the dataset
            save_path: Directory to save the file
            framed_audio: Framed audio data
            visibility_matrices: Visibility matrices
            sample_rate: Sample rate
        """
        os.makedirs(save_path, exist_ok=True)
        hdf5_path = os.path.join(save_path, f"{dataset_name}.hdf5")
        
        with h5py.File(hdf5_path, 'w') as f:
            # Create datasets
            f.create_dataset(
                "audio_frames", 
                data=framed_audio, 
                compression='gzip', 
                compression_opts=9
            )
            
            f.create_dataset(
                "visibility_matrices", 
                data=visibility_matrices, 
                compression='gzip', 
                compression_opts=9
            )
            
            # Add metadata
            f.attrs["sample_rate"] = sample_rate
            f.attrs["n_frames"] = framed_audio.shape[0]
            f.attrs["frame_samples"] = framed_audio.shape[1]
            f.attrs["n_channels"] = framed_audio.shape[2]
            f.attrs["n_bands"] = visibility_matrices.shape[1]
            f.attrs["frame_duration_ms"] = self.processor.frame_duration_ms
            f.attrs["scale"] = self.processor.scale
            
            # Verify data integrity
            assert framed_audio.shape[0] == visibility_matrices.shape[0], \
                "Frame count mismatch between audio and visibility matrices"
                
        logging.info(f"HDF5 file saved to: {hdf5_path}")


def create_full_hdf_data(dataset_name: str = 'train', data_src: Optional[str] = None, split: str = 'train',
                        save_path: Optional[str] = None, frame_duration_ms: float = 100.0,
                        scale: str = "log", nbands: int = 16) -> None:
    """
    Create HDF5 dataset with framed audio and visibility matrices.
    
    Args:
        dataset_name: Name of the dataset
        data_src: Source directory containing audio files
        save_path: Directory to save the HDF5 file
        frame_duration_ms: Frame duration in milliseconds
        scale: Scale for frequency bands ("log" or "linear")
        nbands: Number of frequency bands
    """
    generator = HDF5DatasetGenerator(frame_duration_ms, scale, nbands)
    generator.generate_dataset(dataset_name, data_src, split, save_path)


def main():
    """Main function to run the dataset generation."""
    parser = argparse.ArgumentParser(description="Generate HDF5 dataset for SoundRain training.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to be processed.",
    )
    parser.add_argument(
        "--data_src",
        type=str,
        required=True,
        help="Path to the source data directory containing .wav files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split to use (train, val, test). Default: train",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data_hdf",
        help="Directory where the HDF5 file will be saved. Default: data_hdf",
    )
    parser.add_argument(
        "--frame_duration_ms",
        type=float,
        default=100.0,
        help="Frame duration in milliseconds. Default: 100.0",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="Scale for frequency bands. Default: log",
    )
    parser.add_argument(
        "--nbands",
        type=int,
        default=16,
        help="Number of frequency bands. Default: 16",
    )

    args = parser.parse_args()

    try:
        create_full_hdf_data(
            dataset_name=args.dataset_name,
            data_src=args.data_src,
            split=args.split,
            save_path=args.save_path,
            frame_duration_ms=args.frame_duration_ms,
            scale=args.scale,
            nbands=args.nbands
        )
        logging.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logging.error(f"Dataset generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()