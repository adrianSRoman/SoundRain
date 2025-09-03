#!/usr/bin/env python3
"""
SoundRain Inference Script

This script performs inference on audio files using a pre-trained SoundRain model.
It applies windowing and overlap-add to prevent discontinuities between frames.

Usage:
    python inference.py --input input.wav --output output.wav --checkpoint checkpoints/best_model.tar [options]
    
Example:
    python inference.py --input test_audio.wav --output enhanced_audio.wav --checkpoint train/checkpoints/best_model.tar --config config/train/soundstream_train.json
"""

import argparse
import os
import json5
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly, windows

from util.utils import initialize_config, load_checkpoint


def load_model(config_path, checkpoint_path, device):
    """Load the SoundRain model from config and checkpoint."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json5.load(f)
    
    # Initialize model
    model = initialize_config(config["model"])
    
    # Load checkpoint
    if checkpoint_path.endswith('.tar'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint["model"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:  # .pth file
        model_state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded model weights from {checkpoint_path}")
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def apply_window(audio_chunk, window_type='hann', fade_samples=128):
    """Apply windowing to audio chunk to reduce artifacts."""
    chunk_length = audio_chunk.shape[-1]
    
    if window_type == 'hann':
        # Create Hann window using scipy.signal
        window = windows.hann(chunk_length)
    elif window_type == 'hamming':
        # Create Hamming window using scipy.signal
        window = windows.hamming(chunk_length)
    elif window_type == 'fade':
        # Simple fade-in/fade-out
        window = np.ones(chunk_length)
        fade_samples = min(fade_samples, chunk_length // 2)  # Ensure fade doesn't exceed half the chunk
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        window[:fade_samples] = fade_in
        window[-fade_samples:] = fade_out
    else:
        # No windowing
        return audio_chunk
    
    # Apply window to all channels
    if audio_chunk.ndim == 2:  # [channels, samples]
        window = window[np.newaxis, :]  # Broadcast to all channels
    elif audio_chunk.ndim == 3:  # [batch, channels, samples]
        window = window[np.newaxis, np.newaxis, :]  # Broadcast to batch and channels
    
    return audio_chunk * window


def overlap_add(chunks, hop_length, total_length):
    """Reconstruct audio from overlapping chunks using overlap-add."""
    if not chunks:
        return np.zeros((4, total_length))  # 4-channel audio
    
    # Get dimensions
    n_channels = chunks[0].shape[0] if chunks[0].ndim == 2 else chunks[0].shape[1]
    chunk_length = chunks[0].shape[-1]
    
    # Initialize output
    output = np.zeros((n_channels, total_length))
    
    # Add each chunk to the output
    for i, chunk in enumerate(chunks):
        start_sample = i * hop_length
        end_sample = min(start_sample + chunk_length, total_length)
        
        # Handle chunk dimensions
        if chunk.ndim == 3:  # [batch, channels, samples]
            chunk = chunk[0]  # Remove batch dimension
        
        # Add chunk to output (overlap-add)
        chunk_end = end_sample - start_sample
        output[:, start_sample:end_sample] += chunk[:, :chunk_end]
    
    return output


def process_audio(model, audio, sample_rate=24000, chunk_length=16000, overlap=0.25, 
                 window_type='hann', device='cpu'):
    """
    Process audio through the model with windowing and overlap.
    
    Args:
        model: Trained SoundRain model
        audio: Input audio array [channels, samples]
        sample_rate: Audio sample rate
        chunk_length: Length of each processing chunk in samples
        overlap: Overlap ratio (0.0 to 1.0)
        window_type: Type of window to apply ('hann', 'hamming', 'fade', 'none')
        device: Device to run inference on
    
    Returns:
        Enhanced audio array [channels, samples]
    """
    
    # Convert to tensor and add batch dimension
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()
    
    # Ensure correct format: [channels, samples]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
    
    # Pad to 4 channels if needed (SoundRain expects 4-channel input)
    if audio_tensor.shape[0] < 4:
        padding = torch.zeros(4 - audio_tensor.shape[0], audio_tensor.shape[1])
        audio_tensor = torch.cat([audio_tensor, padding], dim=0)
    elif audio_tensor.shape[0] > 4:
        audio_tensor = audio_tensor[:4]  # Take first 4 channels
    
    total_samples = audio_tensor.shape[1]
    hop_length = int(chunk_length * (1 - overlap))
    
    # Calculate number of chunks needed
    n_chunks = (total_samples + hop_length - 1) // hop_length
    
    enhanced_chunks = []
    
    print(f"Processing {n_chunks} chunks of length {chunk_length} with {overlap*100:.1f}% overlap...")
    
    with torch.no_grad():
        for i in range(n_chunks):
            # Extract chunk
            start_sample = i * hop_length
            end_sample = min(start_sample + chunk_length, total_samples)
            
            # Pad chunk if it's shorter than expected
            chunk = audio_tensor[:, start_sample:end_sample]
            if chunk.shape[1] < chunk_length:
                padding = torch.zeros(4, chunk_length - chunk.shape[1])
                chunk = torch.cat([chunk, padding], dim=1)
            
            # # Apply windowing to reduce artifacts
            # if window_type != 'none':
            #     chunk_windowed = apply_window(chunk.numpy(), window_type)
            #     chunk = torch.from_numpy(chunk_windowed).float()
            
            # Add batch dimension and move to device
            chunk_batch = chunk.unsqueeze(0).to(device)
            print(chunk_batch.shape)
            # Process through model
            enhanced_chunk = model(chunk_batch)
            
            # Remove batch dimension and move to CPU
            enhanced_chunk = enhanced_chunk.squeeze(0).cpu()
            
            # # Apply window to output as well for smooth blending
            if window_type != 'none':
                enhanced_chunk_windowed = apply_window(enhanced_chunk.numpy(), window_type)
                enhanced_chunk = torch.from_numpy(enhanced_chunk_windowed)
            
            enhanced_chunks.append(enhanced_chunk.numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_chunks} chunks")
    
    # Reconstruct full audio using overlap-add
    enhanced_audio = overlap_add(enhanced_chunks, hop_length, total_samples)
    
    return enhanced_audio


def main():
    parser = argparse.ArgumentParser(description="SoundRain Audio Inference")
    parser.add_argument("--input", "-i", required=True, help="Input audio file path")
    parser.add_argument("--output", "-o", required=True, help="Output audio file path")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path (.tar or .pth)")
    parser.add_argument("--config", required=False, help="Model configuration file (required for .pth checkpoints)")
    parser.add_argument("--chunk-length", type=int, default=16000, help="Processing chunk length in samples (default: 16000)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio between chunks (default: 0.25)")
    parser.add_argument("--window", choices=['hann', 'hamming', 'fade', 'none'], default='hann', 
                       help="Window function to apply (default: hann)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Audio sample rate (default: 24000)")
    parser.add_argument("--device", default="auto", help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--output-channels", type=int, default=None, 
                       help="Number of output channels to save (default: same as input, max 4)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Determine config file
    config_path = args.config
    if config_path is None:
        if args.checkpoint.endswith('.tar'):
            # Try to find config file in same directory or parent directory
            checkpoint_dir = Path(args.checkpoint).parent
            possible_configs = [
                checkpoint_dir / "config.json",
                checkpoint_dir.parent / "config.json",
                checkpoint_dir.parent.parent / "config" / "train" / "soundstream_train.json"
            ]
            for config_file in possible_configs:
                if config_file.exists():
                    config_path = str(config_file)
                    print(f"Using config file: {config_path}")
                    break
            
            if config_path is None:
                raise ValueError("Could not find config file. Please specify with --config")
        else:
            raise ValueError("Config file is required for .pth checkpoints. Please specify with --config")
    
    # Check if files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load model
    print("Loading model...")
    model, config = load_model(config_path, args.checkpoint, device)
    print("Model loaded successfully")
    
    # Load audio
    print(f"Loading audio from {args.input}...")
    audio, sr = librosa.load(args.input, sr=None, mono=False)
    # resample to 48kHz with scipy
    if sr != 24000:
        # Compute up/down factors
        audio = librosa.resample(y=audio.T, orig_sr=sr, target_sr=24000).T
        sr = 24000
    
    # Ensure audio is 2D [channels, samples]
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # Add channel dimension
    
    original_channels = audio.shape[0]
    print(f"Loaded audio: {audio.shape[0]} channels, {audio.shape[1]} samples, {sr} Hz")
    
    # Process audio
    print("Starting inference...")
    enhanced_audio = process_audio(
        model=model,
        audio=audio,
        sample_rate=48000,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        window_type=args.window,
        device=device
    )
    
    # Determine output channels
    if args.output_channels is not None:
        output_channels = min(args.output_channels, enhanced_audio.shape[0])
    else:
        output_channels = min(original_channels, enhanced_audio.shape[0])
    
    # Take only the required number of channels
    enhanced_audio = enhanced_audio[:output_channels]
    
    # Convert to mono if single channel
    if enhanced_audio.shape[0] == 1:
        enhanced_audio = enhanced_audio[0]
    
    # Save output
    print(f"Saving enhanced audio to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sf.write(args.output, enhanced_audio.T, 24000)  # soundfile expects [samples, channels]
    
    print("Inference completed successfully!")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Processed {audio.shape[1]} samples in {enhanced_audio.shape[0]} channels")


if __name__ == "__main__":
    main()
