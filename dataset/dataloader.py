import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSample

class AudioDataset(Dataset):
    """
    Dataset for loading audio files for SoundRain training.
    Handles multiple audio formats and preprocessing.
    """
    
    def __init__(
        self,
        audio_dirs: List[str],
        segment_length: int = 16384,  # ~1 second at 16kHz
        sample_rate: int = 16000,
        hop_length: Optional[int] = None,
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
        normalize: bool = True,
        augment: bool = False,
        cache_limit: int = 1000
    ):
        self.audio_dirs = audio_dirs if isinstance(audio_dirs, list) else [audio_dirs]
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length or segment_length // 4
        self.extensions = extensions
        self.normalize = normalize
        self.augment = augment
        self.cache_limit = cache_limit
        
        # Build file list
        self.audio_files = []
        for audio_dir in self.audio_dirs:
            audio_dir = Path(audio_dir)
            for ext in extensions:
                self.audio_files.extend(list(audio_dir.rglob(f'*{ext}')))
        
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Cache for frequently accessed files
        self.audio_cache = {}
        
        # Audio transforms
        if augment:
            self.noise_transform = AddGaussianNoise(std=0.01)
            self.pitch_shift = PitchShift(sample_rate, n_steps_range=(-2, 2))
        
    def __len__(self):
        return len(self.audio_files) * 10  # Multiple segments per file
    
    def __getitem__(self, idx):
        # Map index to file and segment
        file_idx = idx % len(self.audio_files)
        segment_idx = idx // len(self.audio_files)
        
        audio_file = self.audio_files[file_idx]
        
        # Load audio (with caching)
        if str(audio_file) not in self.audio_cache or len(self.audio_cache) > self.cache_limit:
            if len(self.audio_cache) > self.cache_limit:
                # Remove oldest entries
                keys_to_remove = list(self.audio_cache.keys())[:len(self.audio_cache) - self.cache_limit + 1]
                for key in keys_to_remove:
                    del self.audio_cache[key]
            
            try:
                waveform, sr = torchaudio.load(audio_file)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    waveform = resample(waveform, sr, self.sample_rate)
                
                self.audio_cache[str(audio_file)] = waveform
                
            except Exception as e:
                logger.warning(f"Failed to load {audio_file}: {e}")
                # Return silence as fallback
                waveform = torch.zeros(1, self.segment_length)
        else:
            waveform = self.audio_cache[str(audio_file)]
        
        # Extract segment
        if waveform.shape[1] < self.segment_length:
            # Pad if too short
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            # Random crop if longer
            max_start = waveform.shape[1] - self.segment_length
            if max_start > 0:
                start = random.randint(0, max_start)
                waveform = waveform[:, start:start + self.segment_length]
        
        # Normalize
        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Augmentation
        if self.augment and random.random() < 0.5:
            if hasattr(self, 'noise_transform') and random.random() < 0.3:
                waveform = self.noise_transform(waveform)
            if hasattr(self, 'pitch_shift') and random.random() < 0.2:
                waveform = self.pitch_shift(waveform)
        
        return waveform.squeeze(0)  # Return (segment_length,)

class AddGaussianNoise:
    """Add Gaussian noise for data augmentation."""
    
    def __init__(self, std=0.01):
        self.std = std
    
    def __call__(self, waveform):
        noise = torch.randn_like(waveform) * self.std
        return waveform + noise

class PitchShift:
    """Simple pitch shifting using resampling."""
    
    def __init__(self, sample_rate, n_steps_range=(-2, 2)):
        self.sample_rate = sample_rate
        self.n_steps_range = n_steps_range
    
    def __call__(self, waveform):
        n_steps = random.uniform(*self.n_steps_range)
        rate_change = 2 ** (n_steps / 12)
        new_sr = int(self.sample_rate * rate_change)
        
        # Resample to change pitch
        waveform = resample(waveform, self.sample_rate, new_sr)
        waveform = resample(waveform, new_sr, self.sample_rate)
        
        return waveform
