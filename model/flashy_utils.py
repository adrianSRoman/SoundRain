"""
Simple flashy utilities for the balancer.
This provides the minimal flashy functionality needed for the balancer to work.
"""
import typing as tp
import torch


class EMADict:
    """Exponential Moving Average for dictionaries of tensors."""
    
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.state: tp.Dict[str, torch.Tensor] = {}
        
    def __call__(self, new_values: tp.Dict[str, torch.Tensor]) -> tp.Dict[str, torch.Tensor]:
        """Update EMA and return current averaged values."""
        if not self.state:
            # First call - initialize state
            self.state = {k: v.clone().detach() for k, v in new_values.items()}
            return self.state.copy()
        
        # Update EMA
        for key, new_val in new_values.items():
            if key in self.state:
                self.state[key] = self.decay * self.state[key] + (1 - self.decay) * new_val.detach()
            else:
                self.state[key] = new_val.clone().detach()
                
        return self.state.copy()


def averager(decay: float = 0.999) -> EMADict:
    """Create an EMA averager."""
    return EMADict(decay)


class distrib:
    """Distributed utilities stub."""
    
    @staticmethod
    def average_metrics(metrics: tp.Dict[str, torch.Tensor], count: int = 1) -> tp.Dict[str, torch.Tensor]:
        """
        Average metrics across workers. 
        In single-GPU setup, this just returns the input metrics.
        """
        # In a real distributed setup, this would use torch.distributed
        # For now, we just return the metrics as-is
        return metrics
