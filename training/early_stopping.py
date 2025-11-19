from typing import Dict, Optional
import torch

class EarlyStopping:
    
    def __init__(self, patience: int = 5, delta: float = 1e-4):

        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.best_wts: Optional[Dict] = None
        self.counter = 0

    def step(self, loss: float, model: torch.nn.Module) -> bool:

        if loss < self.best_loss - self.delta:
            # Improvement detected
            self.best_loss = loss
            self.best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            return self.counter >= self.patience
