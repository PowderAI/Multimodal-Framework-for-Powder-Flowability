import time
import torch
from typing import List

def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    val_loader,
    n_warmup: int = 5,
    n_iter: int = 20
) -> List[float]:

    model.eval()
    it = iter(val_loader)
    times = []
    
    with torch.inference_mode():
        for i in range(n_warmup + n_iter):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(val_loader)
                batch = next(it)
            
            images = batch['image'][:1].to(device)
            tabular = batch['tabular'][:1].to(device)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t0 = time.time()
            _ = model(images, tabular)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Record time after warmup
            if i >= n_warmup:
                times.append((time.time() - t0) * 1000)
    
    return times
