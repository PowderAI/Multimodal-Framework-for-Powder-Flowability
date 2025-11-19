import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from typing import Dict, Optional
import wandb

from training.early_stopping import EarlyStopping
from evaluation.metrics import calculate_all_metrics
from utils.helpers import measure_latency

class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            targets = batch['target'].to(self.device)
            flow_avail = batch['flow_rate_available'].to(self.device)
            flow_vals = batch['flow_rate_value'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            orig_pred, flow_class_pred, flow_reg_pred, _ = self.model(images, tabular)
            
            # Calculate losses
            loss_orig = self.mse_loss(orig_pred, targets)
            loss_flow_class = self.ce_loss(flow_class_pred, flow_avail)
            
            # Flow regression loss (only for available samples)
            mask = flow_avail == 1
            if mask.sum() > 0:
                loss_flow_reg = self.mse_loss(
                    flow_reg_pred[mask].squeeze(), 
                    flow_vals[mask]
                )
            else:
                loss_flow_reg = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss_batch = loss_orig + loss_flow_class + loss_flow_reg
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(dataloader)

    def validate(self, dataloader, return_predictions: bool = True):
        self.model.eval()
        total_loss = 0
        
        # Storage for predictions
        all_orig_pred, all_orig_true = [], []
        all_flow_class_pred, all_flow_class_true = [], []
        all_flow_reg_pred, all_flow_reg_true = [], []
        all_tabular, all_indices = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                targets = batch['target'].to(self.device)
                flow_avail = batch['flow_rate_available'].to(self.device)
                flow_vals = batch['flow_rate_value'].to(self.device)
                idx_batch = batch['idx'].numpy()
                
                # Forward pass
                orig_pred, flow_class_pred, flow_reg_pred, _ = self.model(images, tabular)
                
                # Calculate losses
                loss_orig = self.mse_loss(orig_pred, targets)
                loss_flow_class = self.ce_loss(flow_class_pred, flow_avail)
                
                mask = flow_avail == 1
                if mask.sum() > 0:
                    loss_flow_reg = self.mse_loss(
                        flow_reg_pred[mask].squeeze(), 
                        flow_vals[mask]
                    )
                else:
                    loss_flow_reg = torch.tensor(0.0, device=self.device)
                
                total_loss += (loss_orig + loss_flow_class + loss_flow_reg).item()
                
                # Store predictions
                if return_predictions:
                    all_orig_pred.append(orig_pred.cpu().numpy())
                    all_orig_true.append(targets.cpu().numpy())
                    all_flow_class_pred.append(
                        torch.softmax(flow_class_pred, dim=1).cpu().numpy()
                    )
                    all_flow_class_true.append(flow_avail.cpu().numpy())
                    all_flow_reg_pred.append(flow_reg_pred.cpu().numpy().squeeze())
                    all_flow_reg_true.append(flow_vals.cpu().numpy())
                    all_tabular.append(tabular.cpu().numpy())
                    all_indices.append(idx_batch)
        
        avg_loss = total_loss / len(dataloader)
        
        if not return_predictions:
            return avg_loss, None
        
        # Concatenate all predictions
        predictions = {
            'orig_pred': np.vstack(all_orig_pred),
            'orig_true': np.vstack(all_orig_true),
            'flow_class_pred': np.vstack(all_flow_class_pred),
            'flow_class_true': np.concatenate(all_flow_class_true),
            'flow_reg_pred': np.concatenate(all_flow_reg_pred),
            'flow_reg_true': np.concatenate(all_flow_reg_true),
            'tabular': np.concatenate(all_tabular),
            'idx': np.concatenate(all_indices)
        }
        
        return avg_loss, predictions

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        patience: int,
        model_name: str,
        save_dir: str,
        results_dir: str,
        tabular_feature_names: list,
        use_wandb: bool = True
    ) -> Dict:
        
        stopper = EarlyStopping(patience=patience)
        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, predictions = self.validate(val_loader, return_predictions=True)
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Log to WandB
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            
            # Early stopping check
            if stopper.step(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Load best weights
        self.model.load_state_dict(stopper.best_wts)
        
        # Calculate final metrics
        metrics = self._calculate_final_metrics(
            predictions,
            train_loader,
            val_loader,
            train_losses,
            val_losses,
            start_time
        )
        
        # Save model and results
        self._save_results(
            model_name,
            save_dir,
            results_dir,
            metrics,
            predictions,
            tabular_feature_names,
            val_loader,
            use_wandb
        )
        
        return metrics

    def _calculate_final_metrics(
        self,
        predictions,
        train_loader,
        val_loader,
        train_losses,
        val_losses,
        start_time
    ) -> Dict:
        
        metrics = calculate_all_metrics(
            predictions['orig_pred'],
            predictions['orig_true'],
            predictions['flow_class_pred'],
            predictions['flow_class_true'],
            predictions['flow_reg_pred'],
            predictions['flow_reg_true']
        )
        
        # Add training info
        metrics.update({
            "TrainTime(s)": time.time() - start_time,
            "Param": sum(p.numel() for p in self.model.parameters()),
            "TrainSamples": len(train_loader.dataset),
            "ValSamples": len(val_loader.dataset),
            "Backbone": self.model.backbone.name,
            "train_losses": train_losses,
            "val_losses": val_losses
        })
        
        # Measure latency
        latency_times = measure_latency(self.model, self.device, val_loader)
        metrics["Latency_ms_times"] = latency_times
        metrics["Latency_ms_mean"] = np.mean(latency_times)
        metrics["Latency_ms_std"] = np.std(latency_times)
        
        return metrics

    def _save_results(
        self,
        model_name,
        save_dir,
        results_dir,
        metrics,
        predictions,
        tabular_feature_names,
        val_loader,
        use_wandb
    ):
        
        # Save model
        model_path = os.path.join(save_dir, f'best_{model_name}.pth')
        torch.save({'model_state_dict': self.model.state_dict()}, model_path)
        metrics["ModelSize(MB)"] = os.path.getsize(model_path) / (1024 ** 2)
        
        # Save metrics
        joblib.dump(metrics, os.path.join(results_dir, f"{model_name}_metrics.pkl"))
        pd.DataFrame([metrics]).to_csv(
            os.path.join(results_dir, f"{model_name}_metrics.csv"), 
            index=False
        )
        
        # Save predictions
        predictions['tabular_names'] = tabular_feature_names
        joblib.dump(
            predictions, 
            os.path.join(results_dir, f"{model_name}_val_predictions.pkl")
        )
        
        # Create predictions DataFrame
        df_logs = pd.DataFrame({
            'idx': predictions['idx'],
            'AOR_true': predictions['orig_true'][:, 0],
            'HR_true': predictions['orig_true'][:, 1],
            'CI_true': predictions['orig_true'][:, 2],
            'AOR_pred': predictions['orig_pred'][:, 0],
            'HR_pred': predictions['orig_pred'][:, 1],
            'CI_pred': predictions['orig_pred'][:, 2],
            'FlowAvail_true': predictions['flow_class_true'],
            'FlowAvail_prob0': predictions['flow_class_pred'][:, 0],
            'FlowAvail_prob1': predictions['flow_class_pred'][:, 1],
            'FlowRate_true': predictions['flow_reg_true'],
            'FlowRate_pred': predictions['flow_reg_pred'],
        })
        
        # Add tabular features
        tabular_arr = predictions['tabular']
        if tabular_arr.ndim == 2 and len(tabular_feature_names) == tabular_arr.shape[1]:
            tabular_df = pd.DataFrame(tabular_arr, columns=tabular_feature_names)
            df_logs = pd.concat([df_logs, tabular_df], axis=1)
        
        df_logs.to_csv(
            os.path.join(results_dir, f"{model_name}_val_predictions.csv"), 
            index=False
        )
        
        # Upload to WandB
        if use_wandb:
            try:
                wandb.save(model_path, policy="now")
            except Exception as e:
                print(f"Warning: Could not upload to WandB: {e}")
