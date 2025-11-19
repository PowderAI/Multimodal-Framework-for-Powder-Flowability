import os
import pandas as pd
import wandb
from configs.config import (
    TrainingConfig, DataConfig, ModelConfig, PathConfig, WandbConfig
)
from data.dataloader import create_dataloaders
from models.multimodal import MultiModalModel
from training.trainer import Trainer

def main():
    # Load configurations
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    path_cfg = PathConfig()
    wandb_cfg = WandbConfig()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=data_cfg.train_csv,
        val_csv=data_cfg.val_csv,
        test_csv=data_cfg.test_csv,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        image_size=train_cfg.image_size
    )
    
    # Train each backbone
    all_val_metrics = []
    
    for backbone in model_cfg.backbones:
        print(f"\n{'='*50}")
        print(f"Training {backbone}")
        print(f"{'='*50}\n")
        
        # Initialize WandB
        if wandb_cfg.enabled:
            run = wandb.init(
                project=wandb_cfg.project,
                name=f"{backbone}_run",
                config={
                    "backbone": backbone,
                    "batch_size": train_cfg.batch_size,
                    "epochs": train_cfg.epochs,
                    "lr": train_cfg.learning_rate,
                    "weight_decay": train_cfg.weight_decay
                },
                reinit=True
            )
        
        # Create model
        model = MultiModalModel(
            tabular_dim=model_cfg.tabular_dim,
            backbone_name=backbone
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=train_cfg.device,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )
        
        # Train
        metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_cfg.epochs,
            patience=train_cfg.patience,
            model_name=f"MultiModal_{backbone.upper()}",
            save_dir=str(path_cfg.model_save_dir),
            results_dir=str(path_cfg.results_dir),
            tabular_feature_names=[data_cfg.magnif_col],
            use_wandb=wandb_cfg.enabled
        )
        
        all_val_metrics.append(metrics)
        
        if wandb_cfg.enabled:
            wandb.log(metrics)
            run.finish()
    
    # Save comparison results
    pd.DataFrame(all_val_metrics).to_csv(
        os.path.join(str(path_cfg.results_dir), "full_val_metrics_all_backbones.csv"),
        index=False
    )


if __name__ == "__main__":
    main()
