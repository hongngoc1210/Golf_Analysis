from torch.utils.data import DataLoader, TensorDataset
from src.models.nam import NAMClassifier,  NAMBinaryLoss
from src.models.trainer import NAMTrainerClassification
import pandas as pd
import torch
import numpy as np
import argparse
from src.utils.nam_export import export_nam_feature_weights
from src.utils.load_config import load_config, resolve_device
import json
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NAM Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nam_classifier.yaml",
        help="Path to config file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(cfg.get("runtime", {}))

    # --------------------------------------------------
    # Load processed Stage 2 data
    # --------------------------------------------------
    train_df = pd.read_csv("datasets/processed/train_stage2.csv")
    val_df   = pd.read_csv("datasets/processed/val_stage2.csv")

    feature_cols = [c for c in train_df.columns if c != "target"]
    
    feature_list_path = Path("outputs/models/nam_classifier_features.json")
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(feature_list_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"✅ Saved feature list ({len(feature_cols)} features)")

    X_train = torch.tensor(
        train_df[feature_cols].values,
        dtype=torch.float32
    )
    y_train = torch.tensor(
        train_df["target"].values,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        val_df[feature_cols].values,
        dtype=torch.float32
    )
    y_val = torch.tensor(
        val_df["target"].values,
        dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    # --------------------------------------------------
    # Build NAM model
    # --------------------------------------------------
    features_cols = [c for c in train_df.columns if c!= "target"]
    model = NAMClassifier(
        num_features=len(features_cols),
        hidden_units=cfg["model"]["hidden_units"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # --------------------------------------------------
    # Loss & optimizer
    # --------------------------------------------------
    loss_fn = NAMBinaryLoss(
        l2_lambda=cfg["loss"]["l2_lambda"],
        contrib_lambda=cfg["loss"]["contrib_lambda"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = NAMTrainerClassification(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        threshold=0.5,
        early_stopping_patience=cfg["training"]["early_stopping_patience"]
    )

    trainer.fit(
        num_epochs=cfg["training"]["num_epochs"],
        save_path="outputs/models/nam_classifier_best.pth"
    )
    
    export_nam_feature_weights(
    model=model,
    feature_names=feature_cols,
    save_path="outputs/models/nam_classifier_weights.json"
   )



if __name__ == "__main__":
    main()
