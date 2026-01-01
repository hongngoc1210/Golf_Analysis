"""
Training script for NAM model (NaN-safe version)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import json
from pathlib import Path
import argparse

from src.models.nam import NAM, NAMLoss
from src.utils.metrics import compute_metrics
from src.utils.load_config import load_config, resolve_device


# ============================================================
# Utility: NaN / Inf checks
# ============================================================

def assert_no_nan_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"❌ NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"❌ Inf detected in {name}")


def drop_zero_variance_features(df, feature_names):
    stds = df[feature_names].std()
    zero_var = stds[stds == 0].index.tolist()

    if zero_var:
        print("⚠️ Dropping zero-variance features:", zero_var)

    df = df.drop(columns=zero_var)
    feature_names = [f for f in feature_names if f not in zero_var]

    return df.reset_index(drop=True), feature_names



# ============================================================
# Trainer
# ============================================================

class NAMTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = resolve_device(config.get("runtime", {}))

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        path_cfg = config.get("paths", {})

        # Model
        self.model = NAM(
            num_features=model_cfg.get("num_features", 17),
            hidden_units=model_cfg.get("hidden_units", [64, 32]),
            dropout=model_cfg.get("dropout", 0.1)
        ).to(self.device)

        # Loss
        self.criterion = NAMLoss(
            l2_lambda=loss_cfg.get("l2_lambda", 0.0),
            output_lambda=loss_cfg.get("output_lambda", 0.0)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.3,
            patience=10
        )

        self.batch_size = train_cfg.get("batch_size", 16)
        self.num_epochs = train_cfg.get("num_epochs", 100)
        self.early_stop = train_cfg.get("early_stopping_patience", 30)

        self.checkpoint_dir = path_cfg.get("checkpoint_dir")
        self.best_model_path = path_cfg.get("best_model_path")

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------

    def load_data(self):
        """
        Load preprocessed data (NaN-safe, alignment-safe, float32-safe)
        """
        print("📂 Loading preprocessed data...")

        # --------------------------------------------------
        # 1. Load CSV
        # --------------------------------------------------
        train_df = pd.read_csv("datasets/processed/train.csv")
        val_df   = pd.read_csv("datasets/processed/val.csv")
        test_df  = pd.read_csv("datasets/processed/test.csv")

        # Reset index: BẮT BUỘC
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        print(train_df.index[:10])  # DEBUG: phải là RangeIndex

        # --------------------------------------------------
        # 2. Load feature names
        # --------------------------------------------------
        with open("datasets/feature_definitions.json", "r") as f:
            feature_defs = json.load(f)["features"]
        feature_names = [f["name"] for f in feature_defs]

        # --------------------------------------------------
        # 3. Drop zero-variance features (rất quan trọng)
        # --------------------------------------------------
        stds = train_df[feature_names].std()
        zero_var_features = stds[stds == 0].index.tolist()

        if zero_var_features:
            print("⚠️ Dropping zero-variance features:", zero_var_features)
            feature_names = [f for f in feature_names if f not in zero_var_features]
            train_df = train_df.drop(columns=zero_var_features)
            val_df   = val_df.drop(columns=zero_var_features)
            test_df  = test_df.drop(columns=zero_var_features)

        # Reset index lại lần nữa cho chắc
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # --------------------------------------------------
        # 4. Extract FEATURES (X)
        # --------------------------------------------------
        X_train = train_df[feature_names].values.astype(np.float32)
        X_val   = val_df[feature_names].values.astype(np.float32)
        X_test  = test_df[feature_names].values.astype(np.float32)

        # Replace non-finite in X (phòng ngừa)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

        # --------------------------------------------------
        # 5. Extract TARGET (y) – FIX DỨT ĐIỂM NaN
        # --------------------------------------------------
        def clean_target(series):
            y = pd.to_numeric(series, errors="coerce")
            y = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=0.0)
            y = np.clip(y, 0.0, 10.0)
            return y.astype(np.float32)

        y_train = clean_target(train_df["score"])
        y_val   = clean_target(val_df["score"])
        y_test  = clean_target(test_df["score"])

        # --------------------------------------------------
        # 6. Convert to torch tensors
        # --------------------------------------------------
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # --------------------------------------------------
        # 7. HARD ASSERT (torch-level)
        # --------------------------------------------------
        assert torch.isfinite(X_train).all(), "❌ Non-finite X_train"
        assert torch.isfinite(y_train).all(), "❌ Non-finite y_train"
        assert torch.isfinite(X_val).all(),   "❌ Non-finite X_val"
        assert torch.isfinite(y_val).all(),   "❌ Non-finite y_val"
        assert torch.isfinite(X_test).all(),  "❌ Non-finite X_test"
        assert torch.isfinite(y_test).all(),  "❌ Non-finite y_test"

        # --------------------------------------------------
        # 8. Create DataLoaders
        # --------------------------------------------------
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            drop_last=False
        )

        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.config.get("batch_size", 32),
            shuffle=False
        )

        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.config.get("batch_size", 32),
            shuffle=False
        )

        print(f"✅ Data loaded successfully:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        losses = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            assert_no_nan_tensor(batch_x, "batch_x")
            assert_no_nan_tensor(batch_y, "batch_y")

            preds, contribs = self.model(
                batch_x,
                return_contributions=True,
                clamp_output=False
            )

            assert_no_nan_tensor(preds, "predictions")

            loss_dict = self.criterion(
                preds,
                batch_y,
                self.model,
                contribs
            )

            loss = loss_dict["total"]

            if torch.isnan(loss):
                raise RuntimeError("❌ NaN detected in loss")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return float(np.mean(losses))

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self):
        self.model.eval()
        losses, preds, targets = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                out = self.model(batch_x, clamp_output=True)
                loss = torch.mean((out - batch_y) ** 2)

                losses.append(loss.item())
                preds.extend(out.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        metrics = compute_metrics(
            np.array(preds),
            np.array(targets)
        )

        return float(np.mean(losses)), metrics

    # --------------------------------------------------------
    # Train loop
    # --------------------------------------------------------

    def train(self):
        print("🚀 STARTING NAM TRAINING")
        print(self.config)

        self.load_data()

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val MAE:    {metrics['mae']:.4f}")
            print(f"Val R²:     {metrics['r2']:.4f}")

            self.scheduler.step(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if is_best or epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.early_stop:
                print("⏹️ Early stopping triggered")
                break

        self.evaluate_test()

    # --------------------------------------------------------
    # Checkpoint & Test
    # --------------------------------------------------------

    def save_checkpoint(self, epoch, is_best):
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss
        }

        torch.save(state, ckpt_dir / f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            best_path = Path(self.best_model_path)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, best_path)
            print(f"⭐ Saved best model → {best_path}")

    def evaluate_test(self):
        print("\n🧪 Evaluating on test set...")

        state = torch.load(self.best_model_path)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                out = self.model(X, clamp_output=True)
                preds.extend(out.cpu().numpy())
                targets.extend(y.numpy())

        metrics = compute_metrics(
            np.array(preds),
            np.array(targets)
        )

        print("Test Results:")
        print(f"MAE : {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²  : {metrics['r2']:.4f}")


# ============================================================
# Entry
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nam_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    trainer = NAMTrainer(cfg)
    trainer.train()
