import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class NAMTrainerClassification:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        device,
        threshold=0.5,
        early_stopping_patience=20
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.threshold = threshold
        self.early_stop = early_stopping_patience

        self.history = []
        self.best_f1 = -1.0
        self.patience_counter = 0

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------
    def train_epoch(self):
        self.model.train()
        losses = []

        for x, y in tqdm(self.train_loader, desc="Training", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            logits, probs, contribs = self.model(
                x, return_contributions=True
            )

            loss_dict = self.loss_fn(
                logits, y, self.model, contribs
            )
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        all_probs = []
        all_preds = []
        all_targets = []

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits, probs = self.model(x)
            preds = (probs >= self.threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)

        # Guard ROC-AUC
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = None

        return {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc
        }

    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------
    def fit(self, num_epochs, save_path=None):
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            train_loss = self.train_epoch()
            metrics = self.evaluate()

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "val_roc_auc": metrics["roc_auc"]
            }
            self.history.append(epoch_log)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AUC: {metrics['roc_auc']}"
            )

            # Early stopping on F1
            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.patience_counter = 0

                if save_path:
                    torch.save(
                        self.model.state_dict(),
                        save_path
                    )
                    print("⭐ Saved best model")

            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stop:
                print("⏹️ Early stopping triggered")
                break

        # --------------------------------------------------
        # Save training log ONCE
        # --------------------------------------------------
        if save_path:
            log_path = save_path.replace(".pth", "_training_log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            print(f"📝 Training log saved to {log_path}")
