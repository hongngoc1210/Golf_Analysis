import os
import json
import argparse
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.models.nam import NAMClassifier
from src.utils.load_config import load_config, resolve_device


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    df,
    feature_names,
    device,
    threshold=0.5
):
    X = torch.tensor(
        df[feature_names].values,
        dtype=torch.float32
    ).to(device)

    y_true = df["target"].values

    logits, probs = model(X)
    probs = probs.cpu().numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist()
    }

    return metrics


# ============================================================
# Entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NAM Classifier on Test Set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nam_classifier.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/nam_classifier_stage2"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="datasets/processed/test_stage2.csv"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reports/nam_test_metrics.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("runtime", {}))

    # --------------------------------------------------
    # Load feature list
    # --------------------------------------------------
    feature_path = os.path.join("D:\\DataStorm\\outputs\\models\\nam_classifier_features.json")
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------
    df = pd.read_csv(args.test_data)

    assert "target" in df.columns, "❌ test set must contain 'target'"
    assert all(
        f in df.columns for f in feature_names
    ), "❌ Feature mismatch between model and test data"

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = NAMClassifier(
        num_features=len(feature_names),
        hidden_units=cfg["model"]["hidden_units"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    state = torch.load(
        os.path.join(args.model),
        map_location=device
    )
    model.load_state_dict(state)
    model.eval()

    print("✅ Model loaded")
    print(f"🔢 Num features: {len(feature_names)}")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    metrics = evaluate(
        model=model,
        df=df,
        feature_names=feature_names,
        device=device,
        threshold=args.threshold
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------
    print("\n🧪 TEST SET RESULTS")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
