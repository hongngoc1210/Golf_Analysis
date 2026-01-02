import os
import json
import torch
import argparse
import pandas as pd
import numpy as np

from src.models.nam import NAMClassifier
from src.xai.explainer import NAMExplainerClassification
from src.utils.load_config import load_config, resolve_device


# ============================================================
# Inference Engine
# ============================================================

class NAMInferenceEngine:
    def __init__(self, config, model_path):
        self.config = config
        self.device = resolve_device(config.get("runtime", {}))
        self.model_path = model_path

        self.model = None
        self.explainer = None
        self.feature_names = None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    def load_model(self, num_features):
        self.model = NAMClassifier(
            num_features=num_features,
            hidden_units=self.config["model"]["hidden_units"],
            dropout=self.config["model"]["dropout"]
        ).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)

        print("🧪 Keys in checkpoint (sample):")
        for k in list(state.keys())[:10]:
            print(k)

        feature_net_keys = [
            k for k in state.keys() if k.startswith("feature_nets")
        ]

        max_idx = max(
            int(k.split(".")[1]) for k in feature_net_keys
        )
        print(f"🧪 Checkpoint has {max_idx + 1} feature nets")

        self.model.load_state_dict(state)
        self.model.eval()

        print(f"✅ Model loaded from {self.model_path}")

    # --------------------------------------------------------
    # Run inference on dataframe
    # --------------------------------------------------------
    
    def load_feature_list(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def run(self, df, threshold=0.5, save_path=None):
        feature_list_path = "outputs/models/nam_classifier_features.json"
        self.feature_names = self.load_feature_list(feature_list_path)

        # Ensure column alignment
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise RuntimeError(f"❌ Missing features in inference data: {missing}")

        df = df[self.feature_names]  # reorder + filter

        self.load_model(num_features=len(self.feature_names))

        self.explainer = NAMExplainerClassification(
            model=self.model,
            feature_names=self.feature_names
        )

        results = []

        for idx in range(len(df)):
            x = df.iloc[idx].values.astype(np.float32)

            explanation = self.explainer.explain(
                x, threshold=threshold
            )

            results.append({
                "index": int(idx),
                **explanation
            })

            if (idx + 1) % 10 == 0:
                print(f"🔍 Processed {idx + 1}/{len(df)} samples")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Inference results saved to {save_path}")

        return results


# ============================================================
# Entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nam_config.yaml"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/processed/test_stage2.csv"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/nam_classifier/best_model.pth"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference/nam_predictions.json"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    df = pd.read_csv(args.data)

    # Drop label if accidentally included
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    engine = NAMInferenceEngine(
        config=config,
        model_path=args.model
    )

    engine.run(
        df=df,
        threshold=args.threshold,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
