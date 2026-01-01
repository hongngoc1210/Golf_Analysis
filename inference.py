"""
InferenceEngine v1.0
--------------------
Golf Swing Analysis Pipeline

Components:
- NAM (Prediction)
- XAI (Feature & Phase Attribution)
- Symbolic Technical Reasoner
- LLM Feedback Generator (Gemini / Template)

Design goals:
- Reproducible (thesis-ready)
- Cost-safe (paid LLM API)
- Robust (LLM never breaks ML inference)

Author: DataStorm
"""

# ============================================================
# ENV & STANDARD IMPORTS
# ============================================================

import os
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from src.models.nam import NAM
from src.xai.explainer import NAMExplainer
from src.llm.technical_reasoner import build_technical_insights
from src.llm.llm import get_feedback_generator

def to_json_safe(obj: Any):
    """
    Convert numpy / torch objects to JSON-serializable types.
    Ensures reproducibility and clean artifact storage.
    """
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    return obj


class InferenceEngine:
    """
    Thesis-ready inference engine for golf swing analysis.
    """

    def __init__(
        self,
        model_path: str,
        feature_def_path: str,
        device: str | None = None,
        use_llm: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_llm = use_llm

        with open(feature_def_path, "r", encoding="utf-8") as f:
            defs = json.load(f)

        self.feature_defs = defs["features"]
        self.feature_names = [f["name"] for f in self.feature_defs]

        self.model = NAM(num_features=len(self.feature_names))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ NAM model loaded: {model_path}")

        self.explainer = NAMExplainer(model=self.model)
        print("✅ XAI explainer initialized")

        self.feedback_generator = get_feedback_generator()

        self.llm_backend = (
            "gemini"
            if getattr(self.feedback_generator, "enabled", False)
            else "template"
        )

        print(f"✅ Feedback backend: {self.llm_backend}")
        self.llm_sleep_sec = float(os.getenv("LLM_SLEEP_SEC", 10.0))


    def analyze_single(self, row: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single golf swing sample.
        """

        x = row[self.feature_names].values.astype(float)

        analysis = self.explainer.explain(x)

        structured_reasoning = {
            "score": analysis["score"],
            "band": analysis["band"],
            "band_label": analysis["band_label"],
            "priority_phase": self._get_priority_phase(
                analysis["phase_analysis"]
            ),
            "key_issues": analysis["key_issues"],
            "strengths": analysis["explanation"]["top_positive_features"],
            "phase_analysis": analysis["phase_analysis"],
        }

        structured_reasoning = to_json_safe(structured_reasoning)
        feedback = None

        if self.use_llm:
            try:
                feedback = self.feedback_generator.generate(structured_reasoning)
            except Exception as e:
                print("⚠️ LLM generation failed:", e)
                feedback = None

        return {
            "analysis": to_json_safe(analysis),
            "feedback": feedback,
            "metadata": {
                "llm_backend": self.llm_backend,
                "llm_model": os.getenv("GEMINI_MODEL", "n/a"),
            },
        }

    def analyze_batch(
        self,
        df: pd.DataFrame,
        n_samples: int = 10,
        output_dir: str = "outputs/reports",
    ):
        results = []

        for idx in range(min(n_samples, len(df))):
            print(f"\n🏌️ ANALYZING SWING {idx + 1}/{n_samples}")

            result = self.analyze_single(df.iloc[idx])
            results.append(result)

            if self.use_llm and self.llm_backend == "gemini":
                time.sleep(self.llm_sleep_sec)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "batch_analysis.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_path}")

    @staticmethod
    def _get_priority_phase(phase_analysis: Dict):
        """
        Identify the most problematic swing phase.
        """
        if not phase_analysis:
            return None

        phase, stats = min(
            phase_analysis.items(),
            key=lambda x: x[1]["total_contribution"],
        )

        return {
            "phase": phase,
            "total_contribution": float(stats["total_contribution"]),
        }

    @staticmethod
    def plot_contributions(analysis: Dict, save_path: str):
        contribs = analysis["analysis"]["explanation"]["feature_contributions"]

        features = list(contribs.keys())
        values = list(contribs.values())
        colors = ["green" if v > 0 else "red" for v in values]

        plt.figure(figsize=(10, 6))
        plt.barh(features, values, color=colors)
        plt.axvline(0, linestyle="--", color="black")
        plt.title("Feature Contributions")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    print("🏌️ GOLF SWING ANALYSIS – INFERENCE ENGINE v1.0")

    engine = InferenceEngine(
        model_path="outputs/models/best_model.pth",
        feature_def_path="datasets/feature_definitions.json",
        use_llm=True,
    )

    test_df = pd.read_csv("datasets/processed/test.csv")
    test_df = test_df.reset_index(drop=True)

    engine.analyze_batch(
        test_df,
        n_samples=1,
        output_dir="outputs/reports",
    )


if __name__ == "__main__":
    main()
