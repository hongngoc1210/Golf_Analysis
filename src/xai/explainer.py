import torch
import numpy as np


class NAMExplainerClassification:
    """
    Explainer for NAM Binary Classification
    GOOD = 1, BAD = 0
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names

    @torch.no_grad()
    def explain(self, x, threshold=0.5):
        """
        Args:
            x: (num_features,) or (1, num_features)
        Returns:
            structured explanation dict (JSON-safe)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = x.to(device)

        logits, probs, contributions = self.model(
            x, return_contributions=True
        )

        prob = float(probs.item())
        logit = float(logits.item())
        pred_label = int(prob >= threshold)

        contribs = contributions.squeeze().cpu().numpy()

        feature_contribs = []
        for i, fname in enumerate(self.feature_names):
            feature_contribs.append({
                "feature": fname,
                "contribution": float(contribs[i]),
                "direction": "good" if contribs[i] > 0 else "bad"
            })

        # Sort by absolute impact
        feature_contribs.sort(
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )

        top_positive = [
            f for f in feature_contribs if f["contribution"] > 0
        ][:5]

        top_negative = [
            f for f in feature_contribs if f["contribution"] < 0
        ][:5]

        return {
            "prediction": "GOOD" if pred_label == 1 else "BAD",
            "probability_good": prob,
            "logit": logit,
            "threshold": threshold,

            "top_positive_features": top_positive,
            "top_negative_features": top_negative,

            "all_feature_contributions": feature_contribs
        }
