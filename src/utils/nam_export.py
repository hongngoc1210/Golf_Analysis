import torch
import json
from pathlib import Path
import numpy as np


def export_nam_feature_weights(
    model,
    feature_names,
    save_path="outputs/models/nam_feature_weights.json"
):
    """
    Export feature importance proxies for NAMClassifier.

    Importance proxy = L2 norm of last layer weights
    """

    model.eval()
    results = []

    for i, (fname, feature_net) in enumerate(
        zip(feature_names, model.feature_nets)
    ):
        # FeatureNN.net is nn.Sequential
        layers = list(feature_net.net)

        # Last Linear layer
        last_linear = None
        for layer in reversed(layers):
            if isinstance(layer, torch.nn.Linear):
                last_linear = layer
                break

        if last_linear is None:
            raise RuntimeError(
                f"No Linear layer found in FeatureNN for feature {fname}"
            )

        weight = last_linear.weight.detach().cpu().numpy()
        bias = last_linear.bias.detach().cpu().numpy()

        importance = float(np.linalg.norm(weight))

        results.append({
            "feature": fname,
            "importance_l2": importance,
            "last_layer_weight_shape": list(weight.shape),
            "last_layer_bias": float(bias[0])
        })

    # Sort by importance
    results.sort(
        key=lambda x: x["importance_l2"],
        reverse=True
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ NAM feature weights exported to {save_path}")

    return results
