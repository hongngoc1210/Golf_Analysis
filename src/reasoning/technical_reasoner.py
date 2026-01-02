"""
Technical Reasoner for NAM Binary Classification
------------------------------------------------
Transforms NAM explanations into structured technical issues
for downstream LLM feedback generation.

GOOD = 1
BAD  = 0
"""

import json
import argparse
from pathlib import Path


class TechnicalReasonerClassification:
    def __init__(
        self,
        negative_threshold=-0.3,
        high_severity_threshold=-0.6,
        top_k_strengths=3
    ):
        """
        Args:
            negative_threshold: contribution below this is considered an issue
            high_severity_threshold: contribution below this is high severity
            top_k_strengths: number of strengths to keep
        """
        self.negative_threshold = negative_threshold
        self.high_severity_threshold = high_severity_threshold
        self.top_k_strengths = top_k_strengths


    def reason(self, explanation: dict) -> dict:
        """
        Convert explainer output → technical reasoning

        Input:
            One sample from nam_predictions.json

        Output:
            Structured reasoning dict
        """
        prediction = explanation["prediction"]
        prob_good = explanation["probability_good"]

        all_features = explanation["all_feature_contributions"]

        key_issues = self._extract_key_issues(all_features)
        strengths = self._extract_strengths(all_features)

        overall_level = self._estimate_skill_level(prediction, prob_good)

        return {
            "prediction": prediction,
            "probability_good": prob_good,
            "skill_level": overall_level,

            "key_issues": key_issues,
            "strengths": strengths,

            "summary": {
                "num_issues": len(key_issues),
                "num_strengths": len(strengths),
                "confidence": prob_good
            }
        }

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _extract_key_issues(self, features):
        issues = []

        for f in features:
            c = f["contribution"]

            if c <= self.negative_threshold:
                severity = (
                    "high" if c <= self.high_severity_threshold else "medium"
                )

                issues.append({
                    "feature": f["feature"],
                    "contribution": c,
                    "severity": severity,
                    "mechanism": self._mechanism_hint(f["feature"]),
                    "correction_hint": self._correction_hint(f["feature"])
                })

        # Sort by most negative first
        issues.sort(key=lambda x: x["contribution"])
        return issues

    def _extract_strengths(self, features):
        positives = [
            f for f in features if f["contribution"] > 0
        ]

        positives.sort(
            key=lambda x: x["contribution"],
            reverse=True
        )

        return [
            {
                "feature": f["feature"],
                "contribution": f["contribution"]
            }
            for f in positives[:self.top_k_strengths]
        ]

    def _estimate_skill_level(self, prediction, prob):
        """
        Rough skill estimation for LLM tone control
        """
        if prediction == "GOOD" and prob > 0.8:
            return "advanced"
        if prediction == "GOOD":
            return "intermediate"
        if prediction == "BAD" and prob < 0.3:
            return "beginner"
        return "developing"


    def _mechanism_hint(self, feature):
        mapping = {
            "hip_rotation_impact":
                "Insufficient hip rotation reduces energy transfer to the club",
            "hip_shoulder_separation":
                "Limited separation lowers torque generation",
            "arm_plane_mid":
                "Incorrect arm plane affects swing path consistency",
            "spine_angle_impact":
                "Poor spine control reduces strike stability",
            "head_motion_impact":
                "Excessive head movement disrupts impact accuracy",
            "balance_finish":
                "Loss of balance indicates poor weight transfer"
        }
        return mapping.get(
            feature,
            "This feature negatively affects swing efficiency"
        )

    def _correction_hint(self, feature):
        mapping = {
            "hip_rotation_impact":
                "Focus on initiating downswing with hip rotation before the arms",
            "hip_shoulder_separation":
                "Practice drills emphasizing delayed shoulder rotation",
            "arm_plane_mid":
                "Maintain a more neutral arm plane during downswing",
            "spine_angle_impact":
                "Stabilize spine angle through impact",
            "head_motion_impact":
                "Keep head centered until after ball contact",
            "balance_finish":
                "Finish with weight fully on lead foot and stable posture"
        }
        return mapping.get(
            feature,
            "Apply slow, controlled swings focusing on this movement"
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Technical Reasoner for NAM Classification"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to nam_predictions.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save technical_reasoner.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    reasoner = TechnicalReasonerClassification()

    results = []
    for item in predictions:
        reasoning = reasoner.reason(item)
        results.append({
            "index": item.get("index"),
            **reasoning
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Technical reasoning saved to {output_path}")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()