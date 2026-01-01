"""
XAI Module - Giải thích predictions từ NAM
Tạo feature contributions và phase-based analysis
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class NAMExplainer:
    def __init__(self, model, feature_definitions_path="datasets/feature_definitions.json"):
        """
        Initialize explainer với trained NAM model
        
        Args:
            model: Trained NAM model
            feature_definitions_path: Path to feature definitions
        """
        self.model = model
        self.model.eval()
        
        # Load feature definitions
        with open(feature_definitions_path, "r", encoding="utf-8") as f:
            self.feature_defs = json.load(f)
        
        self.features = self.feature_defs["features"]
        self.feature_names = [f["name"] for f in self.features]
        self.band_defs = self.feature_defs["band_definitions"]
        
        # Create feature-to-event mapping
        self.feature_to_event = {
            f["name"]: f["event"] for f in self.features
        }
        
    def score_to_band(self, score):
        """
        Convert score to band
        """
        for band_def in self.band_defs:
            if band_def["range"][0] <= score < band_def["range"][1]:
                return band_def["band"], band_def["label"]
        # Edge case: score = 10
        return 5, "Gần chuẩn huấn luyện"
    
    def explain_single(self, features, feature_names=None):
        """
        Giải thích prediction cho 1 sample
        
        Args:
            features: (17,) numpy array hoặc torch tensor
            feature_names: list of feature names (optional)
        
        Returns:
            explanation dict
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        features = features.to(next(self.model.parameters()).device)
        
        # Get prediction and contributions
        with torch.no_grad():
            score, contributions = self.model(features, return_contributions=True)
        
        score = score.item()
        contributions = contributions.squeeze().cpu().numpy()
        
        # Get band
        band, band_label = self.score_to_band(score)
        
        # Create feature contributions dict
        feature_contribs = {}
        for i, fname in enumerate(feature_names):
            feature_contribs[fname] = {
                "value": features[0, i].item(),
                "contribution": contributions[i],
                "event": self.feature_to_event.get(fname, "Unknown")
            }
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contribs.items(),
            key=lambda x: abs(x[1]["contribution"]),
            reverse=True
        )
        
        explanation = {
            "score": score,
            "band": band,
            "band_label": band_label,
            "bias": self.model.bias.item(),
            "feature_contributions": feature_contribs,
            "top_positive_features": [],
            "top_negative_features": []
        }
        
        # Get top positive/negative features
        for fname, fdata in sorted_features:
            if fdata["contribution"] > 0:
                explanation["top_positive_features"].append({
                    "feature": fname,
                    "contribution": fdata["contribution"],
                    "event": fdata["event"]
                })
            else:
                explanation["top_negative_features"].append({
                    "feature": fname,
                    "contribution": fdata["contribution"],
                    "event": fdata["event"]
                })
        
        # Limit to top 5
        explanation["top_positive_features"] = explanation["top_positive_features"][:5]
        explanation["top_negative_features"] = explanation["top_negative_features"][:5]
        
        return explanation
    
    def explain_by_phase(self, explanation):
        """
        Group explanations by swing phase/event
        
        Args:
            explanation: output from explain_single
        
        Returns:
            phase_analysis dict
        """
        phase_analysis = {}
        
        for fname, fdata in explanation["feature_contributions"].items():
            event = fdata["event"]
            
            if event not in phase_analysis:
                phase_analysis[event] = {
                    "features": [],
                    "total_contribution": 0,
                    "positive_count": 0,
                    "negative_count": 0
                }
            
            phase_analysis[event]["features"].append({
                "name": fname,
                "contribution": fdata["contribution"],
                "value": fdata["value"]
            })
            
            phase_analysis[event]["total_contribution"] += fdata["contribution"]
            
            if fdata["contribution"] > 0:
                phase_analysis[event]["positive_count"] += 1
            else:
                phase_analysis[event]["negative_count"] += 1
        
        # Sort phases by absolute total contribution
        sorted_phases = sorted(
            phase_analysis.items(),
            key=lambda x: abs(x[1]["total_contribution"]),
            reverse=True
        )
        
        return dict(sorted_phases)

    def explain(self, features):
        """
        High-level explanation pipeline used by inference

        Args:
            features: (17,) numpy / torch tensor

        Returns:
            structured explanation dict
        """
        explanation = self.explain_single(features)

        phase_analysis = self.explain_by_phase(explanation)
        key_issues = self.identify_key_issues(explanation)

        return {
            "score": explanation["score"],
            "band": explanation["band"],
            "band_label": explanation["band_label"],
            "bias": explanation["bias"],

            "feature_contributions": explanation["feature_contributions"],
            "top_positive_features": explanation["top_positive_features"],
            "top_negative_features": explanation["top_negative_features"],
            "explanation": explanation,
            "phase_analysis": phase_analysis,
            "key_issues": key_issues
        }
    
    def identify_key_issues(self, explanation, threshold=0.3):
        """
        Identify key technical issues (negative contributions)
        
        Args:
            explanation: output from explain_single
            threshold: minimum absolute contribution to consider
        
        Returns:
            list of issues
        """
        issues = []
        
        for fname, fdata in explanation["feature_contributions"].items():
            contrib = fdata["contribution"]
            
            if contrib < -threshold:
                severity = "high" if contrib < -0.6 else "medium"
                
                issues.append({
                    "feature": fname,
                    "phase": fdata["event"],
                    "contribution": contrib,
                    "severity": severity,
                    "description": self._get_issue_description(fname, fdata["event"])
                })
        
        # Sort by severity
        issues.sort(key=lambda x: x["contribution"])
        
        return issues
    
    def _get_issue_description(self, feature_name, event):
        """
        Generate human-readable description for technical issues
        """
        descriptions = {
            "spine_angle_impact": "Excessive spine angle at impact reduces consistency",
            "hip_rotation_impact": "Insufficient hip rotation at impact limits power",
            "head_motion_impact": "Excessive head movement at impact affects accuracy",
            "balance_finish": "Poor balance at finish indicates stability issues",
            "arm_plane_mid": "Arm plane off during downswing affects club path",
            "hip_shoulder_separation": "Limited hip-shoulder separation reduces power generation"
        }
        
        return descriptions.get(feature_name, f"Issue with {feature_name} during {event}")
    
    def visualize_contributions(self, explanation, save_path=None):
        """
        Visualize feature contributions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort features by contribution
        feature_data = [
            (fname, fdata["contribution"]) 
            for fname, fdata in explanation["feature_contributions"].items()
        ]
        feature_data.sort(key=lambda x: x[1])
        
        features = [x[0] for x in feature_data]
        contributions = [x[1] for x in feature_data]
        
        # Bar plot
        colors = ['red' if c < 0 else 'green' for c in contributions]
        ax1.barh(features, contributions, color=colors, alpha=0.6)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Contribution to Score', fontsize=12)
        ax1.set_title('Feature Contributions', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Phase-based analysis
        phase_analysis = self.explain_by_phase(explanation)
        phases = list(phase_analysis.keys())
        phase_contribs = [phase_analysis[p]["total_contribution"] for p in phases]
        
        colors = ['red' if c < 0 else 'green' for c in phase_contribs]
        ax2.barh(phases, phase_contribs, color=colors, alpha=0.6)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Total Contribution', fontsize=12)
        ax2.set_title('Phase-based Analysis', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add score info
        fig.suptitle(
            f"Score: {explanation['score']:.2f} | Band: {explanation['band']} ({explanation['band_label']})",
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")
        
        return fig
    
    def generate_summary(self, explanation):
        """
        Generate text summary of explanation
        """
        summary = f"""
=== GOLF SWING ANALYSIS ===

Overall Score: {explanation['score']:.2f}/10
Technical Band: {explanation['band']} - {explanation['band_label']}

KEY STRENGTHS:
        """
        
        for i, feat in enumerate(explanation['top_positive_features'][:3], 1):
            summary += f"{i}. {feat['feature']} ({feat['event']}): +{feat['contribution']:.2f}\n"
        
        summary += "\nAREAS FOR IMPROVEMENT:\n"
        
        for i, feat in enumerate(explanation['top_negative_features'][:3], 1):
            summary += f"{i}. {feat['feature']} ({feat['event']}): {feat['contribution']:.2f}\n"
        
        # Phase analysis
        phase_analysis = self.explain_by_phase(explanation)
        worst_phase = min(phase_analysis.items(), key=lambda x: x[1]["total_contribution"])
        
        summary += f"\nMOST PROBLEMATIC PHASE: {worst_phase[0]}\n"
        summary += f"Total impact: {worst_phase[1]['total_contribution']:.2f}\n"
        
        return summary