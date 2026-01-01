"""
Technical Reasoner for CaddieSet
--------------------------------
Chuyển output XAI (NAM) → structured technical insights
KHÔNG dùng LLM, logic thuần Python, reproducible
"""

from typing import Dict, List

# ============================================================
# DOMAIN KNOWLEDGE MAP
# ============================================================

FEATURE_MECHANISM = {
    # Address
    "stance_width": "A stance that is too narrow or too wide reduces balance and consistency.",
    "spine_angle_address": "Incorrect spine angle at address disrupts posture and swing plane.",
    "shoulder_angle_setup": "Poor shoulder alignment at setup affects swing direction.",

    # Backswing / Top
    "hip_shoulder_separation": "Limited X-factor reduces power generation.",
    "shoulder_rotation_top": "Insufficient shoulder turn limits backswing depth.",
    "hip_rotation_top": "Over or under hip rotation at the top affects sequencing.",

    # Downswing
    "arm_plane_mid": "An improper arm plane during downswing causes swing path issues.",
    "hip_rotation_downswing": "Poor hip rotation timing reduces speed and control.",

    # Impact
    "hip_rotation_impact": "Insufficient hip rotation at impact reduces energy transfer.",
    "spine_angle_impact": "Losing spine angle at impact causes inconsistent strikes.",
    "hip_hanging_back": "Hanging back on the trail hip leads to weak ball contact.",

    # Release / Finish
    "spine_angle_release": "Good spine extension during release supports solid contact.",
    "finish_balance": "Balanced finish indicates good swing sequencing."
}

FEATURE_EFFECT = {
    "hip_shoulder_separation": "loss of power",
    "hip_rotation_impact": "reduced ball speed",
    "spine_angle_impact": "inconsistent strike",
    "arm_plane_mid": "offline shots",
    "hip_hanging_back": "fat or thin contact"
}

PHASE_PRIORITY = [
    "Impact",
    "Downswing",
    "Top",
    "Address",
    "Finish",
    "Release"
]

# ============================================================
# CORE REASONER
# ============================================================

def build_technical_insights(
    explanation: Dict,
    phase_analysis: Dict,
    issues: List[Dict],
    strengths: List[Dict],
    contribution_threshold: float = 0.2
) -> Dict:
    """
    Build structured technical reasoning from XAI output

    Returns a JSON-serializable dict for LLM consumption
    """

    insights = {
        "priority_phase": None,
        "key_issues": [],
        "strengths": [],
        "compensation_note": None
    }

    # --------------------------------------------------
    # 1. Identify priority phase (worst-performing)
    # --------------------------------------------------
    worst_phase = None
    worst_value = float("inf")

    for phase, stats in phase_analysis.items():
        value = stats.get("total_contribution", 0.0)
        if value < worst_value:
            worst_value = value
            worst_phase = phase

    insights["priority_phase"] = {
        "phase": worst_phase,
        "reason": "This phase contributes the most negative impact to the overall score."
    }

    # --------------------------------------------------
    # 2. Process key technical issues
    # --------------------------------------------------
    for issue in issues:
        feature = issue["feature"]
        contrib = issue["contribution"]

        if abs(contrib) < contribution_threshold:
            continue

        insights["key_issues"].append({
            "feature": feature,
            "phase": issue["phase"],
            "severity": issue.get("severity", "medium"),
            "mechanism": FEATURE_MECHANISM.get(
                feature,
                "This feature negatively affects swing efficiency."
            ),
            "effect": FEATURE_EFFECT.get(
                feature,
                "overall performance reduction"
            ),
            "impact_score": round(float(contrib), 3)
        })

    # Sort issues by severity of contribution
    insights["key_issues"] = sorted(
        insights["key_issues"],
        key=lambda x: abs(x["impact_score"]),
        reverse=True
    )[:3]

    # --------------------------------------------------
    # 3. Process strengths
    # --------------------------------------------------
    for strength in strengths[:3]:
        insights["strengths"].append({
            "feature": strength["feature"],
            "phase": strength["phase"],
            "description": FEATURE_MECHANISM.get(
                strength["feature"],
                "This feature contributes positively to swing quality."
            ),
            "impact_score": round(float(strength["contribution"]), 3)
        })

    # --------------------------------------------------
    # 4. Compensation logic
    # --------------------------------------------------
    if insights["key_issues"] and insights["strengths"]:
        insights["compensation_note"] = (
            "Some technical weaknesses are partially compensated by existing strengths, "
            "which helps maintain baseline performance."
        )

    return insights