import os
import json
import google.generativeai as genai


class GeminiLLMConsumer:
    def __init__(self, model_name="gemma-3-27b-it"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini model ready: {model_name}")

    def generate_feedback(self, technical_reasoning: dict) -> str:
        safe_input = self._sanitize(technical_reasoning)
        prompt = self._build_prompt(safe_input)

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 2048
            }
        )

        return response.text

    def _sanitize(self, obj):
        """Ensure JSON-safe numeric values"""
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return 0.0
            return obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

    def _build_prompt(self, data: dict) -> str:
        skill_level = data.get("skill_level")
        prediction = data.get("prediction")
        probability_good = data.get("probability_good")
        return f"""
You are a professional golf coach.

You are given a STRICT technical analysis derived from an explainable AI model.
You MUST follow these rules:

- DO NOT invent new issues.
- DO NOT contradict the analysis.
- ONLY use the provided key_issues and strengths.
- Adjust explanation depth based on skill level.

PLAYER PROFILE:
- Classification: {prediction}
- Probability GOOD: {probability_good}
- Skill Level: {skill_level}

TECHNICAL ANALYSIS (JSON):
{json.dumps(data, indent=2)}

TASK:

1. Overall Assessment
   - 2–3 sentences.
   - Use simple language for beginners.
   - Use more technical language for advanced players.

2. Main Technical Problems
   - Explain what is wrong biomechanically.
   - Explain why it hurts performance.
   - Reference ONLY the provided key_issues.

3. How to Improve (VERY IMPORTANT)
   - Give clear, concrete advice:
     - body rotation
     - arm position
     - posture
     - stability
   - Use phrases like:
     - "Ở giai đoạn này, bạn nên..."
     - "Hãy cảm nhận rằng..."

4. Drills
   - ONE drill per issue.
   - Easy to practice.
   - Explain what to focus on.

5. Coach Feedback (Vietnamese)
   - Entire section in Vietnamese.
   - Encouraging, realistic, supportive.
   - Reference strengths.

OUTPUT FORMAT (STRICT):

## Overall Assessment
(text)

## Key Technical Issues
- Issue 1
- Issue 2

## How to Improve Your Swing
(detailed guidance)

## Drills to Practice
- Drill 1
- Drill 2

## Coach’s Feedback
(Vietnamese, personalized)
OUTPUT MUST BE VIETNAMESE, DO NOT TRANSLATE TO ENGLISH.
"""
