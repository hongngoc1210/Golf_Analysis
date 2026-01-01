"""
LLM Feedback Generator
----------------------
Multi-backend: Gemini (preferred) → Template fallback
"""

import os
import json
import google.generativeai as genai


class BaseFeedbackGenerator:
    def generate(self, structured_input: dict) -> str:
        raise NotImplementedError

class GeminiFeedbackGenerator(BaseFeedbackGenerator):
    def __init__(self):
        self.enabled = False
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.enabled = True

    def generate(self, structured_input: dict) -> str:
        prompt = self._build_prompt(structured_input)

        response = self.model.generate_content(
            prompt,
            generation_config={
                # "temperature": 0.4,
                # "top_p": 0.9,
                "max_output_tokens": 2048
            }
        )
        return response.text

    def _build_prompt(self, data: dict) -> str:
        return f"""
    You are a professional golf coach with experience training amateur and competitive players.

    You are given a structured technical analysis generated from motion capture data and an explainable AI model (Neural Additive Model).

    IMPORTANT RULES (STRICT):
    - DO NOT invent new problems.
    - DO NOT contradict the provided analysis.
    - DO NOT mention the AI model.
    - ALL feedback MUST be grounded in the given data.
    - ALL sections MUST be written in VIETNAMESE.
    - Focus on PRACTICAL COACHING FEEDBACK, not academic explanation.

    TECHNICAL ANALYSIS (JSON):
    {json.dumps(data, indent=2, ensure_ascii=False)}

    YOUR TASK:

    ### 1. Overall Assessment (Ngắn gọn)
    - Tóm tắt chất lượng swing trong **2–3 câu**.
    - Phản ánh đúng **band kỹ thuật**:
      - Band 1–2: dùng ngôn ngữ đơn giản, dễ hiểu, động viên.
      - Band 3: cân bằng giữa giải thích và kỹ thuật.
      - Band 4–5: dùng thuật ngữ kỹ thuật chính xác hơn.

    ### 2. Key Technical Issues (Vấn đề chính)
    Với mỗi vấn đề:
    - Giải thích **đang sai ở đâu** (tay, thân, hông, đầu, tư thế).
    - Giải thích **vì sao điều này làm giảm hiệu suất** (khoảng cách, độ chính xác, ổn định).
    - Nhắc rõ **giai đoạn swing** (Setup, Top, Impact, Finish…).
    - Tránh thuật ngữ phức tạp nếu band thấp.

    ### 3. Coaching Feedback – Cách Cải Thiện CỤ THỂ (PHẦN QUAN TRỌNG NHẤT)
    Hãy đưa ra hướng dẫn thực hành chi tiết, ví dụ:
    - Tay: nên nâng cao hay hạ thấp? mặt phẳng tay dốc hay phẳng?
    - Tư thế đứng: độ gập gối, góc lưng, phân bổ trọng tâm.
    - Xoay người: hông nên xoay trước hay sau, mức độ xoay.
    - Đầu và thăng bằng: giữ ổn định như thế nào.

    Sử dụng ngôn ngữ huấn luyện dễ hiểu, ví dụ:
    - “Ở giai đoạn impact, bạn nên…”
    - “Hãy cảm nhận rằng…”
    - “Tránh việc…”

    ### 4. Bài Tập Luyện (Drills)
    - Đề xuất **1 bài tập cụ thể cho mỗi vấn đề**.
    - Bài tập phải:
      - Dễ hình dung
      - Có thể tập tại nhà hoặc sân tập
    - Nói rõ **khi tập cần chú ý điều gì**.

    ### 5. Lời Nhận Xét Từ HLV (Vietnamese – Cá nhân hóa)
    - Viết như một HLV đang nói trực tiếp với người chơi.
    - Động viên dựa trên **điểm mạnh hiện có**.
    - Nhấn mạnh rằng cải thiện là khả thi nếu tập trung đúng điểm.

    OUTPUT FORMAT (STRICT – KHÔNG THÊM MỤC):

    ## Overall Assessment
    ...

    ## Key Technical Issues
    - ...

    ## Coaching Feedback
    ...

    ## Drills to Practice
    - ...

    ## Coach’s Feedback
    ...
    """

class TemplateFeedbackGenerator(BaseFeedbackGenerator):
    def generate(self, data: dict) -> str:
        lines = []

        lines.append("## Overall Assessment")
        lines.append(
            f"Your swing shows several technical limitations, particularly during the "
            f"{data['priority_phase']['phase']} phase."
        )

        if data["key_issues"]:
            lines.append("\n## Key Issues")
            for issue in data["key_issues"]:
                lines.append(
                    f"- **{issue['feature']} ({issue['phase']})**: "
                    f"{issue['mechanism']} This often leads to {issue['effect']}."
                )

        if data["key_issues"]:
            lines.append("\n## Recommended Drills")
            for issue in data["key_issues"]:
                lines.append(
                    f"- Practice slow-motion swings focusing on correcting "
                    f"{issue['feature']} during the {issue['phase']} phase."
                )

        if data["strengths"]:
            lines.append("\n## Strengths")
            for s in data["strengths"]:
                lines.append(
                    f"- **{s['feature']} ({s['phase']})** contributes positively to your swing."
                )

        if data.get("compensation_note"):
            lines.append("\n## Final Note")
            lines.append(data["compensation_note"])

        return "\n".join(lines)



def get_feedback_generator():
    try:
        gemini = GeminiFeedbackGenerator()
        print("✅ Gemini LLM enabled")
        return gemini
    except Exception as e:
        print("⚠️ Gemini unavailable:", e)
        return TemplateFeedbackGenerator()
