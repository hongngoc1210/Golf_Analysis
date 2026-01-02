from src.llm.llm_consumer import GeminiLLMConsumer
import json

with open("D:\\DataStorm\\outputs\\reports\\technical_reasoner.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

llm = GeminiLLMConsumer()

for i, sample in enumerate(samples[:5]):
    print(f"\n🧠 Feedback {i+1}")
    feedback = llm.generate_feedback(sample)
    print(feedback)
