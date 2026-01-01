# Golf Swing Analysis với NAM + XAI + LLM Feedback

Hệ thống phân tích kỹ thuật swing golf sử dụng Neural Additive Models (NAM) kết hợp Explainable AI và LLM feedback.

## 🎯 Tổng quan

Pipeline hoàn chỉnh:
```
CaddieSet (70+ features) 
    → Feature Engineering (17 features) 
    → NAM Model (Score 0-10) 
    → Band Classification (1-5) 
    → XAI Explanations 
    → LLM Feedback
```

## 📁 Cấu trúc Project

```
golf_nam_project/
├── data/
│   ├── raw/                    # Raw CaddieSet CSV
│   ├── processed/              # Processed train/val/test
│   └── feature_definitions.json
├── models/
│   └── nam/                    # Trained models
├── src/
│   ├── models/                 # NAM implementation
│   ├── xai/                    # Explainability
│   ├── llm/                    # LLM feedback
│   └── utils/                  # Utilities
├── scripts/
│   ├── train.py               # Training script     
│   └── inference.py           # Inference pipeline
└── outputs/                    # Analysis results
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Tạo cấu trúc project
python setup_environment.py

# Install dependencies
pip install -r requirements.txt
```

### 2. Chuẩn bị Data

```bash
# Đặt file caddieset.csv vào data/raw/
cp /path/to/caddieset.csv data/raw/

# Preprocess data
preprocessing.ipynb
```

Output: `data/processed/` sẽ có train.csv, val.csv, test.csv

### 3. Train Model

```bash
python train.py
```

Model tốt nhất được lưu tại: `outputs/models/nam/best_model.pth`

### 4. Run Inference

```bash
python inference.py
```

## 📊 17 Features và Events Mapping

| Feature | Event | Ý nghĩa |
|---------|-------|---------|
| spine_tilt | Address | Góc nghiêng cột sống ban đầu |
| stance_width | Address | Độ rộng stance |
| hip_shoulder_separation | Top | Độ tách vai-hông ở top |
| hip_rotation_top | Top | Xoay hông ở top backswing |
| arm_plane_mid | Mid-downswing | Mặt phẳng cánh tay |
| hip_rotation_mid | Mid-downswing | Xoay hông giữa downswing |
| spine_angle_impact | Impact | Góc cột sống tại impact |
| hip_rotation_impact | Impact | Xoay hông tại impact |
| head_motion_impact | Impact | Chuyển động đầu |
| shaft_lean_impact | Impact | Độ nghiêng shaft |
| spine_angle_release | Release | Góc cột sống ở release |
| arm_extension_release | Release | Duỗi tay |
| balance_finish | Finish | Cân bằng ở finish |
| hip_angle_finish | Finish | Góc hông ở finish |
| ... | ... | ... |

## 🧠 NAM Model Architecture

```python
NAM(
  num_features=17,
  hidden_units=[64, 32],
  dropout=0.1
)

# Score = β₀ + Σ fᵢ(xᵢ)
# Mỗi feature có 1 FeatureNN riêng
```

**Ưu điểm:**
- ✅ Explainable: Contribution từng feature rõ ràng
- ✅ Additivity: Score = tổng các contributions
- ✅ Non-linear: NN học non-linear patterns

## 🎯 Band Definitions

| Band | Score Range | Label |
|------|-------------|-------|
| 1 | 0-2 | Rất nhiều lỗi kỹ thuật |
| 2 | 2-4 | Kỹ thuật yếu, thiếu ổn định |
| 3 | 4-6 | Trung bình |
| 4 | 6-8 | Tốt, còn vài lỗi nhỏ |
| 5 | 8-10 | Gần chuẩn huấn luyện |

## 🔍 XAI Output Example

```json
{
  "score": 6.8,
  "band": 4,
  "band_label": "Tốt, còn vài lỗi nhỏ",
  "feature_contributions": {
    "spine_angle_impact": -1.2,
    "hip_shoulder_separation": 0.6,
    "balance_finish": 0.8
  },
  "phase_analysis": {
    "Impact": {
      "total_contribution": -1.6,
      "issues": ["spine_angle", "head_motion"]
    },
    "Finish": {
      "total_contribution": 0.8,
      "strengths": ["balance"]
    }
  }
}
```

## 💬 LLM Feedback

Sử dụng Claude API để generate feedback:

```python
from src.llm.feedback_generator import LLMFeedbackGenerator

generator = LLMFeedbackGenerator(api_key="your-api-key")
feedback = generator.generate_feedback(explanation, phase_analysis, issues)
```

**Output Example:**
```markdown
# Golf Swing Analysis Report

## Overall Assessment
Your swing scored 6.8/10, placing you in Band 4 (Tốt, còn vài lỗi nhỏ).

## Your Strengths 💪
- Balance at finish: +0.8
- Hip-shoulder separation: +0.6

## Areas for Improvement 🎯
1. Spine angle at impact: -1.2
   - Excessive backward lean reduces consistency
   - Drill: Practice impact bag with spine angle check

2. Head motion at impact: -0.4
   - Too much head movement affects accuracy
   - Drill: "Head against wall" drill
...
```

## 📈 Evaluation Metrics

### Regression Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Band Metrics
- Band Accuracy
- Within-1-Band Accuracy
- Per-band Precision/Recall/F1

## 🛠️ Advanced Usage

### Custom Feature Engineering

```python
from src.data.preprocessing import CaddieSetPreprocessor

preprocessor = CaddieSetPreprocessor()
# Modify feature extraction
features = preprocessor.extract_17_features(df)
```

### Model Configuration

```python
from src.models.nam import NAMConfig

config = NAMConfig()
config.hidden_units = [128, 64, 32]
config.learning_rate = 5e-4
config.batch_size = 64
```

### Batch Analysis

```python
from scripts.inference import GolfSwingAnalyzer

analyzer = GolfSwingAnalyzer()
results = analyzer.analyze_batch(test_features_df)
```

## 📚 References

1. **Neural Additive Models**: Agarwal et al., "Neural Additive Models: Interpretable Machine Learning with Neural Nets"
2. **CaddieSet**: Golf swing biomechanics dataset with MediaPipe features
3. **Claude API**: Anthropic's language model for feedback generation

## ⚙️ Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
anthropic>=0.18.0
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional feature engineering
- [ ] Multi-task learning (distance + accuracy)
- [ ] Real-time video analysis integration
- [ ] Mobile app deployment

## 📝 License

MIT License

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: Để sử dụng LLM feedback, cần ANTHROPIC_API_KEY:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Hoặc hệ thống sẽ tự động fallback sang template-based feedback.