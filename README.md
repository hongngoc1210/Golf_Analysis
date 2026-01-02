# 🏌️ Golf Swing Quality Assessment with Neural Additive Models (NAM)

## 📌 Overview

This project proposes an **end-to-end explainable machine learning pipeline** for **golf swing quality assessment**, combining:

* **Stage 1**: Feature selection using tree-based models (LightGBM + SHAP)
* **Stage 2**: Explainable **Neural Additive Model (NAM)** for **binary classification**
* **XAI**: Feature-level contribution analysis
* **LLM-based feedback generation** (Gemini / Gemma / Template fallback)

The system not only predicts whether a golf swing is **GOOD** or **BAD**, but also explains *why* and provides **human-readable coaching feedback**.

---

## 🎯 Problem Definition

* **Input**: Motion-derived golf swing features (angles, ratios, positions)
* **Output**:

  * Binary classification:

    * `0` → Bad swing
    * `1` → Good swing
  * Feature contributions
  * Personalized coaching feedback

---

## 🧠 Core Ideas

1. **Interpretability-first modeling**
   Each feature contributes independently via a small neural network:
   [
   \text{logit} = b + \sum_i f_i(x_i)
   ]

2. **Two-stage learning**

   * Stage 1: Learn global feature importance
   * Stage 2: Learn interpretable per-feature effects

3. **Human-in-the-loop explainability**

   * Model → Explainer → Reasoner → LLM → Feedback

---

## 🗂️ Project Structure

```text
DataStorm/
├── datasets/
│   ├── raw/
│   └── processed/
│       ├── train_stage2.csv
│       ├── val_stage2.csv
│       └── test_stage2.csv
│
├── src/
│   ├── models/
│   │   ├── nam.py                 # NAMClassifier + Loss
│   │   └── trainer.py             # Training loop
│   │
│   ├── xai/
│   │   └── explainer.py           # NAMExplainerClassification
│   │
│   ├── reasoning/
│   │   └── technical_reasoner_classification.py
│   │
│   ├── llm/
│   │   ├── llm_consumer.py        # Gemini / Gemma / Template
│   │   └── prompts.py
│   │
│   └── utils/
│       ├── load_config.py
│       ├── metrics.py
│       └── nam_export.py
│
├── outputs/
│   ├── models/
│   │   └── nam_classifier_stage2/
│   │       ├── best_model.pth
│   │       └── feature_list.json
│   ├── inference/
│   │   └── nam_predictions.json
│   └── reports/
│       └── nam_test_metrics.json
│
├── configs/
│   └── nam_classifier.yaml
│
├── train.py
├── inference.py
├── evaluate_test.py
├── generate_feedback.py
└── README.md
```

---

## ⚙️ Pipeline Description

### 🔹 Stage 1 – Feature Selection (LightGBM)

* Train a tree-based model on all available features
* Use SHAP values to estimate global feature importance
* Select **Top-N features**
* Create `*_stage2.csv` datasets

> Purpose: Reduce noise & stabilize NAM training

---

### 🔹 Stage 2 – NAM Binary Classification

* Model: **Neural Additive Model**
* Each feature has its own sub-network
* Output:

  * Logit
  * Probability
  * Per-feature contribution

**Loss function**:
[
\mathcal{L} =
\text{BCEWithLogits}

* \lambda_1 \lVert \theta \rVert^2
* \lambda_2 \mathbb{E}[f_i(x_i)^2]
  ]

---

### 🔹 Evaluation Metrics

* Accuracy
* F1-score
* ROC-AUC
* Confusion Matrix

Evaluation is performed strictly on **held-out test set**.

---

### 🔹 Explainability (XAI)

`NAMExplainerClassification` produces:

* Prediction (`GOOD` / `BAD`)
* Probability
* Top positive features
* Top negative features
* Full contribution list

All outputs are **JSON-safe**.

---

### 🔹 Technical Reasoning Layer

`TechnicalReasonerClassification` converts raw contributions into:

* Key technical issues
* Severity estimation
* Strengths vs weaknesses
* Structured reasoning schema for LLM

This layer ensures:

* No hallucination
* Domain grounding
* Stable feedback

---

### 🔹 LLM Feedback Generation

Supported backends:

* Gemini API
* Gemma (via API)
* Template fallback

LLM receives **structured reasoning**, not raw numbers.

Output:

* Overall assessment
* Technical explanation
* Improvement guidance
* Drills
* Encouragement (Vietnamese)

---

## 🚀 How to Run

### 1️⃣ Train Stage 2 NAM

```bash
python train.py --config configs/nam_classifier.yaml
```

---

### 2️⃣ Evaluate on Test Set

```bash
python evaluate_test.py \
  --config configs/nam_classifier.yaml \
  --model_dir outputs/models/nam_classifier_stage2 \
  --test_data datasets/processed/test_stage2.csv
```

---

### 3️⃣ Run Inference

```bash
python inference.py \
  --config configs/nam_classifier.yaml \
  --data datasets/processed/test_stage2.csv \
  --model outputs/models/nam_classifier_stage2/best_model.pth \
  --output outputs/inference/nam_predictions.json
```

---

### 4️⃣ Generate Technical Reasoning

```bash
python technical_reasoner_classification.py \
  --input outputs/inference/nam_predictions.json \
  --output outputs/inference/technical_reasoning.json
```

---

### 5️⃣ Generate LLM Feedback

```bash
python generate_feedback.py \
  --input outputs/inference/technical_reasoning.json
```

---

## 📊 Key Advantages

* ✅ Fully explainable architecture
* ✅ Feature-level interpretability
* ✅ Stable reasoning before LLM
* ✅ Suitable for academic research
* ✅ Ready for real coaching systems

---

## 📚 Intended Use

* Master / Bachelor thesis
* Sports analytics research
* Explainable AI case study
* Intelligent coaching systems

---

## 📌 Notes

* The project is designed to be **model-agnostic at Stage 1**
* NAM architecture is extensible to:

  * Regression
  * Multi-class classification
* LLM backend can be swapped without retraining

---

## ✍️ Author

Developed as part of an academic research project on **Explainable AI for Sports Performance Analysis**.
