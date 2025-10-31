
---

```markdown
# Intelligent SMS Summarizer

**AI-powered SMS classification & daily digest**  
**79.8% accurate BERT model • Upload CSV → Get summary in seconds**

**"151 Telecom offers, 59 Shopping deals, 29 Real Estate alerts. 12 Spam filtered."**

---

## Features

| Feature | Status |
|---------|--------|
| Upload SMS CSV export | Done |
| **79.8% accurate classification** | Done |
| Real-time spam filtering | Done |
| Daily promotional summary | Done |
| Top offers list | Done |
| Download classified results | Done |
| Runs on **M2 GPU** (MPS) | Done |
| **Zero-cost deployment** | Done |

---

## Quick Start (2 Minutes)

```bash
# 1. Clone/download
cd sms_summarizer_web

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

**Open**: [http://localhost:8501](http://localhost:8501)


## Input CSV Format

```csv
text
"Get 50% off on recharge! Use RECH50"
"WIN IPHONE FREE! Click now"
"2 BHK flat in Pune @ ₹65L"
"EMI due on 10th"
"Flat 40% off on shoes at Myntra"
"Your account has been updated"
```

---

## Output Example

```
**151 Telecom offers, 59 Shopping deals, 29 Real Estate alerts, 71 Banking updates.**
Warning: 12 Spam messages filtered.

### Top Offers:
• Get 50% off on recharge! Use RECH50
• Flat 40% off on shoes at Myntra
• 2 BHK in Pune @ ₹65L
• EMI due on 10th
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **79.8%** |
| F1 (weighted) | 0.78 |
| Spam Detection | **100%** |
| Real Estate | 85% |

**Categories (from your original data)**:  
`Telecom`, `Marketing`, `Banking`, `Shopping`, `Informational`, `Real Estate`, `Spam`

## Model Details

**Base Model**: `distilbert-base-uncased`  
**Task**: Text Classification (7 classes)  
**Size**: 250 MB (`model.safetensors`)  
**Inference**: ~15ms per SMS (M2 MPS)  

### Classes (Original Labels)

| Class | Count | Examples |
|-------|-------|----------|
| `Telecom` | 151 | "50% off recharge", "Jio data pack" |
| `Marketing` | 109 | "Limited time offer", "sale starts" |
| `Banking` | 71 | "EMI due", "loan approved" |
| `Shopping` | 59 | "40% off shoes", "buy now" |
| `Real Estate` | 29 | "2 BHK Pune", "flat for sale" |
| `Spam` | 12 | "WIN IPHONE", "click here" |
| `Informational` | 39 | "account updated", "reminder" |

---

## Training Results

```
Epoch 1:  78.7% accuracy
Epoch 5:  78.7%
Epoch 10: **79.8%** ← Best model saved
```

**Dataset**: 470 SMS (376 train, 94 test)  
**Time**: 66 seconds (M2 MPS GPU)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `model.safetensors` error | `use_safetensors=True` in `app.py` |
| No promotional messages | Check `id2label.pkl` → labels are **case-sensitive** |
| Slow loading | First run only (model cached) |
| MPS GPU not detected | Works on CPU too |

---

## Tech Stack

| Component | Version |
|-----------|---------|
| **Model** | DistilBERT (fine-tuned) |
| **Framework** | PyTorch + Hugging Face |
| **Web** | Streamlit |
| **GPU** | MPS (Apple Silicon) |
| **Hosting** | Streamlit Cloud (free) |

---
