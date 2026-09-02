# 🧠 Sales Sentiment Analysis Platform
### Dual-Engine NLP System — Logistic Regression & Fine-tuned DeBERTa Transformer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DeBERTa--v3-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

**Sales Sentiment Analysis Platform** is a production-ready NLP application built to automatically classify sales CRM remarks into **Positive**, **Negative**, and **Neutral** sentiments — helping sales teams and CRM managers make faster, data-driven decisions.

This project features **two independent prediction engines** that can be run and compared side-by-side:

| Engine | Model Architecture | Default Port | Primary Use Case |
|---|---|---|---|
| 🟢 **Classical ML** | Logistic Regression + TF-IDF / LSA / Word2Vec | `8000` / `8024` | Ultra-fast, lightweight CPU inference |
| 🔵 **Deep Learning** | Fine-tuned DeBERTa-v3-small Transformer | `8042` | High-accuracy contextual sentiment understanding |

---

## 💼 Business Impact & Purpose

> **Designed for CRM-integrated sales pipelines and B2B sales intelligence.**

Sentiment analysis of sales remarks helps organizations:

- 📈 **Increase CRM Efficiency by ~33%** — Automatically tagging thousands of CRM entries eliminates manual review time.
- 🎯 **Prioritize High-Potential Leads** — Focus sales energy on **Positive**-sentiment customers who are ready to close.
- 🚨 **Flag At-Risk Accounts Early** — Detect **Negative** signals before churn or deal loss happens.
- 🤝 **Manage Neutral Prospects Smarter** — Re-engage fence-sitters with targeted follow-ups, improving pipeline conversion.
- 📊 **Drive Data-Driven Sales Decisions** — Replace guesswork with confidence-scored predictions to guide team strategy.
- 💰 **Increase Business Revenue** — Focusing on positive customers and addressing negatives promptly accelerates revenue closure.
- 🏢 **Scalable Across Teams** — Batch-process entire sales datasets in seconds with CSV/Excel upload.

---

## ✨ Key Features

- ✅ **Dual Prediction Engine** — Classical ML (fast, lightweight) + Transformer (high accuracy)
- ✅ **Single Text Prediction** — Paste any CRM remark and get instant sentiment + confidence score
- ✅ **Batch File Processing** — Upload Excel/CSV files to classify thousands of remarks at once
- ✅ **Confidence Scoring** — Every prediction comes with a confidence percentage (0–100%)
- ✅ **Interactive Web Dashboard** — Visual sentiment distribution charts powered by Chart.js
- ✅ **Manual Override & Audit Log** — Reviewers can correct predictions with explanation tracking
- ✅ **Export Results** — Download classified data as CSV or multi-sheet Excel
- ✅ **REST API + Swagger Docs** — Ready for CRM system integration via `/docs`
- ✅ **PII Masking** — Emails, URLs, phone numbers, and dates are auto-masked before inference

---

## 🧪 Models & Techniques

### 1. 🟢 Classical ML Engine — Logistic Regression

| Component | Details |
|---|---|
| **Algorithm** | Logistic Regression (Multi-class) |
| **Vectorizer** | TF-IDF (`max_features=10,000`, `ngram_range=(1,3)`) |
| **Dimensionality Reduction** | Latent Semantic Analysis (LSA via TruncatedSVD) |
| **Word Embeddings** | Custom Word2Vec trained on sales domain data |
| **Class Imbalance** | `class_weight='balanced'` to handle skewed classes |
| **Hyperparameter Tuning** | GridSearchCV with 5-fold cross-validation (`f1_macro`) |
| **Label Encoding** | Scikit-Learn `LabelEncoder` (Positive / Negative / Neutral) |

### 2. 🔵 Deep Learning Engine — Fine-tuned DeBERTa

| Component | Details |
|---|---|
| **Base Model** | `microsoft/deberta-v3-small` |
| **Fine-tuned On** | Sales CRM remarks (domain-specific) |
| **Inference** | Hugging Face `pipeline` with `torch.no_grad()` |
| **Optimization** | Token truncation to 512, batch_size=16, singleton service loading |
| **Hardware** | Auto GPU (CUDA) / CPU fallback |
| **Text Normalization** | `ftfy` for encoding fix + Regex PII masking |

### 3. 🔧 NLP Preprocessing Pipeline

- Lowercase normalization → Regex special character removal
- Lemmatization via NLTK `WordNetLemmatizer`
- **No stopword removal** *(Critical: negations like "not interested", "won't buy" are preserved)*
- PII masking: Emails → `EMAIL`, URLs → `URL`, Phones → `PHONE`, Dates → `DATE`

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| **Source** | Real-world B2B Sales CRM remarks |
| **Total Raw Rows** | 4,770 |
| **After Cleaning** | 3,290 |
| **Columns** | `Company Name`, `Opportunity Name`, `Remarks`, `Sentiment` |
| **Classes** | Positive, Negative, Neutral |
| **Train / Test Split** | 80% / 20% (stratified) |

### Class Distribution (After Cleaning)

| Sentiment | Count | Percentage |
|---|---|---|
| **Neutral** | 1,653 | 50.2% |
| **Negative** | 1,307 | 39.7% |
| **Positive** | 330 | 10.0% |

> **Note:** Class imbalance is addressed using `class_weight='balanced'` for Logistic Regression and loss weighting during DeBERTa fine-tuning.

---

## 📁 Project Structure

```text
MultipleSentimentLogistic/
│
├── app/
│   ├── main.py                    # FastAPI app — Logistic Regression engine
│   ├── main_deberta.py            # FastAPI app — DeBERTa Transformer engine
│   ├── transformer_service.py     # DeBERTa inference pipeline & PII masking
│   ├── utils.py                   # Text preprocessing utilities
│   ├── models/                    # Model weights directory
│   │   ├── New_Updated_Models/    # Trained Logistic, TF-IDF, LSA, Word2Vec models
│   │   └── deberta_sales_v1/      # Fine-tuned DeBERTa model weights (ignored in Git)
│   ├── static/                    # CSS, JavaScript, Chart.js assets
│   └── templates/                 # Jinja2 HTML templates (dashboard)
│
├── data/                          # Dataset files (CSV / XLSX)
├── notebook/
│   └── EDA_4770.ipynb             # Exploratory Data Analysis notebook
│
├── launcher_logistic.py           # One-click launcher — Logistic server (Port 8000)
├── launcher_deberta.py            # One-click launcher — DeBERTa server (Port 8042)
├── start_deberta.bat              # Windows batch launcher for DeBERTa
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git rules (excludes heavy model weights & venv)
├── ALGORITHMS_AND_TECHNIQUES.md   # Detailed ML/NLP methodology documentation
├── DATA_SUMMARY.md                # Dataset analysis and cleaning report
└── README.md                      # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) NVIDIA GPU with CUDA for accelerated DeBERTa inference

### 1. Clone the Repository

```bash
git clone https://github.com/naveenkumar-koli/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create & Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

You can run either engine individually or both simultaneously on separate ports.

### 🟢 Option A — Classical Logistic Regression Server (Port 8000)

```bash
python launcher_logistic.py
```

### 🔵 Option B — DeBERTa Transformer Server (Port 8042)

```bash
python launcher_deberta.py
```

*Or double-click `start_deberta.bat` on Windows.*

The launcher will automatically open your web browser. If not, open the links manually:

| Engine Server | Local Dashboard URL | API Documentation |
|---|---|---|
| **Logistic Regression** | `http://127.0.0.1:8000` | `http://127.0.0.1:8000/docs` |
| **DeBERTa Transformer** | `http://127.0.0.1:8042` | `http://127.0.0.1:8042/docs` |

---

## 🌐 API Endpoints

Both servers expose identical REST API interfaces for seamless integration:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Web Dashboard UI |
| `GET` | `/health` | Server & model health status check |
| `POST` | `/predict` | Single text prediction (raw JSON) |
| `POST` | `/predict-batch` | Batch texts array prediction (raw JSON) |
| `POST` | `/bulk_analyze_file` | Bulk CSV/Excel file analysis (returns JSON summary) |
| `POST` | `/batch_analyze` | Dashboard file upload & rendering |
| `GET` | `/export_data` | Download filtered analysis as CSV |
| `GET` | `/docs` | Swagger Interactive API Documentation |

### Example — Single Text Prediction

**Request:**
```bash
curl -X POST "http://127.0.0.1:8042/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"The customer is very interested and wants a demo next week.\"}"
```

**Response:**
```json
{
  "text": "The customer is very interested and wants a demo next week.",
  "clean_text": "The customer is very interested and wants a demo next week.",
  "sentiment": "Positive",
  "confidence": 0.9412,
  "scores": [
    {"label": "Positive", "score": 0.9412},
    {"label": "Neutral", "score": 0.0421},
    {"label": "Negative", "score": 0.0167}
  ]
}
```

---

## 📤 Output & Analytics Dashboard

For every CRM remark — entered individually or uploaded in bulk — the platform provides:

| Field | Description | Example |
|---|---|---|
| `sentiment` | Classification outcome | `Positive` / `Negative` / `Neutral` |
| `confidence` | Model certainty score | `0.9412` (94.12% confidence) |
| `scores` | Full probability breakdown across all 3 classes | `[Positive: 94%, Neutral: 4%, Negative: 2%]` |

### Dashboard Analytics Features:
- 📊 **Dynamic Charts** — Visual distribution of Positive vs Neutral vs Negative leads.
- 🏢 **Company-wise Drilldown** — Analyze sentiment breakdown per client account.
- 🔍 **Filtering & Search** — Search remarks by keyword, company, or sentiment tag.
- ✏️ **Manual Edit & Audit Trail** — Override predictions and log reasons for model retraining.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI (Async REST API) |
| **Classical ML / NLP** | Scikit-Learn, NLTK, Gensim (Word2Vec) |
| **Deep Learning** | Hugging Face Transformers, PyTorch, DeBERTa-v3 |
| **Text Cleaning** | NLTK, `ftfy`, Regex PII Masking |
| **Templating & UI** | Jinja2, HTML5, Vanilla CSS |
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Data Visualization** | Chart.js |
| **ASGI Web Server** | Uvicorn |

---

## 📖 Documentation & References

- 📄 [`ALGORITHMS_AND_TECHNIQUES.md`](ALGORITHMS_AND_TECHNIQUES.md) — In-depth ML/NLP pipeline, feature engineering, and DeBERTa fine-tuning specs
- 📄 [`DATA_SUMMARY.md`](DATA_SUMMARY.md) — Dataset statistics, cleaning steps, class distribution, and data quality report
- 📓 `notebook/EDA_4770.ipynb` — Exploratory Data Analysis & visualization notebook

---

## 👨‍💻 Author

**Naveen Kumar Koli**  
- GitHub: [@naveenkumar-koli](https://github.com/naveenkumar-koli)
- Project Repository: [Sentiment-Analysis](https://github.com/naveenkumar-koli/Sentiment-Analysis)

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
