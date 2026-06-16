# Sales Sentiment Dataset — Data Summary Report

> **Dataset:** `Sales Final Data (4770).csv` / `Sales_Final_Data_4770.csv`
> **Project:** MultipleSentimentLogistic
> **Last Updated:** 2026-05-20

---

## 1. Raw Data Overview

| Property | Value |
|---|---|
| **File Name** | `Sales_Final_Data_4770.csv` |
| **Encoding** | `latin-1` |
| **Total Rows (raw)** | 4,769 |
| **Total Columns** | 2 |
| **Columns** | `Remarks`, `Sentiment` |

---

## 2. Data Quality Issues Found

### 2.1 Null Values

| Column | Null Count |
|---|---|
| Remarks | 2 |
| Sentiment | 0 |
| **Total Nulls** | **2** |

> These rows had a Sentiment label but no Remarks text — likely data entry errors in the CRM.

---

### 2.2 Duplicate Rows

| Metric | Value |
|---|---|
| **Total rows involved in duplicates** | 2,499 |
| **Unique remarks that were duplicated** | 1,021 |
| **Extra copies removed** | 1,478 |
| **Duplication rate** | ~52% of total rows |

#### What this means:
- **1,021 distinct remarks** each appeared **more than once** in the dataset
- Together they occupied **2,499 rows** in total
- After `drop_duplicates`, **1,478 extra copies** were removed (keeping one of each)

#### Why does this happen?
Most likely caused by the same sales remarks being **re-entered multiple times** in the CRM system — e.g., copy-pasting previous notes, repeated follow-up entries, or bulk data exports that included historical records multiple times.

---

### 2.3 Sentiment Label Inconsistencies (Before Cleaning)

The raw data contained **inconsistent capitalization** of sentiment labels:

| Variant | Example |
|---|---|
| `neutral` | lowercase |
| `Neutral ` | trailing space |
| `Neutral` | correct |
| `positive` | lowercase |
| `POSITIVE` | uppercase |

**Fix applied:** `str.strip().str.capitalize()` to normalize all labels.

---

## 3. Data Cleaning Pipeline

```python
# Step 1: Remove null Remarks/Sentiment
data = data.dropna(subset=['Remarks', 'Sentiment'])

# Step 2: Remove duplicate remarks
data = data.drop_duplicates(subset=['Remarks'])

# Step 3: Normalize sentiment labels
data['Sentiment'] = data['Sentiment'].str.strip().str.capitalize()

# Step 4: Keep only valid sentiment classes
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

data_clean = data
```

---

## 4. Clean Dataset Summary

| Step | Rows Remaining |
|---|---|
| Raw data | 4,769 |
| After `dropna` | 4,767 |
| After `drop_duplicates` | 3,290 |
| After label filter | **3,290** ✅ |

### Sentiment Distribution (Clean Data)

| Sentiment | Count | Percentage |
|---|---|---|
| Neutral | 1,653 | 50.2% |
| Negative | 1,307 | 39.7% |
| Positive | 330 | 10.0% |
| **Total** | **3,290** | **100%** |

> [!WARNING]
> **Class Imbalance:** Positive class is severely underrepresented at only **10%**.
> The model uses `class_weight='balanced'` in Logistic Regression to compensate.

---

## 5. Train / Test Split

| Set | Rows | Percentage |
|---|---|---|
| **Training** | 2,632 | 80% |
| **Testing** | 658 | 20% |
| **Total (clean)** | 3,290 | 100% |

```python
train_test_split(..., test_size=0.2, random_state=42, stratify=y)
```

> `stratify=y` ensures the class distribution is preserved in both train and test sets.

---

## 6. Feature Engineering

| Feature | Configuration |
|---|---|
| **Vectorizer** | TF-IDF |
| `max_features` | 10,000 |
| `ngram_range` | (1, 3) — unigrams, bigrams, trigrams |
| `min_df` | 2 |
| `max_df` | 0.95 |
| `sublinear_tf` | True |
| `stop_words` | None — negations preserved intentionally |

> **Why no stopword removal?**
> Sales remarks rely heavily on negation words like *"not interested"*, *"no plans"*, *"won't buy"*.
> Removing stopwords would flip the sentiment meaning of these critical phrases.

---

## 7. Model

| Property | Value |
|---|---|
| **Algorithm** | Logistic Regression |
| **Tuning** | GridSearchCV (5-fold CV) |
| `class_weight` | `balanced` |
| `scoring` | `f1_macro` |
| **C values tested** | 0.1, 0.5, 1, 3, 10 |
| **Solvers tested** | liblinear, lbfgs |

### Model Files Saved

| File | Description |
|---|---|
| `sentiment_model_4770.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer_4770.pkl` | Fitted TF-IDF vectorizer |
| `label_encoder_4770.pkl` | Label encoder (Positive/Negative/Neutral → 0/1/2) |
| `word2vec_sales_4770.model` | Word2Vec embeddings (semantic similarity) |
| `lsa_model_4770.pkl` | LSA model (topic analysis) |
| `lsa_vectorizer_4770.pkl` | LSA vectorizer |

> All files saved at:
> `app/models/New_Updated_Models/`

---

## 8. Key Observations

1. **Data volume loss:** Only **3,290 of 4,769 rows (69%)** survive cleaning — 31% was noise/duplicates
2. **Duplication is the biggest issue:** 1,478 extra duplicate rows removed (vs only 2 null rows)
3. **Positive class needs attention:** Only 330 Positive examples — model may struggle with rare wins
4. **Negation preservation is critical:** "not interested", "no plans", "won't buy" must not be stripped
5. **Training set is small:** 2,632 training rows is relatively small for a 3-class NLP model — more data collection recommended

---

## 9. Recommendations

| Priority | Action |
|---|---|
| 🔴 High | Collect more **Positive** labeled remarks to balance the dataset |
| 🔴 High | Investigate root cause of 1,021 duplicated remarks in CRM |
| 🟡 Medium | Review `positive_predicted_as_neutral.csv` error report and add those to training |
| 🟡 Medium | Consider data augmentation for Positive class |
| 🟢 Low | Re-run model training after adding new clean data |
