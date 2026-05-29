# 📧 Spam Email Classifier

A machine-learning model that classifies emails as **spam** or **ham (legitimate)** using classic NLP feature extraction and supervised classification.

## Pipeline
1. **Text preprocessing** — lowercasing, tokenization, stop-word removal, cleaning.
2. **Feature extraction** — TF-IDF / bag-of-words vectorization of message text.
3. **Classification** — a supervised classifier trained on labelled spam/ham data.
4. **Evaluation** — accuracy, precision, recall, and confusion matrix (precision matters here — false positives hide real mail).

## What it demonstrates
Fundamentals of **text classification**: turning raw language into numeric features and choosing metrics that match the cost of each error type.

## Stack
Python · scikit-learn · pandas · NLTK / regex · Jupyter

## Run
Open the notebook and run all cells — it covers load → preprocess → train → evaluate.
