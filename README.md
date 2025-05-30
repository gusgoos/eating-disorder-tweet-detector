# Tweet Eating Disorder Classifier 

This project implements a machine learning pipeline to detect signs of eating disorders — specifically anorexia — from Spanish-language tweets. It compares three modeling approaches:  
- **Bag of Words (BoW)** with traditional classifiers 
- **BETO (BERT for Spanish)** embeddings  
- **LLaMA (Generative LLM via Ollama)** classification  

---

## Overview

- **Task**: Binary classification of tweets (`control` vs `anorexia`)
- **Language**: Spanish 🇪🇸
- **Goal**: Identify social media signals that may reflect disordered eating behavior using interpretable and modern NLP techniques.

---

## Methods & Pipelines

### 1. BoW + Traditional Classifiers
- Text preprocessed using hashtag splitting, lemmatization, emoji conversion
- Vectorized with `CountVectorizer`
- Optional augmentation using `nlpaug` (synonym + word swaps)
- Models: Logistic Regression, Random Forest, SVM (with hyperparameter tuning)

### 2. BETO (Transformers)
- Spanish BERT (`dccuchile/bert-base-spanish-wwm-uncased`)
- [CLS] token extracted → SVD (dim. reduction) → model input
- Same classifiers and evaluation metrics as BoW

### 3. LLaMA via Ollama (Generative)
- Prompts generated per tweet for binary classification
- Model responds directly with `0` (control) or `1` (anorexia)
- Evaluated separately using Confusion Matrix and ROC

---

## Evaluation

- **Cross-validation (Stratified K-Fold)**
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Visualizations**:
  - Confusion Matrix
  - ROC Curves
  - Learning Curves

---

## Installation & Setup

### Requirements
> ⚠️ **IMPORTANT**: This project requires additional language resource downloads beyond `requirements.txt`. Please follow **all** steps below.

- Python 3.11.5
- Ollama (for local LLaMA usage): https://ollama.com/
- GPU recommended for BETO and data augmentation

### Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng omw-1.4
python -m spacy download es_core_news_sm
