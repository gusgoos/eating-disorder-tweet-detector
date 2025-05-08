# Auto-generated module: functions.py

import re
import random
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TFBertModel, BertTokenizer
from wordsegment import segment, load
import spacy
import emoji
import nlpaug.augmenter.word as naw
from ollama import Client
import joblib

# === Inicialización de variables necesarias para funciones exportadas ===
random_state = 1

# spaCy y wordsegment
nlp = spacy.load("es_core_news_sm")
load()

# BETO (Tokenizador y modelo)
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
model = TFBertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

# Augmenters
augmenter = naw.ContextualWordEmbsAug(
    model_path='bert-base-multilingual-uncased',
    model_type='bert',
    action="substitute",
    device='cpu'
)
syn_aug = naw.SynonymAug(aug_src='wordnet', lang='spa', aug_p=0.3)
swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)

# Cliente de Ollama
ollama_client = Client(host='http://localhost:11434')

# Lista de resultados (global)
results_list = []

def convert_emojis(text):
    return emoji.demojize(text)


def preprocess_text_bert(text):
    text = convert_emojis(text)
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='tf')
    model_output = model(encoded_input)
    embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()


def preprocess_text_no_bert(text):
    text = convert_emojis(text).lower()
    hashtags = re.findall(r"#\w+", text)
    for tag in hashtags:
        text = text.replace(tag, split_hashtag(tag))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    cleaned = " ".join(lemmatized_words)
    return cleaned


def split_hashtag(tag):
    tag = tag.lstrip('#')
    camel_split = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag).lower()
    if camel_split == tag.lower():
        try:
            segmented = segment(tag.lower())
            return " ".join(segmented) if segmented else tag.lower()
        except:
            return tag.lower()
    return camel_split


def augment_text_complex(text, num_augments=1):
    augmented_texts = []
    for _ in range(num_augments):
        aug_type = random.choice(['synonym', 'swap'])
        if aug_type == 'synonym':
            augmented = syn_aug.augment(text)
        elif aug_type == 'swap':
            augmented = swap_aug.augment(text)
        else:
            augmented = text
        if isinstance(augmented, list):
            augmented = augmented[0]
        augmented_texts.append(augmented)
    return augmented_texts


def generate_augmented_bow_data(X_texts, y_labels, augment_factor=1):
    aug_texts, aug_labels = [], []
    for i, text in enumerate(X_texts):
        for _ in range(augment_factor):
            try:
                aug = augmenter.augment(text)
            except:
                aug = text
            aug_texts.append(aug)
            aug_labels.append(y_labels[i])
    return aug_texts, aug_labels


def llama_classify_tweet_generate(text):
    prompt = (
        "A continuación se presenta un tweet escrito en español. Tu tarea es analizarlo cuidadosamente y "
        "determinar si su contenido refleja señales de conductas asociadas a trastornos alimenticios, específicamente anorexia. "
        "Considera el tono emocional, la mención de hábitos alimenticios, el lenguaje corporal implícito o explícito, "
        "y cualquier indicio de preocupación extrema por el peso o la comida.\n"
        f"Tweet: '{text}'\n"
        "¿Este tweet sugiere conductas relacionadas con anorexia?: "
        "Responde únicamente con 1 si detectas evidencia que sugiere tal conducta, o con 0 si no hay indicios suficientes."
    )
    response = ollama_client.generate(model='llama3', prompt=prompt)
    reply = response['response'].strip()
    return 1 if '1' in reply else 0


def classify_all_tweets_parallel(texts, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(llama_classify_tweet_generate, texts), total=len(texts)))
    return results


def evaluate_model_cv(model, param_grid, X, y, model_name, pipeline_name, cv=5, save_dir="saved_models"):
    print(f"\nEvaluando {model_name} en {pipeline_name} con KFold CV...")

    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    # Guardar modelo
    model_filename = f"{save_dir}/{pipeline_name}_{model_name.replace(' ', '_')}.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Modelo guardado como {model_filename}")

    y_pred = cross_val_predict(best_model, X, y, cv=skf, n_jobs=-1)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')
    rec = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    # Subplots: 1 row x 3 cols
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} - {pipeline_name}", fontsize=14)

    plot_confusion_matrix_cv(y, y_pred, labels=np.unique(y), model_name=model_name, pipeline_name=pipeline_name, ax=axs[0])
    if len(np.unique(y)) == 2:
        plot_roc_curve_cv(best_model, X, y, cv, ax=axs[1])
    else:
        axs[1].axis('off')  # ROC not applicable
    plot_learning_curve_cv(best_model, X, y, cv, ax=axs[2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    plt.show()

    results_list.append({
        'Pipeline': pipeline_name,
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Best Params': grid.best_params_
    })


def plot_confusion_matrix_cv(y_true, y_pred, labels, model_name, pipeline_name, ax):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")


def plot_roc_curve_cv(model, X, y, cv, ax):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X[test_idx])[:,1]
        else:
            y_score = model.decision_function(X[test_idx])
        y_test_bin = label_binarize(y[test_idx], classes=np.unique(y)).ravel()
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        auc = roc_auc_score(y_test_bin, y_score)
        ax.plot(fpr, tpr, label=f"Fold {i+1} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_title("ROC Curve")
    ax.set_xlabel("Falsos Positivos")
    ax.set_ylabel("Verdaderos Positivos")
    ax.grid(True)
    ax.legend(fontsize='small')


def plot_learning_curve_cv(model, X, y, cv, ax):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring='f1_weighted', n_jobs=-1
    )
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o--', color='blue', label='Train Mean')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Val Mean')

    ax.set_title("Learning Curve")
    ax.set_xlabel("Muestras de entrenamiento")
    ax.set_ylabel("F1 Score")
    ax.grid(True)
    ax.legend(fontsize='small')


