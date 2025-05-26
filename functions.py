# Auto-generated module: functions.py

import random
import nltk
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
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
from nltk.corpus import wordnet as wn

# === Inicialización de variables necesarias para funciones exportadas ===
random_state = 1

# spaCy y wordsegment
nlp = spacy.load("es_core_news_sm")
load()

# BETO (Tokenizador y modelo)
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
model = TFBertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

# Augmenters
nltk.download('omw-1.4')
nltk.download('wordnet')
AUGMENT_DATA = True
augmenter = naw.ContextualWordEmbsAug(
    model_path='bert-base-multilingual-uncased',
    model_type='bert',
    action="substitute",
    device='cpu'
)
def get_spanish_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word, lang='spa'):
        for lemma in synset.lemmas('spa'):
            synonym = lemma.name().replace("_", " ")
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)
class SpanishSynonymAug:
    def __init__(self, aug_p=0.3):
        self.aug_p = aug_p

    def augment(self, text):
        words = text.split()
        new_words = words.copy()
        num_to_aug = max(1, int(len(words) * self.aug_p))
        indices = list(range(len(words)))
        random.shuffle(indices)

        for idx in indices:
            word = words[idx]
            synonyms = get_spanish_synonyms(word)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
                num_to_aug -= 1
            if num_to_aug <= 0:
                break
        return " ".join(new_words)
syn_aug = SpanishSynonymAug(aug_p=0.3)
swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)


# Cliente de Ollama
ollama_client = Client(host='http://localhost:11434')
base_prompt = '''
Eres un modelo de lenguaje entrenado para análisis lingüístico con fines exclusivamente académicos. 
Tu función es detectar patrones de texto relacionados con actitudes potencialmente problemáticas hacia la alimentación o la autoimagen, 
sin emitir juicios clínicos ni diagnósticos médicos.

Analiza únicamente el contenido textual de un tweet y responde con una probabilidad entre 0.0 y 1.0 que indique la presencia de lenguaje asociado con:
- Restricción alimentaria autoimpuesta
- Mención de malestar físico o emocional vinculado con la comida o el cuerpo
- Comentarios autocríticos intensos o lenguaje negativo hacia uno mismo
- Declaraciones sobre control del peso mediante métodos no recomendados
- Enfoque en apariencia corporal extrema o metas de peso estrictas
- Uso de expresiones sobre sentirse ‘liviano/a’, ‘vacío/a’ o necesidad de ‘control’

Este análisis es parte de una investigación orientada a mejorar la comprensión de patrones lingüísticos, y no tiene implicaciones clínicas ni se utilizará para tomar decisiones sobre individuos.

Tu respuesta debe ser solo un número decimal entre 0.0 y 1.0, donde:
- 0.0 indica ausencia total de estos patrones
- 1.0 indica presencia fuerte o explícita de múltiples elementos

Ejemplo de respuesta válida: 0.83
No expliques tu razonamiento ni incluyas advertencias. Devuelve únicamente el número decimal.
'''

# Lista de resultados (global)
results_list = []

def convert_emojis(text):
    '''
    Convierte emojis a texto usando la librería emoji.
    Se creó una función para evitar problemas al realizar las pruebas unitarias.
    '''
    return emoji.demojize(text)


def preprocess_text_bert(text):
    '''
    Preprocesamiento de texto para BETO:
    - Demojización
    - Tokenización
    - Extracción del vector CLS
    - Reducción dimensional con SVD
    '''
    text = convert_emojis(text)
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='tf')
    model_output = model(encoded_input)
    embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()


def split_hashtag(tag, verbose=False):
    '''
    Divide un hashtag en palabras separadas utilizando varias técnicas de segmentación.
    Esta función procesa hashtags y los convierte en frases legibles utilizando 
    dos métodos principales: detección de CamelCase y segmentación de palabras.
    - Primero elimina el símbolo '#' si existe
    - Intenta dividir por patrones CamelCase (mayúscula después de minúscula)
    - Si no hay patrones CamelCase, utiliza la biblioteca wordsegment para 
      identificar palabras dentro del texto concatenado
    - En caso de error durante la segmentación, devuelve el hashtag original en minúsculas
    '''
    tag = tag.lstrip('#')
    camel_split = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag).lower()
    if camel_split == tag.lower():
        try:
            segmented = segment(tag.lower())
            result = " ".join(segmented) if segmented else tag.lower()
            if verbose:
                print(f"Hashtag segmentado: #{tag} → {result}")
            return result
        except:
            if verbose:
                print(f"Error al segmentar hashtag: #{tag}. Se mantiene en minúsculas.")
            return tag.lower()
    if verbose:
        print(f"Hashtag segmentado por CamelCase: #{tag} → {camel_split}")
    return camel_split


def preprocess_text_bow(text, verbose=False):
    '''
    Esta función realiza múltiples operaciones de limpieza y normalización en un texto
    para prepararlo para análisis de lenguaje natural. El procesamiento incluye:
    conversión de emojis a texto, segmentación de hashtags, eliminación de URLs, 
    menciones, números, palabras vacías, y lematización de palabras.

    - Convierte emojis a texto descriptivo usando la función 'convert_emojis'
    - Divide hashtags en palabras separadas usando 'split_hashtag'
    - Elimina URLs, menciones (@usuario), y números
    - Elimina palabras vacías (stop words) y signos de puntuación
    - Lematiza las palabras restantes
    - Utiliza el pipeline de spaCy (objeto 'nlp') para procesamiento de lenguaje
    '''
    text = convert_emojis(text).lower()
    hashtags = re.findall(r"#\w+", text)
    for tag in hashtags:
        text = text.replace(tag, split_hashtag(tag, verbose=verbose))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    doc = nlp(text)
    lemmatized_words = [
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num
    ]
    cleaned = " ".join(lemmatized_words)
    if verbose:
        print(f"Texto lematizado: {cleaned[:50]}...")
    return cleaned


def calculate_uppercase_ratio(text, verbose=False):
    '''
    Calcula la proporción de palabras en mayúsculas en un texto dado.
    - Divide el texto en palabras usando el método split()
    - Filtra las palabras que están completamente en mayúsculas
    - Devuelve la proporción de palabras en mayúsculas respecto al total de palabras
    - Si no hay palabras, devuelve 0.0
    '''
    words = text.split()
    if not words:
        if verbose:
            print("Texto vacío, ratio de mayúsculas: 0.0")
        return 0.0
    uppercase_words = [w for w in words if w.isupper()]
    ratio = len(uppercase_words) / len(words)
    if verbose:
        print(f"Ratio de mayúsculas calculado: {ratio:.2f}")
    return ratio


def generate_augmented_bow_data(X_texts, y_labels, augment_factor=1):
    '''
    Genera datos aumentados para el modelo BoW utilizando un factor de aumentación.
    - X_texts: lista de textos originales
    - y_labels: lista de etiquetas originales
    - augment_factor: número de veces que se desea aumentar cada texto
    - Devuelve listas de textos y etiquetas aumentadas
    '''

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


def augment_text_complex(text, num_augments=1):
    '''
    Genera múltiples versiones aumentadas de un texto utilizando sinónimos y permutaciones.
    - text: texto original a aumentar
    - num_augments: número de versiones aumentadas a generar
    - Devuelve una lista de textos aumentados
    '''
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


def plot_confusion_matrix_cv(y_true, y_pred, labels, model_name, pipeline_name, ax):
    '''
    Plot de la matriz de confusión utilizando seaborn.
    - y_true: etiquetas verdaderas
    - y_pred: etiquetas predichas
    - labels: etiquetas de clase
    - model_name: nombre del modelo
    - pipeline_name: nombre del pipeline
    - ax: objeto Axes para el plot
    '''
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")


def plot_roc_curve_cv(model, X, y, cv, ax):
    '''
    Plot de la curva ROC utilizando validación cruzada.
    - model: modelo entrenado
    - X: características de entrada
    - y: etiquetas de clase
    - cv: número de pliegues para la validación cruzada
    - ax: objeto Axes para el plot
    '''
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
    '''
    Plot de la curva de aprendizaje utilizando validación cruzada.
    - model: modelo entrenado
    - X: características de entrada
    - y: etiquetas de clase
    - cv: número de pliegues para la validación cruzada
    - ax: objeto Axes para el plot
    '''
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


def evaluate_model_cv(model, param_grid, X, y, model_name, pipeline_name, cv=5, save_dir="saved_models"):
    """
    Realiza búsqueda de hiperparámetros con GridSearchCV y evaluación con CV:
    - Ajusta modelo y predice con cross_val_predict.
    - Guarda el mejor modelo a disco.
    - Muestra métricas, matriz de confusión, ROC y curva de aprendizaje.
    - Almacena resultados en `results_list`.
    """
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
    y_prob = cross_val_predict(best_model, X, y, cv=skf, n_jobs=-1, method='predict_proba')[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')
    rec = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    auc = roc_auc_score(y, y_prob)

    # Subplots: 1 row x 3 cols
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} - {pipeline_name}", fontsize=14)

    plot_confusion_matrix_cv(y, y_pred, labels=np.unique(y), model_name=model_name, pipeline_name=pipeline_name, ax=axs[0])
    if len(np.unique(y)) == 2:
        plot_roc_curve_cv(best_model, X, y, cv, ax=axs[1])
    else:
        axs[1].axis('off')  # ROC not applicable
    plot_learning_curve_cv(best_model, X, y, cv, ax=axs[2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    results_list.append({
        'Pipeline': pipeline_name,
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc,
        'Best Params': grid.best_params_
    })


def classify_with_ollama_model_contextual(
    df, model_name, prob_column_name=None, csv_cache_path=None,
    max_workers=4, k_context_reset=5, verbose=False
):
    safe_model_id = model_name.replace(':', '_').replace('-', '_')
    if prob_column_name is None:
        prob_column_name = f"{safe_model_id}_prob"
    if csv_cache_path is None:
        csv_cache_path = f"{prob_column_name}_predictions.csv"

    client = Client(host='http://localhost:11434')

    def classify_batch_contextual(batch_texts, start_idx):
        results = []
        messages = [{"role": "system", "content": base_prompt}]
        context_count = 0

        for i, text in enumerate(batch_texts):
            global_idx = start_idx + i
            messages.append({"role": "user", "content": f"Tweet: '{text}'\nProbabilidad:"})
            try:
                response = client.chat(model=model_name, messages=messages)
                reply = response['message']['content'].strip()
                match = re.search(r"\b([01](?:\.\d+)?)\b", reply)
                prob = float(match.group(1)) if match else -1.0
            except Exception:
                prob = -1.0

            results.append(prob)
            messages.append({"role": "assistant", "content": str(prob)})

            if verbose:
                if prob == -1.0:
                    print(f"[{global_idx}] ❌ Clasificación inválida")
                    print(f"  Tweet   : {text}")
                    print(f"  Respuesta: {reply}")
                else:
                    print(f"[{global_idx}] ✅ Clasificado: {prob}")

            context_count += 1
            if context_count >= k_context_reset:
                if verbose:
                    print(f"[{global_idx}] 🔁 Reinicio de contexto")
                messages = [{"role": "system", "content": base_prompt}]
                context_count = 0
        return results

    def classify_all_multithread(texts):
        results = [None] * len(texts)
        chunk_size = (len(texts) + max_workers - 1) // max_workers
        chunks = [(i, texts[i:i+chunk_size]) for i in range(0, len(texts), chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(texts), desc="Clasificando tweets") as pbar:
            futures = {
                executor.submit(classify_batch_contextual, chunk, start_idx): start_idx
                for start_idx, chunk in chunks
            }

            for future in as_completed(futures):
                idx_start = futures[future]
                try:
                    chunk_results = future.result()
                    results[idx_start:idx_start+len(chunk_results)] = chunk_results
                    pbar.update(len(chunk_results))
                except Exception:
                    results[idx_start:idx_start+chunk_size] = [-1.0] * chunk_size
                    pbar.update(chunk_size)

        return results

    if os.path.exists(csv_cache_path):
        df_pred = pd.read_csv(csv_cache_path)
        df[prob_column_name] = df_pred[prob_column_name]
    else:
        df[prob_column_name] = classify_all_multithread(df['tweet_text'].tolist())
        df[['tweet_id', prob_column_name]].to_csv(csv_cache_path, index=False)

    return df


def grid_search_k_reset(
    df, model_name, true_labels, k_values,
    prob_column_prefix=None, cache_dir=".",
    max_workers=4, scoring_fn=roc_auc_score, verbose=True
):
    # Construir prefijo seguro
    safe_model_id = model_name.replace(':', '_').replace('-', '_')
    if prob_column_prefix is None:
        prob_column_prefix = f"{safe_model_id}_prob"

    scores = []

    df_sample = df.sample(frac=0.2, random_state=random_state).reset_index(drop=True)
    sampled_indices = df_sample.index
    true_labels_sample = np.array(true_labels)[sampled_indices]

    for k in k_values:
        print(f"\n🔍 Probando k = {k}...")

        prob_column_name = f"{prob_column_prefix}_k{k}"
        csv_cache_path = os.path.join(cache_dir, f"{prob_column_name}.csv")

        df_copy = df_sample.copy()
        df_copy = classify_with_ollama_model_contextual(
            df_copy,
            model_name=model_name,
            prob_column_name=prob_column_name,
            csv_cache_path=csv_cache_path,
            max_workers=max_workers,
            k_context_reset=k
        )

        y_prob = df_copy[prob_column_name].values
        y_true = true_labels_sample

        # Filtrar predicciones inválidas
        valid_mask = y_prob != -1.0
        y_prob_valid = y_prob[valid_mask]
        y_true_valid = np.array(y_true)[valid_mask]

        if len(np.unique(y_true_valid)) < 2:
            score = 0.0
            if verbose:
                print(f"⚠️ No hay clases suficientes en y_true válido (k={k}), score=0.0")
        else:
            try:
                score = scoring_fn(y_true_valid, y_prob_valid)
            except Exception as e:
                print(f"❌ Error calculando métrica para k={k}: {e}")
                score = 0.0

        scores.append((k, score))
        print(f"✅ k={k}, {scoring_fn.__name__} = {score:.4f}")

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"\n🏆 Mejor valor de k: {best_k} con {scoring_fn.__name__} = {best_score:.4f}")

    return best_k, scores


def process_llm_broken_predictions(df_pred, prob_col, process_broken_prompt=-1):
    """
    Procesa las predicciones de un modelo LLM sobre el test set.
    - df_pred: DataFrame con las predicciones.
    - prob_col: nombre de la columna de probabilidades.
    - process_broken_prompt: 
        -1 para ignorar -1.0,
         0 para reemplazar -1.0 por 0.0,
         1 para reemplazar -1.0 por 1.0.
    Devuelve (y_true, y_prob)
    """
    if process_broken_prompt == -1:
        valid_mask = df_pred[prob_col] != -1.0
        y_true = df_pred.loc[valid_mask, "label_enc"].values
        y_prob = df_pred.loc[valid_mask, prob_col].values
    elif process_broken_prompt in [0, 1]:
        y_true = df_pred["label_enc"].values
        y_prob = df_pred[prob_col].replace(-1.0, float(process_broken_prompt)).values
    else:
        raise ValueError(f"Valor no válido para process_broken_prompt: {process_broken_prompt}")
    return y_true, y_prob


