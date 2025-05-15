import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

import functions

@pytest.fixture(autouse=True)
def clear_results_list():
    # Ensure results_list is empty before each test
    functions.results_list.clear()
    yield
    functions.results_list.clear()

def test_convert_emojis():
    out = functions.convert_emojis("Hola 😀")
    assert ":grinning_face:" in out
    assert out == "Hola :grinning_face:"
    
def test_preprocess_text_no_bert_basic():
    text = "Esto es una #Prueba. Visitantes vayan a https://t.co y saluda a @usuario123! 100%"
    cleaned = functions.preprocess_text_no_bert(text)
    assert "ir" in cleaned  
    assert "Visitantes" not in cleaned
    assert "usuario" not in cleaned
    assert "100" not in cleaned
    assert "prueba" in cleaned
    
def test_split_hashtag():
    assert functions.split_hashtag("#HelloWorld") == "hello world"
    assert functions.split_hashtag("#ejercicioencasa") == "ejercicio en casa"
    assert functions.split_hashtag("#holamundo") == "hola mundo"
    
def test_augment_text_complex(monkeypatch):
    monkeypatch.setattr(functions, "syn_aug", type("A", (), {"augment": lambda self, t: [f"SYN:{t}"]})())
    monkeypatch.setattr(functions, "swap_aug", type("B", (), {"augment": lambda self, t: [f"SWP:{t}"]})())
    sequence = ["synonym", "swap", "other"]
    monkeypatch.setattr(functions.random, "choice", lambda choices: sequence.pop(0))
    out = functions.augment_text_complex("texto", num_augments=3)
    assert out == ["SYN:texto", "SWP:texto", "texto"]
    
def test_generate_augmented_bow_data(monkeypatch):
    monkeypatch.setattr(functions, "augmenter", type("C", (), {"augment": lambda self, t: f"aug_{t}"})())
    X = ["a", "b"]
    y = [1, 0]
    texts, labels = functions.generate_augmented_bow_data(X, y, augment_factor=2)
    assert texts == ["aug_a", "aug_a", "aug_b", "aug_b"]
    assert labels == [1, 1, 0, 0]
    
def test_llama_classify_tweet_generate(monkeypatch):
    class DummyResp:
        def __init__(self, r): self.response = r
    monkeypatch.setattr(functions, "ollama_client", type(
        "D", (), {"generate": lambda self, **kw: {"response": "1 some text"}})())
    assert functions.llama_classify_tweet_generate("t") == 1
    monkeypatch.setattr(functions, "ollama_client", type(
        "D", (), {"generate": lambda self, **kw: {"response": "0 nope"}})())
    assert functions.llama_classify_tweet_generate("t") == 0


def test_classify_all_tweets_parallel(monkeypatch):
    # stub llama_classify_tweet_generate
    monkeypatch.setattr(functions, "llama_classify_tweet_generate", lambda t: int(t))
    out = functions.classify_all_tweets_parallel(["0", "1", "0"], max_workers=2)
    assert out == [0, 1, 0]

def test_evaluate_model_cv(tmp_path, monkeypatch):
    # Use a tiny dataset and DummyClassifier
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    model = DummyClassifier()
    param_grid = {"strategy": ["most_frequent"]}
    # redirect save_dir
    save_dir = tmp_path / "models"
    # Run
    functions.evaluate_model_cv(model, param_grid, X, y,
                                model_name="Dummy", pipeline_name="pipe", cv=2,
                                save_dir=str(save_dir))
    # one result appended
    assert len(functions.results_list) == 1
    res = functions.results_list[0]
    assert res["Pipeline"] == "pipe"
    assert res["Model"] == "Dummy"
    # model file exists
    files = list(save_dir.glob("pipe_Dummy.joblib"))
    assert len(files) == 1


def test_plot_roc_curve_cv():
    # simple binary dataset
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    # dummy model with predict_proba
    class M:
        def fit(self, X, y): pass
        def predict_proba(self, X): 
            # assign .2 to class 0, .8 to class 1
            return np.tile([.2, .8], (len(X),1))
    fig, ax = plt.subplots()
    functions.plot_roc_curve_cv(M(), X, y, cv=2, ax=ax)
    # expect at least one line plotted (excluding diagonal)
    lines = ax.get_lines()
    assert len(lines) >= 1  

def test_plot_learning_curve_cv():
    # tiny dataset and DummyClassifier
    est = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    fig, ax = plt.subplots()
    functions.plot_learning_curve_cv(est, X, y, cv=2, ax=ax)
    # train and val curves plotted
    lines = ax.get_lines()
    assert any(line.get_label() == "Train Mean" for line in lines)
    assert any(line.get_label() == "Val Mean" for line in lines)