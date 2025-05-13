import os
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
