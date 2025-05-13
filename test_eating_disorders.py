import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

#import prueba
from functions import convert_emojis

    

def test_convert_emojis():
    out = convert_emojis("Hola 😀")
    assert ":grinning_face:" in out
    assert out == "Hola :grinning_face:"