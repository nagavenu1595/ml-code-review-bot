import json
from mlreview.engine import review
from mlreview.loader import load_source
from mlreview.rules.leakage import FitBeforeSplit

UNRELATED = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
demo = scaler.fit_transform(toy)
a, b = train_test_split(X, y, random_state=1)
"""

def test_unrelated_data_not_flagged():
    assert review(UNRELATED, [FitBeforeSplit()]) == []

def test_magic_inside_if(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": [
        "import sys\n", "if 'google.colab' in sys.modules:\n",
        "    %pip install foo\n", "x = 1"]}]}
    p = tmp_path / "m.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    src, _ = load_source(str(p))
    compile(src, "x", "exec")

from mlreview.rules.torch_eval import MissingEval, MissingNoGrad

SCALED_DEMO = """
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = SVC().fit(X_scaled, y)
a, b = train_test_split(X, y, random_state=1)
"""
KERAS_LOOP = """
import torch
for i, s in enumerate(test_strings):
    print(s)
"""

def test_scaled_output_not_split():
    assert review(SCALED_DEMO, [FitBeforeSplit()]) == []

def test_non_loader_loop_not_flagged():
    assert review(KERAS_LOOP, [MissingEval(), MissingNoGrad()]) == []

CHAINED = """
enc = OneHotEncoder()
X_enc = enc.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enc)
a, b = train_test_split(X_scaled, y, random_state=1)
"""
SPLIT_FIRST = """
a, b = train_test_split(X, y, random_state=1)
enc = OneHotEncoder()
A = enc.fit_transform(a)
"""

def test_chained_transformers_both_flagged():
    f = review(CHAINED, [FitBeforeSplit()])
    assert [x.line for x in f] == [3, 5]

def test_split_first_is_clean():
    assert review(SPLIT_FIRST, [FitBeforeSplit()]) == []