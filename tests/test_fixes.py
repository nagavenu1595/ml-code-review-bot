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