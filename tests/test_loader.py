import json
from mlreview.loader import load_source
from mlreview.engine import review
from mlreview.rules.seeds import SplitSeed

def test_notebook(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# hi"]},
        {"cell_type": "code", "source": [
            "%matplotlib inline\n",
            "from sklearn.model_selection import train_test_split\n"]},
        {"cell_type": "code", "source": ["a, b = train_test_split(X, y)"]},
    ]}
    p = tmp_path / "t.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    src, labels = load_source(str(p))
    f = review(src, [SplitSeed()])
    assert len(f) == 1 and labels[f[0].line - 1] == "cell 2"