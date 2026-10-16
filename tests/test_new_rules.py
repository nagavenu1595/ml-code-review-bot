from mlreview.engine import review
from mlreview.rules.stratify import SplitStratify
from mlreview.rules.torch_eval import MissingEval, MissingNoGrad

NO_STRAT = """
from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(X, y, random_state=1)
model = RandomForestClassifier()
model.fit(a, c)
"""
STRAT = NO_STRAT.replace("random_state=1", "random_state=1, stratify=y")
REGRESSION = """
from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(X, y, random_state=1)
LinearRegression().fit(a, c)
"""
BAD_EVAL = """
import torch
for x, y in val_loader:
    out = model(x)
"""
GOOD_EVAL = """
import torch
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        out = model(x)
"""

def test_missing_stratify():
    f = review(NO_STRAT, [SplitStratify()])
    assert len(f) == 1 and f[0].line == 3

def test_stratify_ok():
    assert review(STRAT, [SplitStratify()]) == []

def test_regression_not_flagged():
    assert review(REGRESSION, [SplitStratify()]) == []

def test_eval_missing():
    f = review(BAD_EVAL, [MissingEval(), MissingNoGrad()])
    assert sorted(x.rule_id for x in f) == ["ML005", "ML006"]
    assert all(x.line == 3 for x in f)

def test_eval_ok():
    assert review(GOOD_EVAL, [MissingEval(), MissingNoGrad()]) == []