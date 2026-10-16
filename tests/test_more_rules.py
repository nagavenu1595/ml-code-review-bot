from mlreview.engine import review
from mlreview.rules.fit_on_test import FitOnTest
from mlreview.rules.seeds import SplitSeed

FIT_ON_TEST = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
"""

NO_SEED = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y)
"""

def test_fit_on_test():
    f = review(FIT_ON_TEST, [FitOnTest()])
    assert [x.rule_id for x in f] == ["ML002"] and f[0].line == 7

def test_missing_seed():
    f = review(NO_SEED, [SplitSeed()])
    assert [x.rule_id for x in f] == ["ML003"] and f[0].line == 3