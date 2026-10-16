from mlreview.engine import review
from mlreview.rules.leakage import FitBeforeSplit

BAD = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
"""

GOOD = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""

def test_flags_leakage():
    f = review(BAD, [FitBeforeSplit()])
    assert len(f) == 1 and f[0].rule_id == "ML001" and f[0].line == 5

def test_clean_code():
    assert review(GOOD, [FitBeforeSplit()]) == []