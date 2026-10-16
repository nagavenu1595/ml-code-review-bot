from mlreview.engine import review
from mlreview.rules.leakage import FitBeforeSplit
from mlreview.rules.estimator_seed import EstimatorSeed
from mlreview.rules.timeseries import RandomSplitTimeSeries

CV_LEAK = """
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
scores = cross_val_score(model, X_s, y)
"""
def test_cv_leak():
    f = review(CV_LEAK, [FitBeforeSplit()])
    assert len(f) == 1 and f[0].line == 3

def test_unseeded_model():
    f = review("m = RandomForestClassifier(n_estimators=100)\n", [EstimatorSeed()])
    assert len(f) == 1 and f[0].line == 1

def test_seeded_model_ok():
    assert review("m = RandomForestClassifier(random_state=42)\n", [EstimatorSeed()]) == []

def test_deterministic_model_ok():
    assert review("m = LinearRegression()\n", [EstimatorSeed()]) == []

TS = """
df['d'] = pd.to_datetime(df['d'])
a, b = train_test_split(X, y, random_state=1)
"""
def test_timeseries_flagged():
    f = review(TS, [RandomSplitTimeSeries()])
    assert len(f) == 1 and f[0].line == 3

def test_shuffle_false_ok():
    src = TS.replace("random_state=1", "shuffle=False")
    assert review(src, [RandomSplitTimeSeries()]) == []

def test_no_time_hint_ok():
    src = "a, b = train_test_split(X, y, random_state=1)\n"
    assert review(src, [RandomSplitTimeSeries()]) == []