import ast
from .rules.leakage import FitBeforeSplit
from .rules.fit_on_test import FitOnTest
from .rules.seeds import SplitSeed
from .rules.stratify import SplitStratify
from .rules.torch_eval import MissingEval, MissingNoGrad
from .rules.estimator_seed import EstimatorSeed
from .rules.timeseries import RandomSplitTimeSeries

RULES = [FitBeforeSplit(), FitOnTest(), SplitSeed(), SplitStratify(),
         MissingEval(), MissingNoGrad(), EstimatorSeed(), RandomSplitTimeSeries()]

def review(source: str, rules=None):
    tree = ast.parse(source)
    findings = [f for r in (rules or RULES) for f in r.check(tree, source)]
    return sorted(findings, key=lambda f: f.line)