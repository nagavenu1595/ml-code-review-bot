import ast
from .rules.leakage import FitBeforeSplit
from .rules.fit_on_test import FitOnTest
from .rules.seeds import SplitSeed
from .rules.stratify import SplitStratify
from .rules.torch_eval import MissingEval, MissingNoGrad

RULES = [FitBeforeSplit(), FitOnTest(), SplitSeed(),
         SplitStratify(), MissingEval(), MissingNoGrad()]

def review(source: str, rules=None):
    tree = ast.parse(source)
    findings = [f for r in (rules or RULES) for f in r.check(tree, source)]
    return sorted(findings, key=lambda f: f.line)