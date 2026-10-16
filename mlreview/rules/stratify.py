import ast
from ..models import Finding
from .base import Rule
from .leakage import _name

CLASSIFIERS = {"LogisticRegression", "LogisticRegressionCV", "SVC", "LinearSVC",
               "GaussianNB", "MultinomialNB", "BernoulliNB"}

def _is_clf(name):
    return name in CLASSIFIERS or (name or "").endswith("Classifier")

def _has_clf(node):
    return any(isinstance(n, ast.Call) and _is_clf(_name(n.func))
               for n in ast.walk(node))

class SplitStratify(Rule):
    id = "ML004"

    def check(self, tree, source):
        clf_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _has_clf(node.value):
                clf_vars.update(t.id for t in node.targets if isinstance(t, ast.Name))

        fit_args = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "fit"):
                recv = node.func.value
                if (isinstance(recv, ast.Name) and recv.id in clf_vars) or _has_clf(recv):
                    fit_args |= {a.id for a in node.args if isinstance(a, ast.Name)}

        out = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and _name(node.value.func) == "train_test_split"):
                continue
            call = node.value
            kws = {k.arg for k in call.keywords}
            if len(call.args) < 2 or kws & {"stratify", "shuffle", None}:
                continue
            targets = {x.id for t in node.targets for x in ast.walk(t)
                       if isinstance(x, ast.Name)}
            if targets & fit_args:
                out.append(Finding(
                    rule_id=self.id, severity="low", line=call.lineno,
                    message="Split feeds a classifier but has no `stratify`. Class "
                            "proportions may differ between train and test.",
                    fix="Pass `stratify=y`.",
                ))
        return out