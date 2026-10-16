import ast
from ..models import Finding
from .base import Rule
from .leakage import _name

RANDOM_MODELS = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
    "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor", "KMeans", "MLPClassifier",
    "MLPRegressor", "SGDClassifier", "SGDRegressor",
    "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
}

class EstimatorSeed(Rule):
    id = "ML007"

    def check(self, tree, source):
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _name(node.func)
            kws = {k.arg for k in node.keywords}
            if "random_state" in kws or None in kws:
                continue
            shuffled_kfold = name in ("KFold", "StratifiedKFold") and any(
                k.arg == "shuffle" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in node.keywords)
            if name in RANDOM_MODELS or shuffled_kfold:
                out.append(Finding(
                    rule_id=self.id, severity="medium", line=node.lineno,
                    message=f"`{name}` has no `random_state`, so results vary between runs.",
                    fix="Pass `random_state=42` (or any fixed integer).",
                ))
        return out