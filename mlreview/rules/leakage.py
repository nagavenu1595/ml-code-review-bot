import ast
from ..models import Finding
from .base import Rule

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer",
    "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "PCA",
    "TfidfVectorizer", "CountVectorizer", "SelectKBest", "TargetEncoder",
}

def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

class FitBeforeSplit(Rule):
    id = "ML001"

    def check(self, tree, source):
        tvars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if _name(node.value.func) in TRANSFORMERS:
                    tvars.update(t.id for t in node.targets if isinstance(t, ast.Name))

        splits = [(n.lineno, {a.id for a in n.args if isinstance(a, ast.Name)})
                  for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and _name(n.func) == "train_test_split"]

        out = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            f = node.value.func
            if not (isinstance(f, ast.Attribute) and f.attr == "fit_transform"
                    and isinstance(f.value, ast.Name) and f.value.id in tvars):
                continue
            targets = {x.id for t in node.targets for x in ast.walk(t)
                       if isinstance(x, ast.Name)}
            later = [sl for sl, sn in splits if sl > node.lineno and targets & sn]
            if later:
                out.append(Finding(
                    rule_id=self.id, severity="high", line=node.lineno,
                    message=f"`{f.value.id}.fit_transform` output is split later "
                            f"(line {min(later)}). Test-set statistics leak into training.",
                    fix="Split first, then fit on X_train only and call transform on X_test.",
                ))
        return out