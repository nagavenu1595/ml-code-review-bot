import ast
from ..models import Finding
from .base import Rule

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer", "KNNImputer",
    "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "PCA",
    "TfidfVectorizer", "CountVectorizer", "SelectKBest", "SelectPercentile",
    "SelectFromModel", "RFE", "TargetEncoder", "CatBoostEncoder",
    "LeaveOneOutEncoder", "WOEEncoder", "JamesSteinEncoder",
}

def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

class FitBeforeSplit(Rule):
    id = "ML001"

    def check(self, tree, source):
        assigns = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign)),
                         key=lambda n: n.lineno)

        tvars = set()
        for a in assigns:
            if isinstance(a.value, ast.Call) and _name(a.value.func) in TRANSFORMERS:
                tvars.update(t.id for t in a.targets if isinstance(t, ast.Name))

        SINKS = {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict"}
        splits = [(n.lineno, {a.id for a in n.args if isinstance(a, ast.Name)})
                  for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and _name(n.func) in SINKS]

        out = []
        for a in assigns:
            f = a.value.func if isinstance(a.value, ast.Call) else None
            if not (isinstance(f, ast.Attribute) and f.attr == "fit_transform"
                    and isinstance(f.value, ast.Name) and f.value.id in tvars):
                continue

            reached = set()
            for t in a.targets:
                reached |= _names(t)
            for b in assigns:
                if b.lineno > a.lineno and (_names(b.value) - tvars) & reached:
                    for t in b.targets:
                        reached |= _names(t)

            later = [sl for sl, sn in splits if sl > a.lineno and sn & reached]
            if later:
                out.append(Finding(
                    rule_id=self.id, severity="high", line=a.lineno,
                    message=f"`{f.value.id}.fit_transform` output reaches a split or CV"
                            f"(line {min(later)}). Test-set statistics leak into training.",
                    fix="Split first, then fit on X_train only and call transform on X_test.",
                ))
        return out