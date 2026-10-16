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
        split_lines, fits, tvars = [], [], set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if _name(node.value.func) in TRANSFORMERS:
                    tvars.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if isinstance(node, ast.Call):
                if _name(node.func) == "train_test_split":
                    split_lines.append(node.lineno)
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr in ("fit", "fit_transform")
                        and isinstance(f.value, ast.Name)):
                    fits.append((node.lineno, f.value.id))
        if not split_lines:
            return []
        first_split = min(split_lines)
        return [
            Finding(
                rule_id=self.id, severity="high", line=line,
                message=f"`{var}` is fit before train_test_split (line {first_split}). "
                        "Test-set statistics leak into training.",
                fix="Split first, then fit on X_train only and call transform on X_test.",
            )
            for line, var in fits if var in tvars and line < first_split
        ]