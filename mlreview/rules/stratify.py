import ast
from ..models import Finding
from .base import Rule
from .leakage import _name

class SplitStratify(Rule):
    id = "ML004"

    def check(self, tree, source):
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) == "train_test_split":
                kws = {k.arg for k in node.keywords}
                if len(node.args) >= 2 and not kws & {"stratify", "shuffle", None}:
                    out.append(Finding(
                        rule_id=self.id, severity="low", line=node.lineno,
                        message="Split has no `stratify`. If this is classification, "
                                "class proportions may differ between train and test.",
                        fix="Pass `stratify=y` for classification tasks.",
                    ))
        return out