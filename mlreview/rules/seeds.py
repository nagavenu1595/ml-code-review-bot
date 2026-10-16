import ast
from ..models import Finding
from .base import Rule
from .leakage import _name

class SplitSeed(Rule):
    id = "ML003"

    def check(self, tree, source):
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) == "train_test_split":
                if not any(k.arg == "random_state" or k.arg is None for k in node.keywords):
                    out.append(Finding(
                        rule_id=self.id, severity="medium", line=node.lineno,
                        message="`train_test_split` has no `random_state`, so results change every run.",
                        fix="Pass `random_state=42` (or any fixed integer).",
                    ))
        return out