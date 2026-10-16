import ast
import re
from ..models import Finding
from .base import Rule
from .leakage import TRANSFORMERS, _name

TEST_NAME = re.compile(r"(^|_)tests?($|_)")

class FitOnTest(Rule):
    id = "ML002"

    def check(self, tree, source):
        tvars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if _name(node.value.func) in TRANSFORMERS:
                    tvars.update(t.id for t in node.targets if isinstance(t, ast.Name))
        out = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("fit", "fit_transform")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in tvars
                    and node.args and isinstance(node.args[0], ast.Name)
                    and TEST_NAME.search(node.args[0].id.lower())):
                out.append(Finding(
                    rule_id=self.id, severity="high", line=node.lineno,
                    message=f"`{node.func.value.id}.{node.func.attr}` is called on "
                            f"`{node.args[0].id}`. The transformer is refit on test data.",
                    fix="Fit on training data only, then call `.transform()` on test data.",
                ))
        return out