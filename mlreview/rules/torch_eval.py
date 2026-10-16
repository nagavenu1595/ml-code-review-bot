import ast
import re
from ..models import Finding
from .base import Rule

EVAL_NAME = re.compile(r"(^|_)(val|valid|validation|test|eval)(_|$)", re.I)

def _uses_torch(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.startswith("torch") for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("torch"):
            return True
    return False

def _first_eval_loop(tree):
    loops = []
    for n in ast.walk(tree):
        if isinstance(n, ast.For):
            names = [x.id for x in ast.walk(n.iter) if isinstance(x, ast.Name)]
            if any(EVAL_NAME.search(x) for x in names):
                loops.append(n.lineno)
    return min(loops) if loops else None

class MissingEval(Rule):
    id = "ML005"

    def check(self, tree, source):
        if not _uses_torch(tree):
            return []
        line = _first_eval_loop(tree)
        has_eval = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "eval" for n in ast.walk(tree))
        if line and not has_eval:
            return [Finding(
                rule_id=self.id, severity="high", line=line,
                message="Evaluation loop found but `model.eval()` is never called. "
                        "Dropout and batch norm stay in training mode.",
                fix="Call `model.eval()` before evaluating.")]
        return []

class MissingNoGrad(Rule):
    id = "ML006"
    NAMES = {"no_grad", "inference_mode"}

    def check(self, tree, source):
        if not _uses_torch(tree):
            return []
        line = _first_eval_loop(tree)
        has_ng = any(
            (isinstance(n, ast.Attribute) and n.attr in self.NAMES) or
            (isinstance(n, ast.Name) and n.id in self.NAMES)
            for n in ast.walk(tree))
        if line and not has_ng:
            return [Finding(
                rule_id=self.id, severity="medium", line=line,
                message="Evaluation loop without `torch.no_grad()`. "
                        "This wastes memory and compute tracking gradients.",
                fix="Wrap evaluation in `with torch.no_grad():`.")]
        return []