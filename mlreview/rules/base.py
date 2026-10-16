import ast
from ..models import Finding

class Rule:
    id = ""
    def check(self, tree: ast.AST, source: str) -> list[Finding]:
        raise NotImplementedError