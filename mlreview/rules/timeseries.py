import ast
from ..models import Finding
from .base import Rule
from .leakage import _name

HINTS = ("to_datetime", "parse_dates", "forecast")

class RandomSplitTimeSeries(Rule):
    id = "ML008"

    def check(self, tree, source):
        low = source.lower()
        if not any(h in low for h in HINTS) or "timeseriessplit" in low:
            return []
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) == "train_test_split":
                kws = {k.arg for k in node.keywords}
                if "shuffle" not in kws and None not in kws:
                    out.append(Finding(
                        rule_id=self.id, severity="medium", line=node.lineno,
                        message="Data looks time-ordered, but `train_test_split` shuffles "
                                "rows randomly. Future data can leak into training.",
                        fix="For time-ordered data, split by date or use "
                            "`shuffle=False` / `TimeSeriesSplit`.",
                    ))
        return out