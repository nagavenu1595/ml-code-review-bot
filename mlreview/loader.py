import ast
import json

def _clean(line):
    s = line.lstrip()
    if s.startswith(("%", "!")):
        return line[: len(line) - len(s)] + "pass"
    return line

def _safe(cell_lines):
    try:
        ast.parse("\n".join(cell_lines))
        return cell_lines
    except SyntaxError:
        return [""] * len(cell_lines)   # skip cells that aren't valid Python

def load_source(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if not path.endswith(".ipynb"):
        return raw, None

    nb = json.loads(raw)
    lines, labels, n = [], [], 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        n += 1
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for line in _safe([_clean(l) for l in src.splitlines()]):
            lines.append(line)
            labels.append(f"cell {n}")
        lines.append("")
        labels.append(f"cell {n}")
    return "\n".join(lines), labels