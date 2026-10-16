import json

def load_source(path):
    """Return (source, labels). labels[i] describes line i+1, e.g. 'cell 2'."""
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
        for line in src.splitlines():
            if line.lstrip().startswith(("%", "!")):
                line = ""          # Jupyter magics aren't valid Python
            lines.append(line)
            labels.append(f"cell {n}")
        lines.append("")
        labels.append(f"cell {n}")
    return "\n".join(lines), labels