import os
import sys
from .engine import review
from .loader import load_source

SKIP = {".venv", "venv", ".git", "__pycache__", "node_modules"}
RANK = {"low": 0, "medium": 1, "high": 2}

def collect(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP]
        files += [os.path.join(root, n) for n in names
                  if n.endswith((".py", ".ipynb"))]
    return files

def main():
    args = sys.argv[1:]
    show_all = "--all" in args
    paths = [a for a in args if not a.startswith("--")]
    if len(paths) != 1:
        print("Usage: python -m mlreview <file-or-folder> [--all]")
        sys.exit(2)

    min_rank = 0 if show_all else 1
    total = hidden = 0
    for path in collect(paths[0]):
        try:
            source, labels = load_source(path)
            findings = review(source)
        except (SyntaxError, ValueError, UnicodeDecodeError) as e:
            print(f"Skipped {path}: {e}")
            continue
        for f in findings:
            if RANK[f.severity] < min_rank:
                hidden += 1
                continue
            where = f"{path}:{f.line}"
            if labels and f.line - 1 < len(labels):
                where += f" ({labels[f.line - 1]})"
            print(f"{where}  [{f.rule_id}] ({f.severity})")
            print(f"    {f.message}")
            print(f"    Fix: {f.fix}\n")
            total += 1

    print(f"{total} issue(s) found." if total else "No issues found.")
    if hidden:
        print(f"({hidden} low-severity hidden, use --all to show)")
    sys.exit(1 if total else 0)

if __name__ == "__main__":
    main()