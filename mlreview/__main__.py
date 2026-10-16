import os
import sys
from .engine import review
from .loader import load_source

SKIP = {".venv", "venv", ".git", "__pycache__", "node_modules"}

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
    if len(sys.argv) != 2:
        print("Usage: python -m mlreview <file-or-folder>")
        sys.exit(2)

    total = 0
    for path in collect(sys.argv[1]):
        try:
            source, labels = load_source(path)
            findings = review(source)
        except (SyntaxError, ValueError, UnicodeDecodeError) as e:
            print(f"Skipped {path}: {e}")
            continue
        for f in findings:
            where = f"{path}:{f.line}"
            if labels and f.line - 1 < len(labels):
                where += f" ({labels[f.line - 1]})"
            print(f"{where}  [{f.rule_id}] ({f.severity})")
            print(f"    {f.message}")
            print(f"    Fix: {f.fix}\n")
        total += len(findings)

    print(f"{total} issue(s) found." if total else "No issues found.")
    sys.exit(1 if total else 0)

if __name__ == "__main__":
    main()