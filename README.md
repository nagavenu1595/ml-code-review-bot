# mlreview

A static code reviewer for machine learning code. It finds ML-specific mistakes that general linters miss, such as data leakage, wrong split order, and unseeded randomness.

It reads your `.py` files and Jupyter notebooks (`.ipynb`) **without running them**, so it needs no data, no GPU, and none of your ML libraries installed.

## Why

A model can score 95% in testing and fail in production because of a small mistake made weeks earlier. Most of these mistakes are easy to catch automatically, but general-purpose tools like SonarQube or a standard linter don't know that scaling must happen after the train/test split.

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)        # the scaler has now seen the test data
X_train, X_test = train_test_split(X, y)
```

`mlreview` flags this and tells you how to fix it.

## Rules

| ID | Severity | What it catches |
|---|---|---|
| ML001 | high | A scaler, encoder, imputer, or feature selector fit on data that later reaches `train_test_split` or cross-validation |
| ML002 | high | `fit` / `fit_transform` called on test data instead of `transform` |
| ML003 | medium | `train_test_split` without `random_state` |
| ML004 | low | Split feeding a classifier without `stratify` (hidden by default) |
| ML005 | high | PyTorch evaluation loop with no `model.eval()` |
| ML006 | medium | PyTorch evaluation loop with no `torch.no_grad()` |
| ML007 | medium | Randomized estimators (random forests, KMeans, XGBoost, ...) created without `random_state` |
| ML008 | medium | Random `train_test_split` on data that looks time-ordered |

## Install

Requires Python 3.9 or newer. There are no runtime dependencies.

```bash
git clone https://github.com/nagavenu1595/mlreview
cd mlreview
```

## Usage

Review a single file or notebook:

```bash
python -m mlreview train.py
python -m mlreview analysis.ipynb
```

Review a whole folder (`.venv`, `.git`, and similar folders are skipped):

```bash
python -m mlreview path/to/project
```

Show low-severity findings too:

```bash
python -m mlreview path/to/project --all
```

### Example output

```
sample_messy.py:14  [ML001] (high)
    `scaler.fit_transform` output reaches a split (line 16). Test-set statistics leak into training.
    Fix: Split first, then fit on X_train only and call transform on X_test.

sample_messy.py:16  [ML003] (medium)
    `train_test_split` has no `random_state`, so results change every run.
    Fix: Pass `random_state=42` (or any fixed integer).
```

For notebooks, each finding also shows the cell number, for example `analysis.ipynb:42 (cell 7)`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | No issues found |
| 1 | Issues found |
| 2 | Usage error |

This makes it easy to use in CI: a non-zero exit code fails the build.

## Limitations

`mlreview` uses static analysis and heuristics. Please read these before trusting a result:

- **It can be wrong in both directions.** It may flag code that is actually fine, and it may miss real bugs.
- **Data flow is limited.** It follows data through simple variable assignments in one file. It does not follow data through function calls or across files.
- **Separate `fit` and `transform` calls are not tracked.** `scaler.fit(X)` followed by a later `scaler.transform(X)` on data that gets split is not caught yet.
- **Task type is guessed.** ML004 and ML008 infer classification or time-ordered data from the code. Treat those as hints.
- **Notebook cells that are not valid Python are skipped.** Jupyter magics such as `%pip` are neutralized so the rest of the cell still parses.

Findings are suggestions for review, not proof of a bug.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install pytest
python -m pytest -q
```

### Adding a rule

1. Create a file in `mlreview/rules/` with a class that subclasses `Rule` and implements `check(tree, source)`, returning a list of `Finding` objects.
2. Register it in the `RULES` list in `mlreview/engine.py`.
3. Add tests in `tests/`: one snippet the rule should flag, and one similar-looking snippet it should not. The second kind is what prevents false alarms.

Run a single rule in a test with `review(source, [YourRule()])` so that other rules do not interfere.

## Project layout

```
mlreview/
  __main__.py     command-line interface
  engine.py       runs all rules
  loader.py       reads .py and .ipynb files
  models.py       Finding data class
  rules/          one file per rule family
tests/
```

## License

MIT
