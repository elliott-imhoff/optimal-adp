# Copilot Instructions

## Project Purpose
This repository builds a simulator to optimize Average Draft Position (ADP) for the 2024 fantasy football draft using actual statistics from the 2024 season and my league settings. See `docs/problem_statement.md` for full problem description.

---

## Coding Style & Rules
- Write **small, pure functions** with type hints and docstrings. Use python 3.10 docstrings
- Keep **I/O**, **scoring**, **draft simulation**, **regret calculation**, and **ADP updates** in separate modules.
- No global state — pass configuration explicitly.
- Use deterministic behavior for reproducible results (fixed RNG seeds).
- Include unit tests in `tests/` for any new logic.
- Use `csv` library for loading / writing the csv data
- Format with `pre-commit`, `black` and `flake8`; static type check with `mypy`. Make sure to git add all new files before running pre-commit
- Use `logging` library to add logging statements when applicable to give progress information to the user. Keep debug information behind `--verbose` custom log level
- Use python3.10 style type hints e.g. `list[]` instead of `List[]`
- Use `poetry` to run all commands

---

## Development Process
1. **Plan before coding:** When implementing a new feature, outline the steps, target files, and functions first.
2. **Test first:** Write unit tests against a small fixture dataset before writing the implementation.
3. **Small iterations:** Implement one function at a time, run tests, and commit.
