# Development Guide

## Prerequisites

- Python 3.11.x
- Poetry 2.x
- Git

Verify your environment:

```bash
python3.11 --version
poetry --version
git --version
```

---

# Clone the Repository

```bash
git clone git@github.com:anjan-tummalapalli/ai_without_rag.git
cd ai_without_rag
```

---

# Create a Poetry Environment

Use Python 3.11 explicitly.

```bash
poetry env use $(which python3.11)
```

Verify:

```bash
poetry env info
```

---

# Install Dependencies

```bash
poetry install
```

If you receive:

```
pyproject.toml changed significantly since poetry.lock was last generated
```

run:

```bash
poetry lock
poetry install
```

If `poetry.lock` changes, commit the updated lock file.

```bash
git add poetry.lock
git commit -m "Regenerate poetry.lock"
git push
```

---

# Activate Virtual Environment

```bash
poetry shell
```

or

```bash
source $(poetry env info --path)/bin/activate
```

---

# Run the CLI

```bash
ai-cli -q "Hello"
```

Example:

```bash
ai-cli -q "Explain Kubernetes Operators"
```

---

# Running Tests

Run all tests

```bash
pytest
```

Verbose

```bash
pytest -v
```

Stop on first failure

```bash
pytest -x
```

Run a single file

```bash
pytest tests/test_providers.py
```

Run a single test

```bash
pytest tests/test_providers.py::test_openai_provider
```

Run tests matching a keyword

```bash
pytest -k openai
```

---

# Code Coverage

Generate coverage

```bash
pytest --cov=src --cov-report=term-missing
```

Generate HTML report

```bash
pytest --cov=src --cov-report=html
```

Open the report

macOS

```bash
open htmlcov/index.html
```

Linux

```bash
xdg-open htmlcov/index.html
```

Coverage only

```bash
coverage run -m pytest
coverage report
```

Detailed report

```bash
coverage report -m
```

Generate HTML

```bash
coverage html
```

---

# Ruff

Check formatting

```bash
ruff check .
```

Automatically fix issues

```bash
ruff check . --fix
```

Format project

```bash
ruff format .
```

---

# Black

Format source

```bash
black .
```

Check formatting only

```bash
black --check .
```

---

# MyPy

Run static type checking

```bash
mypy src
```

---

# Useful Poetry Commands

Show Poetry version

```bash
poetry --version
```

Show environment

```bash
poetry env info
```

List environments

```bash
poetry env list
```

Remove current environment

```bash
poetry env remove python
```

Install dependencies

```bash
poetry install
```

Update lock file

```bash
poetry lock
```

Update dependencies

```bash
poetry update
```

Validate project

```bash
poetry check
```

Show dependency tree

```bash
poetry show --tree
```

---

# GitHub Actions Troubleshooting

If CI fails with:

```
pyproject.toml changed significantly since poetry.lock was last generated
```

Solution:

```bash
poetry lock
git add poetry.lock
git commit -m "Regenerate poetry.lock"
git push
```

This indicates that `pyproject.toml` has changed without regenerating `poetry.lock`.

---

# Cleaning the Environment

Remove cache

```bash
poetry cache clear pypi --all
```

Remove virtual environment

```bash
poetry env remove python
```

Recreate environment

```bash
poetry env use $(which python3.11)
poetry install
```

---

# Project Structure

```
src/
    ai_cli/
        providers/
        rag/
        telemetry/
        utils/

tests/
docs/
README.md
pyproject.toml
poetry.lock
```

---

# Recommended Development Workflow

```bash
git pull

poetry install

ruff check . --fix

ruff format .

pytest

pytest --cov=src --cov-report=term-missing

git status

git add .

git commit -m "Describe your changes"

git push
```