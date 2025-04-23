.ONESHELL:

setup: uv.lock pyproject.toml
	uv sync --dev
	powershell -ExecutionPolicy Bypass -NoExit -Command ". .venv\Scripts\activate"
	pre-commit install

activate: .venv
	powershell -ExecutionPolicy Bypass -NoExit -Command ". .venv\Scripts\activate"

test:
	pytest

style: .pre-commit-config.yaml
	pre-commit run --all-files