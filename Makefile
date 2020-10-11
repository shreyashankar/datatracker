.PHONY: help test lint run

# Makefile variables
VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python3

# Include your variables here

.DEFAULT: help
help:
	@echo "make prepare-dev"
	@echo "       prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       run pylint and mypy"
	@echo "make run"
	@echo "       run project"

# Install dependencies whenever setup.py is changed.
venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: setup.py
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	rm -rf ./*.egg-info
	touch $(VENV_NAME)/bin/activate

test: venv
	${PYTHON} -m pytest

lint: venv
	${PYTHON} -m pylint
	${PYTHON} -m mypy

run: venv
	${PYTHON} main.py