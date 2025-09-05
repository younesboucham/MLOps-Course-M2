PY=python
VENV=.venv
APP=ml_microservice/app.py

init:
	python3 -m venv $(VENV) && . $(VENV)/bin/activate && pip install -U pip && pip install -r ml_microservice/requirements.txt

run:
	uvicorn ml_microservice.app:app --host 0.0.0.0 --port 8000 --reload

train:
	$(PY) src/train.py

evaluate:
	$(PY) src/evaluate.py

test:
	PYTHONPATH=. pytest -q tests

lint:
	flake8 .

format:
	ruff check --fix . || true

build:
	docker build -t mlops-microservice:latest ml_microservice/
