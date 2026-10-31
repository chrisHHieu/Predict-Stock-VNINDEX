.PHONY: install lint test fetch validate features train evaluate drift serve all clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

fetch:
	$(PYTHON) -m vnindex_forecast fetch

validate:
	$(PYTHON) -m vnindex_forecast validate

features:
	$(PYTHON) -m vnindex_forecast features

train:
	$(PYTHON) -m vnindex_forecast train

evaluate:
	$(PYTHON) -m vnindex_forecast evaluate

drift:
	$(PYTHON) -m vnindex_forecast drift

serve:
	$(PYTHON) -m vnindex_forecast serve

all: fetch validate features train evaluate drift

clean:
	rm -rf models reports/metrics.json reports/drift.json reports/figures/*.png
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
