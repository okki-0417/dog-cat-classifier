.PHONY: run inspect setup

PYTHON = venv/bin/python

run:
	$(PYTHON) -m dog_cat_classifier $(IMG)

inspect:
	$(PYTHON) -m dog_cat_classifier.inspect_model

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
