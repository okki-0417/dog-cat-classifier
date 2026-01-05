.PHONY: setup train resume generate

PYTHON = venv/bin/python
TEXT = data/input.txt
EPOCHS = 20
START = First Citizen:
LENGTH = 500
MODEL = model.pth

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

train:
	$(PYTHON) -m text_generator $(TEXT) --epochs $(EPOCHS) --start "$(START)" --length $(LENGTH)

resume:
	$(PYTHON) -m text_generator $(TEXT) --resume $(MODEL) --epochs $(EPOCHS) --start "$(START)" --length $(LENGTH)

generate:
	$(PYTHON) -m text_generator --load $(MODEL) --start "$(START)" --length $(LENGTH)
