# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Character-level RNN text generator using PyTorch. Learning project to understand how language models work.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train and generate
python -m text_generator <text_file> --epochs 50
```

## Architecture

- **model.py**: CharRNN class - Embedding → LSTM → Linear
- **data.py**: TextDataset - converts text to character indices
- **train.py**: Training loop with CrossEntropyLoss
- **generate.py**: Text generation with temperature sampling
