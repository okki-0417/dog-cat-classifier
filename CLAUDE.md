# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dog vs cat image classifier using PyTorch. Goal is to learn ML fundamentals by progressively leveling up the implementation:

1. Pretrained model (current) → 2. Transfer learning → 3. Breed classification → 4. CNN from scratch → 5. Advanced architectures

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run classification
python classify_dog_cat.py <image_path>

# Inspect model weights
python inspect_model.py
```

## Architecture

- **classify_dog_cat.py**: Main classifier. Uses pretrained ResNet18, sums probabilities across ImageNet dog classes (151-268) and cat classes (281-285).
- **inspect_model.py**: Utility to examine model weight structure.
- Model weights cached at `~/.cache/torch/hub/checkpoints/`
