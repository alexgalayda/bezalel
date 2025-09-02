# TakeHome Assignment: Protein Contact Prediction

## Task Description
The goal is to predict a protein contact map - a binary matrix where each entry indicates whether a pair of residues (amino acids) are in spatial contact (Ca–Ca distance < 8A).

We build on ESM2, a transformer-based protein model that produces embeddings for amino acid sequences. In addition to sequence embeddings, structural information from similar proteins is incorporated to improve prediction accuracy.

## Approach

### Data preprocessing (most challenging part)
- Parsing protein structures from PDB files.
- Handling edge cases:
    - missing Ca atoms
    - multiple models
    - chains per file non-standard residues.
- Computing sequence embeddings using ESM2.
- Aligning sequences of the target protein and its structural neighbors to transfer structural information.

### Feature construction
- Residue embeddings from ESM2.
- Pairwise features for each residue pair (i, j):
    - embeddings Ei, Ej
    - element-wise product Ei * Ej

### Model architecture
- Modular interface for adding custom architectures.
- Default baseline: MLP network.

### Losses and metrics:
- Unified interface for extending with custom loss functions and metrics.
- Default example: Binary Cross Entropy (BCE) loss.


## Quick Start
### Data
Download `ml_interview_task_data.zip` and unzip it in data directory

### Podman

```bash
# Build and run Podman container
make docker-gpu

# Or step by step:
make docker-build-gpu
make docker-run-gpu
```

This will:
- Build a Docker image with CUDA 12.4.1 support
- Mount your current directory to the container
- Provide GPU access for training
- Install all dependencies automatically

### Local Development

#### Prerequisites
- Python 3.12+
- UV package manager

#### Installation
```bash
# Install dependencies
uv sync
```

### Usage

**Command Line:**
```bash
# Preprocess data (parce pdb)
uv run src/bezalel/pdb_loader.py

# Preprocess data (create embeddings)
uv run src/bezalel/embeddings.py

# Preprocess data (create qdrant)
uv run src/bezalel/nearest_search.py

# Train model
uv run run.py mode=train

# Test model
uv run run.py mode=test
```

## Monitoring

**TensorBoard:**
```bash
tensorboard --logdir logs
# Visit http://localhost:6006/
```
