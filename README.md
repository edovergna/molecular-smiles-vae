# Molecular SMILES VAE (PyTorch)

A minimal **Variational Autoencoder (VAE)** for learning a latent space of **SMILES strings** and generating novel small molecules.

---

## Project Objective

This repository is **generation-first**, not reconstruction-first.

The model is intentionally optimized to:
- Learn a **useful and expressive latent space**
- Generate **valid, unique, and novel molecules**
- Encourage **latent usage and diversity**

As a result, **exact SMILES reconstruction** (encode → decode → identical string) can be considered weak. This is **expected behavior** and is due to my approach, not a bug.

If your primary goal is *perfect reconstruction*, this setup is **not optimized for that use case**.


## Dataset
This project uses the **QM9 dataset**, which contains approximately **134k small organic molecules** with up to **nine heavy atoms** (C, N, O, F). Only the **canonical SMILES representations** are used to train a character-level variational autoencoder for molecular generation.


## Requirements

- Python **3.10+**
- PyTorch
- RDKit

All required Python dependencies are listed in `requirements.txt`.


## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


## Data Format

Training data must be a **plain text file** containing **one SMILES string per line**.

Example:

```text
CCO
CC(=O)O
c1ccccc1
```

Invalid SMILES will break training.


## Training

Train the VAE using:

```bash
python train.py
```

### Model Details

- Character-level tokenization
- Encoder: GRU
- Decoder: GRU
- Latent variable with KL regularization
- Teacher forcing during training

Model checkpoints and logs are saved automatically during training.


## Sampling & Evaluation

### Sample molecules from the prior

```bash
python sample_prior.py
```

This samples random latent vectors and decodes them into SMILES strings.


### Visualize generated molecules

```bash
python visualize_generated.py
```

Generates a small grid of molecules for visual inspection.


### Latent space smoothness check

```bash
python latent_check.py
```

Interpolates between latent vectors to evaluate:
- Smoothness of transitions
- Semantic continuity
- Latent collapse vs usage


## Reconstruction vs Generation Tradeoff

Increasing any of the following:
- Word / token dropout
- KL pressure or capacity scheduling
- Sampling temperature or stochastic decoding

Typically results in:
- **Higher diversity**
- **Better latent utilization**
- **Worse exact SMILES reconstruction**

This tradeoff is **intentional** in this repository.


## Notes

- This is a **research / experimentation** codebase
- Not production-hardened
- Metrics are intentionally lightweight
- Readability and clarity are prioritized over extreme optimization

Final take: this project really has taught me a lot on VAEs. The use of different losses, training parameters and approaches to them really changes the outcome, more than I could imagine. In the end, I settled with a model that is far from perfect (but still generates with quite good results) because this project was meant as a learning experience on my end. I was also testing the performance of my new laptop and I cannot complain, my older one would have required 100x the time for each experiment. Lastly, I want to thank anybody who reads this far (props to you), it really means a lot to me.

And now, onto the next challenge!

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE.txt) file for details
