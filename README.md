# Encoding Causal Macrovariables

This repository contains the code for [Encoding Causal Macrovariables](https://arxiv.org/abs/2111.14724).

## Data
- Natural climate data ('El Nino')
- Self-generated data ('Simulated')

## Experiments
1. Detecting macrovariables through the Causal Autoencoder (in the code termed VCE)
2. Transforming detected variables through double VAE and analysing causal relationships through additive noise model (ANM)

## Repo Structure
- Four python scripts to run experiments
- Two jupyter notebooks to visualise results
- `src` containing code base, helper functions and modules
- `data` with two subfolders for the two datasets
- `results` with two subfolders for the experiments on the two datasets
