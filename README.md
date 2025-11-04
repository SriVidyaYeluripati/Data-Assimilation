
# Data-Assimilation

A reproducible and flexible Python framework for data assimilation experiments on the Lorenz-63 system using neural-network architectures (MLP, GRU, LSTM) and fixed-point 3D-Var style assimilation.

##  Overview

This repository implements an experiment orchestration pipeline for comparing three model classes — MLP, GRU, and LSTM — in the context of a fixed‐point assimilation formulation (i.e., a 3D-Var-style correction with a neural encoder) applied to the chaotic Lorenz-63 system. The goal is to provide clean, reproducible code, datasets, visualisations, and diagnostics for data‐assimilation research.

Key features:

* Generation of synthetic data from the Lorenz-63 model for many trajectories.
* Observation operators of varying complexity (e.g., observing just X, or X&Y, or non-linear operator like X²).
* Comparison of baseline MLP, GRU‐encoder assimilation, and LSTM implementation under identical train/val/test splits.
* Fixed‐point assimilation: the encoder takes an observation sequence and outputs a correction (delta) to a background state.
* Full experiment orchestrator: model training, evaluation, diagnostics, saving of model artefacts and visualisation panels (side-by‐side reconstructions, error plots, failure cases) to a well-organised folder structure (e.g., `Documents/DA/lorenz63_project_final/`).
* Focus on avoiding data leakage and ensuring fair comparisons across architectures.

##  Repository structure

```
data/
  └── …                # synthetic trajectory datasets, observation sequences, etc  
results/
  └── …                # saved model weights, metrics, plots, visualisations  
src/
  ├── simulator.py      # Lorenz-63 data generator  
  ├── dataset.py        # AssimilationDataset, observation operator support  
  ├── models.py         # definitions of MLP, GRU‐encoder, LSTM architectures  
  ├── trainer.py        # orchestrator: training loops, experiment variants  
  ├── visualize.py      # functions for AIDA‐style panels, error plots, animations  
  └── utils.py          # utilities for save/load states, logging, config  
README.md              # this file  
```

##  Getting Started

### Requirements

* Python 3.8+
* PyTorch
* NumPy, Matplotlib (for visualisations)
* Optional: Jupyter/Colab for interactive notebooks or animations

You can install via:

```bash
pip install -r requirements.txt
```

*(If you don’t yet have a `requirements.txt`, consider adding one with the libraries you use.)*

### Running an Experiment

1. Generate synthetic data:

   ```bash
   python src/simulator.py --n_traj 1500 --train 1000 --val 500 --obs_mode xy
   ```

   This produces training and validation trajectories for the Lorenz-63 system under the chosen observation mode.

2. Train a model (e.g., GRU assimilation):

   ```bash
   python src/trainer.py --model gru --obs_mode xy --save_dir results/gru_xy
   ```

   This will train the GRU‐based fixed-point assimilation model, save the model weights and produce evaluation metrics.

3. Visualise results:

   ```bash
   python src/visualize.py --result_dir results/gru_xy
   ```

   This produces side-by-side reconstructions, error curves and possibly animations.

4. Repeat for other architectures / observation modes (e.g., MLP, LSTM; obs = x, obs = x2, etc.) and compare.

### Folder Organisation & Overwriting

The orchestrator is designed to **overwrite** existing outputs if rerun (for reproducibility and avoiding stale artefacts). For example, running the same experiment variant will overwrite `results/model_obsVariant/` with fresh output.
Ensure you point to `Documents/DA/lorenz63_project_final/` (or your preferred folder) for final artefacts.

##  Project Goals & Scope

* Provide a **structured and reproducible** experiment pipeline for data assimilation using neural networks.
* Compare architectures (MLP, GRU, LSTM) under **identical experimental conditions** (same data, same loss functions, same observation operators).
* Emphasise **visual diagnostics**: e.g., failure cases, reconstructions (truth vs background vs analysed state), error plots through time.
* Avoid **data leakage**: careful train/validation/test split, no sharing of information across splits.
* Serve as a robust base for a potential master’s thesis or for collaboration (e.g., with the Deutscher Wetterdienst).





