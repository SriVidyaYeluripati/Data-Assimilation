# src/utils/config.py
import os
from datetime import datetime

# ---- Root paths (edit BASE_DIR if you move the project) ----
BASE_DIR = r"C:\Users\VidyaYeluripati\Documents\DA\Newfolder"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Your layout: raw, obs, splits are siblings under data/
RAW_DIR    = os.path.join(DATA_DIR, "raw")
OBS_DIR    = os.path.join(DATA_DIR, "obs")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# Results
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
RESAMPLE_ROOT   = os.path.join(RESULTS_DIR, "resample")
FIXEDMEAN_ROOT  = os.path.join(RESULTS_DIR, "fixedmean")

# Ensure directories exist
for d in [RAW_DIR, OBS_DIR, SPLITS_DIR, RESAMPLE_ROOT, FIXEDMEAN_ROOT]:
    os.makedirs(d, exist_ok=True)

# Dataset defaults
SEQ_LEN = 5
DT      = 0.01
STEPS   = 200
N_TRAJ  = 1500

def make_run_dirs(branch: str):
    """Create a new run folder for a branch and return (run_dir, fig_dir, met_dir, mod_dir)."""
    if branch not in {"resample", "fixedmean"}:
        raise ValueError("branch must be 'resample' or 'fixedmean'")
    root = RESAMPLE_ROOT if branch == "resample" else FIXEDMEAN_ROOT
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, run_tag)
    fig_dir = os.path.join(run_dir, "figures")
    met_dir = os.path.join(run_dir, "metrics")
    mod_dir = os.path.join(run_dir, "models")
    for d in (fig_dir, met_dir, mod_dir):
        os.makedirs(d, exist_ok=True)
    return run_dir, fig_dir, met_dir, mod_dir

def latest_run_dir(branch: str) -> str:
    """Return the full path to the most recent run under a branch (no creation)."""
    root = RESAMPLE_ROOT if branch == "resample" else FIXEDMEAN_ROOT
    runs = [d for d in os.listdir(root) if d.startswith("run_") and os.path.isdir(os.path.join(root, d))]
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}. Train first.")
    runs.sort()
    return os.path.join(root, runs[-1])
