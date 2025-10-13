# src/evaluation.py
import os, sys, argparse, re, glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
np.seterr(all="ignore"); np.nan_to_num = lambda a, **k: __import__('numpy').nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)

# allow running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    RAW_DIR, OBS_DIR, SEQ_LEN, DT, latest_run_dir,
    RESAMPLE_ROOT, FIXEDMEAN_ROOT
)
from models import MLPModel, GRUModel, LSTMModel
from utils.lorenz import lorenz63_step

# Load background mean
B_mean = np.load(os.path.join(RAW_DIR, "B_mean.npy"))


def sequential_eval_fast(model, mode, sigma, test_traj, obs_arr,
                         steps=50, n_test=5, fp_iter=3, device="cpu"):
    model.eval(); model = model.to(device)
    rmse_b_list, rmse_a_list = [], []

    # --- inside sequential_eval_fast ---

    for i in range(n_test):
        traj = test_traj[i]
        obs = obs_arr[i]
        x_b = torch.tensor(B_mean, dtype=torch.float32).unsqueeze(0).to(device)

        pred_b, pred_a = [], []
        for t in range(SEQ_LEN-1, steps):
            y_seq = torch.tensor(obs[t-SEQ_LEN+1:t+1], dtype=torch.float32).unsqueeze(0).to(device)

            x_a = x_b.clone()
            for _ in range(fp_iter):
                delta = model(x_a, y_seq) - x_a
                x_a = x_a + delta

            pred_b.append(x_b.cpu().numpy().squeeze())
            pred_a.append(x_a.detach().cpu().numpy().squeeze())

            # Propagate forward
            x_next = lorenz63_step(x_a.detach().cpu().numpy().squeeze(), dt=DT)
            x_b = torch.tensor(x_next, dtype=torch.float32).unsqueeze(0).to(device)

        # --- NEW: allow exploding / NaN trajectories, but record them ---
        try:
            rmse_b = np.sqrt(mean_squared_error(traj[SEQ_LEN-1:steps], np.array(pred_b)))
            rmse_a = np.sqrt(mean_squared_error(traj[SEQ_LEN-1:steps], np.array(pred_a)))
        except ValueError:
            import warnings
            warnings.warn(f"[Explode] NaN/Inf in predictions for {mode}, σ={sigma}, traj {i}")
            rmse_b, rmse_a = np.nan, np.nan   # record as NaN, don’t skip

        rmse_b_list.append(rmse_b)
        rmse_a_list.append(rmse_a)

    return float(np.nanmean(rmse_b_list)), float(np.nanmean(rmse_a_list))



def load_model_from_file(path, model_name, obs_dim, device="cpu"):
    if model_name == "mlp":
        model = MLPModel(x_dim=3, y_dim=obs_dim, hidden=64)
    elif model_name == "gru":
        model = GRUModel(x_dim=3, y_dim=obs_dim, hidden=64)
    elif model_name == "lstm":
        model = LSTMModel(x_dim=3, y_dim=obs_dim, hidden=64)
    else:
        raise ValueError(f"Unknown model {model_name}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["resample","fixedmean"], default="resample")
    ap.add_argument("--run-dir", type=str, default=None, help="Full path to a run directory")
    ap.add_argument("--latest", action="store_true", help="Use the most recent run under the branch")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=5)
    args = ap.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = latest_run_dir(args.branch)

    mod_dir = os.path.join(run_dir, "models")
    met_dir = os.path.join(run_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    print("Evaluating models in:", mod_dir)

    test_traj = np.load(os.path.join(RAW_DIR, "test_traj.npy"))
    obs_data = {}
    for mode in ["x","xy","x2"]:
        for s in [0.05,0.1,0.5,1.0]:
            obs_data[(mode, s)] = np.load(os.path.join(OBS_DIR, f"obs_{mode}_n{s}.npy"))

    model_files = glob.glob(os.path.join(mod_dir, "*.pth"))
    print(f"Found {len(model_files)} models to evaluate.")
    results = []

    for mfile in model_files:
        fname = os.path.basename(mfile)
        m = re.match(r"([a-z0-9]+)_([a-z]+)_n([0-9.]+)_R([0-9.]+)\.pth", fname)
        if not m: 
            continue
        mode, model_name, sigma_str, _ = m.groups()
        sigma = float(sigma_str)
        obs_dim = obs_data[(mode, sigma)].shape[-1]

        model = load_model_from_file(mfile, model_name, obs_dim, device=args.device)
        rmse_b, rmse_a = sequential_eval_fast(
            model, mode, sigma, test_traj, obs_data[(mode, sigma)],
            steps=args.steps, n_test=args.n_test, device=args.device
        )
        results.append({
            "mode": mode, "model": model_name, "sigma": sigma,
            "mean_rmse_b": rmse_b, "mean_rmse_a": rmse_a,
            "improvement_pct": (rmse_b - rmse_a) / rmse_b * 100 if rmse_b > 0 else np.nan
        })
        print(f"[FAST] {mode}_{model_name}_σ{sigma} -> Bg={rmse_b:.3f}, Ana={rmse_a:.3f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(met_dir, "fast_eval_results.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
