import os, sys, json, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# allow running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    RAW_DIR, OBS_DIR, SPLITS_DIR, SEQ_LEN, DT,
    make_run_dirs
)
from models import MLPModel, GRUModel, LSTMModel, BaselineMLP
from loss import VarLoss
from data.dataset import AssimilationDataset


def train_one(model, loss_fn, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu", tag="", is_baseline=False):
    """
    Training loop supporting both structured models and baseline.
    
    Args:
        is_baseline: If True, model expects only y_seq (no x_b). If False, expects (x_b, y_seq).
    """
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        model.train(); tl=0.0; tb=0
        for batch in train_loader:
            if is_baseline:
                x_true, y_seq = batch
                x_true = x_true.to(device); y_seq = y_seq.to(device)
                x_a = model(y_seq)
                # Baseline uses MSE loss directly
                loss = torch.nn.functional.mse_loss(x_a, x_true)
            else:
                x_true, y_seq, x_b = batch
                x_b = x_b.to(device); y_seq = y_seq.to(device)
                x_a = model(x_b, y_seq)
                loss = loss_fn(x_a, x_b, y_seq)
            
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * (x_true.size(0) if is_baseline else x_b.size(0))
            tb += x_true.size(0) if is_baseline else x_b.size(0)
        
        hist["train_loss"].append(tl/max(1,tb))

        model.eval(); vl=0.0; vb=0
        with torch.no_grad():
            for batch in val_loader:
                if is_baseline:
                    x_true, y_seq = batch
                    x_true = x_true.to(device); y_seq = y_seq.to(device)
                    x_a = model(y_seq)
                    loss = torch.nn.functional.mse_loss(x_a, x_true)
                else:
                    x_true, y_seq, x_b = batch
                    x_b = x_b.to(device); y_seq = y_seq.to(device)
                    x_a = model(x_b, y_seq)
                    loss = loss_fn(x_a, x_b, y_seq)
                
                vl += loss.item() * (x_true.size(0) if is_baseline else x_b.size(0))
                vb += x_true.size(0) if is_baseline else x_b.size(0)
        
        hist["val_loss"].append(vl/max(1,vb))

        if (ep+1) % 5 == 0:
            print(f"[{tag}] Epoch {ep+1}/{epochs} | train={hist['train_loss'][-1]:.5f} | val={hist['val_loss'][-1]:.5f}")

    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["resample","fixedmean","baseline"], default="resample")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir, fig_dir, met_dir, mod_dir = make_run_dirs(args.branch)
    print("Run dir:", run_dir)

    # Load arrays
    train_traj = np.load(os.path.join(RAW_DIR, "train_traj.npy"))
    B          = np.load(os.path.join(RAW_DIR, "B.npy"))
    B_mean     = np.load(os.path.join(RAW_DIR, "B_mean.npy"))

    # Build observation dict
    obs_data = {}
    modes  = ["x","xy","x2"]
    sigmas = [0.05,0.1,0.5,1.0]
    for mode in modes:
        for s in sigmas:
            obs_data[(mode,s)] = np.load(os.path.join(OBS_DIR, f"obs_{mode}_n{s}.npy"))

    # ============================================================
    # BASELINE BRANCH
    # ============================================================
    if args.branch == "baseline":
        print("\n🧪 BASELINE BRANCH: No-Mean MLP")
        
        from data.dataset import BaselineDataset  # Import baseline dataset class
        
        for mode in modes:
            for sigma in sigmas:
                obs_dim = obs_data[(mode, sigma)].shape[-1]
                
                train_ds = BaselineDataset(
                    train_traj, obs_data[(mode, sigma)],
                    seq_len=SEQ_LEN, split="train", splits_dir=SPLITS_DIR
                )
                val_ds = BaselineDataset(
                    train_traj, obs_data[(mode, sigma)],
                    seq_len=SEQ_LEN, split="val", splits_dir=SPLITS_DIR
                )
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

                tag = f"{mode}_baseline_n{sigma}"
                print(f"\n[TRAIN] BASELINE | {tag}")
                model = BaselineMLP(x_dim=3, y_dim=obs_dim, hidden=32)

                model, history = train_one(
                    model, None, train_loader, val_loader,
                    epochs=args.epochs, lr=args.lr, device=args.device, tag=tag, is_baseline=True
                )

                torch.save(model.state_dict(), os.path.join(mod_dir, f"{tag}.pth"))
                with open(os.path.join(met_dir, f"loss_{tag}.json"), "w") as f:
                    json.dump(history, f, indent=2)
    
    # ============================================================
    # STRUCTURED BRANCHES (Resample & FixedMean)
    # ============================================================
    else:
        # split handling
        if args.branch == "resample":
            bg_mode = "resample"
            reuse_dir = None
        else:
            bg_mode = "fixed"
            reuse_dir = SPLITS_DIR

        models_dict = {"mlp": MLPModel, "gru": GRUModel, "lstm": LSTMModel}

        for mode in modes:
            for sigma in sigmas:
                obs_dim = obs_data[(mode, sigma)].shape[-1]
                
                train_ds = AssimilationDataset(
                    train_traj, obs_data[(mode, sigma)], B, B_mean,
                    seq_len=SEQ_LEN, split="train", splits_dir=SPLITS_DIR,
                    background_mode=bg_mode, reuse_splits_dir=reuse_dir
                )
                val_ds = AssimilationDataset(
                    train_traj, obs_data[(mode, sigma)], B, B_mean,
                    seq_len=SEQ_LEN, split="val", splits_dir=SPLITS_DIR,
                    background_mode=bg_mode, reuse_splits_dir=reuse_dir
                )
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

                R = np.eye(obs_dim) * (sigma**2)
                loss_fn = VarLoss(B_inv=np.linalg.inv(B), R_inv=np.linalg.inv(R), obs_mode=mode)

                for name, Cls in models_dict.items():
                    tag = f"{mode}_{name}_n{sigma}_R{sigma}"
                    print(f"\n[TRAIN] {args.branch.upper()} | {tag}")
                    model = Cls(x_dim=3, y_dim=obs_dim, hidden=64)

                    model, history = train_one(
                        model, loss_fn, train_loader, val_loader,
                        epochs=args.epochs, lr=args.lr, device=args.device, tag=tag, is_baseline=False
                    )

                    torch.save(model.state_dict(), os.path.join(mod_dir, f"{tag}.pth"))
                    with open(os.path.join(met_dir, f"loss_{tag}.json"), "w") as f:
                        json.dump(history, f, indent=2)

    print("✅ Finished training:", run_dir)


if __name__ == "__main__":
    main()