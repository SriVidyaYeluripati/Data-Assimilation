# src/utils/loss.py
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
from utils.config import RAW_DIR, OBS_DIR, SEQ_LEN, DT, make_run_dirs
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ObsMode = Literal["x", "xy", "x2"]


class VarLoss(nn.Module):
    """
    3D-Var style loss (no truth-anchoring term).

    J(x_a) = 0.5 * [(x_a - x_b)^T B^{-1} (x_a - x_b)
                    + (h(x_a) - y)^T R^{-1} (h(x_a) - y)]

    Args:
        B_inv: (3x3) background inverse covariance (np.ndarray or torch.Tensor)
        R_inv: (obs_dim x obs_dim) observation inverse covariance (np.ndarray or torch.Tensor)
        obs_mode: "x", "xy", or "x2" (defines h(x))
    """
    def __init__(self,
                 B_inv: np.ndarray | torch.Tensor,
                 R_inv: np.ndarray | torch.Tensor,
                 obs_mode: ObsMode = "x"):
        super().__init__()
        # store as buffers so they move with .to(device)
        B_inv_t = torch.as_tensor(B_inv, dtype=torch.float32)
        R_inv_t = torch.as_tensor(R_inv, dtype=torch.float32)
        self.register_buffer("B_inv", B_inv_t)
        self.register_buffer("R_inv", R_inv_t)
        if obs_mode not in ("x", "xy", "x2"):
            raise ValueError(f"Unknown obs_mode: {obs_mode}")
        self.mode: ObsMode = obs_mode

    def h(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3]
        if self.mode == "x":
            return x[:, [0]]                 # [B, 1]
        elif self.mode == "xy":
            return x[:, [0, 1]]              # [B, 2]
        elif self.mode == "x2":
            return (x[:, [0]] ** 2)          # [B, 1]
        raise RuntimeError("Invalid obs_mode")  # should never hit

    def forward(self,
                x_a: torch.Tensor,
                x_b: torch.Tensor,
                y_seq: torch.Tensor,
                y_last: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_a:   [B, 3] analysis state (model output)
            x_b:   [B, 3] background state
            y_seq: [B, L, obs_dim] observation window (we use last obs)
            y_last: [B, obs_dim] (optional) if provided, overrides y_seq[:, -1]

        Returns:
            scalar loss tensor
        """
        B = x_a.shape[0]

        # ---- Background mismatch ----
        dx = (x_a - x_b).unsqueeze(1)                     # [B, 1, 3]
        B_inv = self.B_inv.unsqueeze(0).expand(B, -1, -1) # [B, 3, 3]
        term_b = torch.bmm(torch.bmm(dx, B_inv), dx.transpose(1, 2)).squeeze(-1).squeeze(-1)
        # shape -> [B]

        # ---- Observation mismatch (use last obs) ----
        y_t = y_last if y_last is not None else y_seq[:, -1]  # [B, obs_dim]
        y_pred = self.h(x_a)                                  # [B, obs_dim]
        dy = (y_pred - y_t).unsqueeze(1)                      # [B, 1, obs_dim]
        R_inv = self.R_inv.unsqueeze(0).expand(B, -1, -1)     # [B, obs_dim, obs_dim]
        term_r = torch.bmm(torch.bmm(dy, R_inv), dy.transpose(1, 2)).squeeze(-1).squeeze(-1)
        # shape -> [B]

        loss = 0.5 * (term_b + term_r).mean()
        return loss


__all__ = ["VarLoss"]
