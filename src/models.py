import torch
import torch.nn as nn
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class MLPModel(nn.Module):
    """
    Deterministic analysis update:
      input: concat([x_b, y_last])
      output: x_a = x_b + f([x_b, y_last])
    """
    def __init__(self, x_dim: int = 3, y_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x_b: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        # y_seq: [B, L, y_dim]  -> use most recent observation only
        y_last = y_seq[:, -1]  # [B, y_dim]
        inp = torch.cat([x_b, y_last], dim=-1)
        dx = self.net(inp)
        return x_b + dx


class GRUModel(nn.Module):
    """
    Sequence encoder with GRU:
      encode obs sequence -> h
      input: concat([x_b, h])
      output: x_a = x_b + g([x_b, h])
    """
    def __init__(self, x_dim: int = 3, y_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=y_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(x_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x_b: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(y_seq)  # h: [1, B, hidden]
        h = h.squeeze(0)        # [B, hidden]
        inp = torch.cat([x_b, h], dim=-1)
        dx = self.fc(inp)
        return x_b + dx


class LSTMModel(nn.Module):
    """
    Sequence encoder with LSTM:
      encode obs sequence -> h
      input: concat([x_b, h])
      output: x_a = x_b + g([x_b, h])
    """
    def __init__(self, x_dim: int = 3, y_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=y_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(x_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x_b: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(y_seq)  # h: [1, B, hidden]
        h = h.squeeze(0)              # [B, hidden]
        inp = torch.cat([x_b, h], dim=-1)
        dx = self.fc(inp)
        return x_b + dx


class BaselineMLP(nn.Module):
    """
    Baseline no-mean model:
      Direct mapping from observations to state.
      input: y_last (only the most recent observation)
      output: x_a (estimated state)
    
    Note: This model does NOT use background state (x_b) and predicts
          the state directly from observations without residual formulation.
    """
    def __init__(self, x_dim: int = 3, y_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        # y_seq: [B, L, y_dim]  -> use only the last observation
        y_last = y_seq[:, -1, :]  # [B, y_dim]
        return self.net(y_last)


__all__ = ["MLPModel", "GRUModel", "LSTMModel", "BaselineMLP"]