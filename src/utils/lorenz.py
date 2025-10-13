# utils/lorenz.py
# src/utils/lorenz.py
import numpy as np

def lorenz63_step(x, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    """One step of Lorenz-63 with overflow/NaN protection."""
    x = np.asarray(x, dtype=np.float64)
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]

    step = np.array([dx, dy, dz], dtype=np.float64) * dt
    new_x = x + step

    # --- Safety: replace NaN/Inf with large finite clip
    new_x = np.nan_to_num(new_x, nan=0.0, posinf=1e6, neginf=-1e6)
    new_x = np.clip(new_x, -1e6, 1e6)
    return new_x



def lorenz63_step_safe(x, sigma=10.0, rho=28.0, beta=8/3, dt=0.01, clip=50):
    """
    One-step Lorenz-63 update with clipping safeguard
    (avoids runaway trajectories during chaotic propagation).
    """
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    x_next = x + dt * np.array([dx, dy, dz])
    return np.clip(x_next, -clip, clip)


# ============================================================
# Trajectory Simulator
# ============================================================
def simulate_lorenz63(init, steps, dt=0.01):
    """
    Generate Lorenz-63 trajectory from initial condition.
    Returns: array of shape [steps, 3]
    """
    traj = np.zeros((steps, 3))
    x = init.copy()
    for i in range(steps):
        x = lorenz63_step(x, dt=dt)
        traj[i] = x
    return traj


def generate_unseen_trajs(K=3, steps=200, dt=0.01, seed=4321, spread=5.0, safe=False):
    """
    Generate K unseen trajectories for evaluation.
    spread: controls random IC deviation
    safe: if True, uses lorenz63_step_safe for propagation
    """
    rng = np.random.default_rng(seed)
    warm = simulate_lorenz63(np.array([1.,1.,1.]), 2000, dt)[-1]
    new_trajs = []
    for _ in range(K):
        ic = warm + rng.normal(0, spread, size=3)
        if safe:
            traj = [ic]
            x = ic.copy()
            for i in range(steps-1):
                x = lorenz63_step_safe(x, dt=dt)
                traj.append(x)
            traj = np.array(traj)
        else:
            traj = simulate_lorenz63(ic, steps, dt)
        new_trajs.append(traj)
    return np.stack(new_trajs, axis=0)
