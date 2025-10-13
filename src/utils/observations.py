import numpy as np
from utils.config import RAW_DIR, OBS_DIR, SEQ_LEN, DT, make_run_dirs

def obs_operator(x, mode="x"):
    if mode == "x": return np.array([x[0]])
    elif mode == "xy": return np.array([x[0], x[1]])
    elif mode == "x2": return np.array([x[0]**2])
    else: raise ValueError("Unknown mode")

def make_observations(trajs, mode="x", sigma_noise=0.05):
    obs = []
    for traj in trajs:
        y = np.array([obs_operator(x, mode) for x in traj])
        y_noisy = y + np.random.normal(0, sigma_noise, y.shape)
        obs.append(y_noisy)
    return np.array(obs)
