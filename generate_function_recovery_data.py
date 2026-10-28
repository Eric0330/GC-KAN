"""Generate the GC-KAN function recovery simulation data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# System designs and simulated datasets.
N_SYSTEMS = 10                 # Number of evaluation system designs.
SEEDS_PER_SYSTEM = 2           # Independent datasets generated for each design.
TUNING_SYSTEM_IDS = (1000, 1001, 1002)  # Separate designs used for model tuning.
N_SRC = 5                      # Number of autoregressive driver series.
N_TGT = 5                      # Number of response series.
N = N_SRC + N_TGT              # Total number of series in each system.
P = 3                         # Number of lags included in each model input.
PARENTS_PER_TARGET = 3        # Number of driver parents for each response series.

# Numbers of time points retained after the initial burn-in period.
T_TRAIN = 1000
T_VAL = 400
T_TEST = 400

# Coefficients and noise scale in the time-series equations.
PHI = 0.6                     # Driver autoregression coefficient.
A_SELF = 0.4                   # Response coefficient on its own previous value.
C_COUPLING = 0.8               # Multiplier on each driver response function.
EPS_STD = 0.5                  # Response noise standard deviation (variance 0.25).

# Centering and scaling for the cosine function with a standard normal input.
COS_MEAN = float(np.exp(-np.pi ** 2 / 2))
COS_SD = float(np.sqrt(0.5 + 0.5 * np.exp(-2 * np.pi ** 2) - np.exp(-np.pi ** 2)))
TANH_SCALE = 0.6255            # Fixed divisor for the tanh response function.

# Possible response functions assigned to driver-to-target edges.
FORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda u: u,
    "quadratic": lambda u: (u ** 2 - 1.0) / np.sqrt(2.0),
    "tanh": lambda u: np.tanh(2.0 * u) / TANH_SCALE,
    "cosine": lambda u: (np.cos(np.pi * u) - COS_MEAN) / COS_SD,
}
FORM_NAMES = list(FORMS)


def make_system(system_id: int) -> List[Tuple[int, int, str]]:
    """Draw a parent graph and balanced form assignment deterministically."""
    rng = np.random.default_rng(10_000 + system_id)
    n_edges = N_TGT * PARENTS_PER_TARGET
    base = (FORM_NAMES * ((n_edges // len(FORM_NAMES)) + 1))[:n_edges]
    forms = list(rng.permutation(base))
    design: List[Tuple[int, int, str]] = []
    k = 0
    for target_offset in range(N_TGT):
        parents = rng.choice(N_SRC, size=PARENTS_PER_TARGET, replace=False)
        for source in parents:
            design.append((N_SRC + target_offset, int(source), forms[k]))
            k += 1
    return design


def simulate(design: Iterable[Tuple[int, int, str]], noise_seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    burn = 300  # Initial time points discarded before the data splits.
    total = length + burn
    sources = np.zeros((total, N_SRC), dtype=float)
    innovation_sd = np.sqrt(1.0 - PHI ** 2)
    for t in range(1, total):
        sources[t] = PHI * sources[t - 1] + rng.normal(0, innovation_sd, N_SRC)
    targets = np.zeros((total, N_TGT), dtype=float)
    eps = rng.normal(0, EPS_STD, (total, N_TGT))
    design_list = list(design)
    for t in range(1, total):
        for target_offset in range(N_TGT):
            target = N_SRC + target_offset
            drive = sum(C_COUPLING * FORMS[form](sources[t - 1, source])
                        for candidate, source, form in design_list if candidate == target)
            targets[t, target_offset] = A_SELF * targets[t - 1, target_offset] + drive + eps[t, target_offset]
    return np.concatenate([sources, targets], axis=1)[burn:]


def lag_matrix(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = len(z)
    columns = [z[P - lag:total - lag, variable]
               for variable in range(N) for lag in range(1, P + 1)]
    return np.stack(columns, axis=1).astype(np.float32), z[P:].astype(np.float32)


def column_index(variable: int, lag: int) -> int:
    return variable * P + (lag - 1)


def build_split(system_id: int, replicate_seed: int):
    design = make_system(system_id)
    noise_seed = 777 * system_id + replicate_seed + 1  # Reproducible dataset seed.
    raw = simulate(design, noise_seed, T_TRAIN + T_VAL + T_TEST)
    mean = raw[:T_TRAIN].mean(axis=0)
    std = raw[:T_TRAIN].std(axis=0)
    z = (raw - mean) / std
    x_lag, y_lag = lag_matrix(z)
    n_train = T_TRAIN - P
    x_train = x_lag[:n_train]
    x_val = x_lag[n_train:n_train + T_VAL]
    x_test = x_lag[n_train + T_VAL:]
    y_train = y_lag[:n_train]
    y_val = y_lag[n_train:n_train + T_VAL]
    y_test = y_lag[n_train + T_VAL:]
    return design, mean, std, x_train, x_val, x_test, y_train, y_val, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-id", type=int, choices=(*range(N_SYSTEMS), *TUNING_SYSTEM_IDS),
        required=True, help="evaluation design (0-9) or tuning design (1000-1002)"
    )
    parser.add_argument(
        "--replicate-seed", type=int, choices=range(SEEDS_PER_SYSTEM),
        required=True, help="dataset replicate (0 or 1)"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .npz file")
    args = parser.parse_args()

    design, mean, std, x_train, x_val, x_test, y_train, y_val, y_test = build_split(
        args.system_id, args.replicate_seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        design=np.asarray(design, dtype=str),
        train_mean=mean,
        train_std=std,
        x_train=x_train,
        x_validation=x_val,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_val,
        y_test=y_test,
        system_id=np.asarray(args.system_id),
        replicate_seed=np.asarray(args.replicate_seed),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
