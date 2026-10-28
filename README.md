# GC-KAN Function Recovery Experiment: Data Generator

This package generates the synthetic time series used in the controlled
response-recovery study in *Granger Causality Detection with
Kolmogorov–Arnold Networks* (Physical Review Research,
doi:10.1103/y1tc-4ms4). It provides the system designs, simulated data,
and lagged training, validation, and test arrays.

## Requirements

Python 3.9 or later and NumPy. 

## Generate a dataset

Run this command from the directory containing `generate_function_recovery_data.py`:

```bash
python generate_function_recovery_data.py --system-id 0 --replicate-seed 0 --out data/system_0_seed_0.npz
```

Use system IDs `0`–`9` and replicate seeds `0` and `1` to generate the
20 evaluation datasets. System IDs `1000`–`1002` produce the separate
tuning designs. For example, to generate every evaluation dataset:

```bash
for system_id in {0..9}; do
  for replicate_seed in 0 1; do
    python generate_function_recovery_data.py \
      --system-id "$system_id" \
      --replicate-seed "$replicate_seed" \
      --out "data/system_${system_id}_seed_${replicate_seed}.npz"
  done
done
```

The graph uses random seed `10000 + system_id`; the observations use seed
`777 * system_id + replicate_seed + 1`. Thus, the same system ID always
gives the same parent graph and response-form assignment, while the
replicate seed changes the simulated observations.

## Simulation

Each system has five driver series (`s_0`–`s_4`) and five response series
(`y_5`–`y_9`). Every response has three driver parents at lag 1. Drivers
follow `s[j,t] = 0.6*s[j,t-1] + eta[j,t]`, where `eta` is Gaussian with
variance `0.64`. Responses follow

```text
y[i,t] = 0.4*y[i,t-1] + 0.8*sum(f_ij(s[j,t-1]) for j in Pa(i)) + epsilon[i,t]
```

where `epsilon` is Gaussian with variance `0.25`. The four possible
functions are:

```text
linear:     u
quadratic:  (u**2 - 1) / sqrt(2)
tanh:       tanh(2*u) / 0.6255
cosine:     (cos(pi*u) - exp(-pi**2/2)) /
            sqrt((1 + exp(-2*pi**2))/2 - exp(-pi**2))
```

The code discards 300 initial time points, then retains 1000 training,
400 validation, and 400 test time points. It centres and scales each
series using the training period only and constructs three lags of all
ten series. This leaves 997 training rows and 400 rows in each later
split. Random seeds and the graph construction are specified in
`generate_function_recovery_data.py`.

## Output format

Each `.npz` file contains:

| Name | Contents |
| --- | --- |
| `design` | 15 rows of `(target index, driver index, response form)`; indices are zero-based and stored as strings |
| `train_mean`, `train_std` | Means and standard deviations of the ten raw series in the training period |
| `x_train`, `x_validation`, `x_test` | Standardised lagged inputs with 30 columns, ordered first by series index and then by lags 1–3 |
| `y_train`, `y_validation`, `y_test` | Standardised values of all ten series at the corresponding prediction times |
| `system_id`, `replicate_seed` | Identifiers used to generate the dataset |

The package provides data generation. Model fitting, response-curve
scoring, and the paper's observational analysis require separate code.
