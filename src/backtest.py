import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof(hits, alpha):
    hits = list(hits)
    T = len(hits)
    M = sum(hits)
    
    if M == 0 or M == T:
        return 0.0
    
    pi = M / T
    lr = -2 * (M * np.log(alpha / pi) + (T - M) * np.log((1 - alpha) / (1 - pi)))
    return float(chi2.sf(lr, df=1))


def christoffersen(hits, alpha):
    h = np.array(hits)
    
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 1e-9
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 1e-9
    
    pi = (n01 + n11) / len(h)
    if pi <= 0 or pi >= 1:
        return 0.0
    
    lr = -2 * ((n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
               - n00 * np.log(max(1 - pi01, 1e-9)) - n01 * np.log(max(pi01, 1e-9))
               - n10 * np.log(max(1 - pi11, 1e-9)) - n11 * np.log(max(pi11, 1e-9)))
    return float(chi2.sf(lr, df=1))


def run_horizon_backtest(model, test_df, tenor_idx, conf_levels, horizon=21, n_paths=1000, use_init=False):
    hits   = {cl: [] for cl in conf_levels}
    values = test_df.values

    for i in range(len(values) - horizon):
        row_now  = values[i]
        r_future = values[i + horizon, tenor_idx] # realized rate at horizon

        if use_init:
            samples = model.generate(n_paths, horizon=horizon,
                                     dt=1/250, init_yields=row_now)
        else:
            samples = model.generate(n_paths, horizon=horizon)

        # Last step of each generated window -> horizon-end distribution
        horizon_end = samples[:, -1, tenor_idx] # shape (n_paths, )

        for cl in conf_levels:
            q = np.quantile(horizon_end, cl)
            hits[cl].append(int(r_future < q))

    return hits