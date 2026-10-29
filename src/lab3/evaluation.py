"""Metrics require the exact same ordered timestamp index on actual/prediction."""
import numpy as np
import pandas as pd

def regression_metrics(actual, predicted):
    if not isinstance(actual,pd.Series) or not isinstance(predicted,pd.Series):
        raise TypeError('Use timestamp-indexed pandas Series.')
    if actual.empty or not actual.index.equals(predicted.index):
        raise ValueError('Actual and prediction must have equal nonempty indices.')
    if not actual.index.is_unique or not actual.index.is_monotonic_increasing:
        raise ValueError('Timestamps must be sorted and unique.')
    a, p = actual.to_numpy(dtype=float), predicted.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require finite values.')
    err = a-p
    mask = a != 0
    return {'n_test':len(a), 'MAE':float(np.abs(err).mean()),
            'RMSE':float(np.sqrt(np.mean(err**2))),
            'MAPE':float(np.mean(np.abs(err[mask]/a[mask]))*100) if mask.any() else None,
            'mape_valid_n':int(mask.sum())}
