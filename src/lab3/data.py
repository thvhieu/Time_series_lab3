"""Nominal-hour AEP loader; preserve missing targets and disclose duplicate policy."""
from pathlib import Path
import numpy as np
import pandas as pd

def load_hourly(path):
    raw = pd.read_csv(Path(path))
    if not {'Datetime', 'AEP_MW'}.issubset(raw.columns) or raw.empty:
        raise ValueError('Expected nonempty Datetime,AEP_MW CSV.')
    dates = pd.to_datetime(raw['Datetime'], errors='raise')
    if dates.isna().any() or not dates.eq(dates.dt.floor('h')).all():
        raise ValueError('Timestamp must be a valid whole nominal hour.')
    values = pd.to_numeric(raw['AEP_MW'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Observed AEP values must be finite and positive.')
    y = pd.Series(values, index=pd.DatetimeIndex(dates), name='y')
    clean = y.groupby(level=0).mean().sort_index().asfreq('h')
    frame = clean.to_frame()
    frame['observed'] = frame['y'].notna()
    quality = {
        'raw_rows': len(raw), 'duplicate_timestamp_rows': int(dates.duplicated().sum()),
        'missing_nominal_hours': int(clean.isna().sum()), 'grid_rows': len(clean),
        'start': str(clean.index[0]), 'end': str(clean.index[-1]),
        'duplicate_policy': 'mean_at_same_nominal_timestamp',
        'timezone_policy': 'naive_local_nominal_hours_not_UTC',
    }
    return frame, quality

def holdout_start(index, train_ratio=0.8):
    if not 0 < train_ratio < 1 or len(index) < 2:
        raise ValueError('Need a nonempty train and test.')
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError('Index must be sorted and unique.')
    cut = int(len(index)*train_ratio)
    if cut == 0 or cut == len(index):
        raise ValueError('Need a nonempty train and test.')
    return index[cut]

def split_supervised(supervised, test_start):
    if not supervised.index.is_monotonic_increasing or not supervised.index.is_unique:
        raise ValueError('Index must be sorted and unique.')
    cutoff = pd.Timestamp(test_start)
    train = supervised.loc[supervised.index < cutoff].copy()
    test = supervised.loc[supervised.index >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError('Need a nonempty train and test.')
    return train, test


def load_prepared(directory, raw_path=None):
    """Read and verify shared Data v1; never impute missing evaluation targets."""
    import hashlib
    import json
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Prepared v1 missing; run python scripts/prepare_data.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != 'v1':
        raise ValueError('Expected prepared data version v1.')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Prepared artifact name must be a basename.')
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared artifact hash mismatch: '+name)
    if raw_path is not None:
        actual = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
        if actual != manifest['raw_sha256']:
            raise ValueError('Raw source differs from prepared data source.')
    raw = pd.read_csv(directory/'AEP_hourly_clean.csv')
    if list(raw.columns) != ['Datetime', 'y', 'observed']:
        raise ValueError('Unexpected prepared columns.')
    dates = pd.DatetimeIndex(pd.to_datetime(raw['Datetime'], errors='raise'), name='Datetime')
    if len(dates) == 0 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Prepared timestamps must be nonempty, sorted and unique.')
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq='h')):
        raise ValueError('Prepared data must retain the complete nominal hourly grid.')
    observed = raw['observed'].astype(str).str.lower().map({'true': True, 'false': False})
    if observed.isna().any():
        raise ValueError('Observed flag must be True or False.')
    values = pd.to_numeric(raw['y'], errors='raise').to_numpy(dtype=float)
    mask = observed.to_numpy(dtype=bool)
    if not np.array_equal(np.isfinite(values), mask) or (values[mask] <= 0).any():
        raise ValueError('Observed targets must be positive/finite; missing targets remain NaN.')
    dates = pd.DatetimeIndex(dates, freq='h', name='Datetime')
    frame = pd.DataFrame({'y': values, 'observed': mask}, index=dates)
    quality = json.loads((directory/'data_quality.json').read_text(encoding='utf-8'))
    if quality['grid_rows'] != len(frame) or quality['missing_nominal_hours'] != int((~mask).sum()):
        raise ValueError('Quality metadata differs from prepared data.')
    if quality['raw_sha256'] != manifest['raw_sha256']:
        raise ValueError('Source hashes disagree.')
    return frame, quality
