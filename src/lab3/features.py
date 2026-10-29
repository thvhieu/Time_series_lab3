"""One-step features at target time t use observations only through t-1."""
import pandas as pd

def make_supervised(frame, p=24, rolling_windows=(24,168), warmup=168):
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise ValueError('p must be a positive integer.')
    windows = tuple(rolling_windows)
    if any(not isinstance(w,int) or isinstance(w,bool) or w < 2 for w in windows):
        raise ValueError('Rolling windows must be integers >=2.')
    if not isinstance(warmup,int) or warmup < max((p, *windows)):
        raise ValueError('Fixed warmup must cover all lags and windows.')
    if frame.empty or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError('Need a sorted unique hourly frame.')
    expected = pd.date_range(frame.index[0], frame.index[-1], freq='h')
    if not frame.index.equals(expected):
        raise ValueError('Call the hourly loader before creating features.')
    history = frame['y'].ffill()  # causal; missing targets are NOT filled
    columns = {f'lag_{i}': history.shift(i) for i in range(1,p+1)}
    past = history.shift(1)
    for w in windows:
        columns[f'roll_mean_{w}'] = past.rolling(w, min_periods=w).mean()
        columns[f'roll_std_{w}'] = past.rolling(w, min_periods=w).std(ddof=1)
    result = pd.DataFrame(columns, index=frame.index)
    result['y'] = frame['y']
    result = result.iloc[warmup:]
    result = result.loc[frame['observed'].reindex(result.index).astype(bool)]
    return result.dropna()

def xy(supervised):
    return supervised.drop(columns=['y']).copy(), supervised['y'].copy()
