"""Starter factories, not tuned/fitted AEP models. TV04 owns this file."""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def make_model(name, seed=42, **overrides):
    if name == 'linear':
        return LinearRegression(**overrides)
    if name == 'rf':
        params = dict(n_estimators=100,max_depth=12,random_state=seed,n_jobs=2)
        params.update(overrides)
        return RandomForestRegressor(**params)
    if name == 'xgboost':
        params = dict(n_estimators=200,max_depth=4,learning_rate=0.05,
                      objective='reg:squarederror',tree_method='hist',
                      random_state=seed,n_jobs=2,importance_type='gain')
        params.update(overrides)
        return XGBRegressor(**params)
    raise ValueError('Unknown model name; choose linear, rf, xgboost.')
