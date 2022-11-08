# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['AirPassengers', 'AirPassengersDF', 'generate_series']

# %% ../nbs/utils.ipynb 3
import random
from itertools import chain

import numpy as np
import pandas as pd
from numba import njit

# %% ../nbs/utils.ipynb 6
def generate_series(n_series: int,
                    freq: str = 'D',
                    min_length: int = 50,
                    max_length: int = 500,
                    n_static_features: int = 0,
                    equal_ends: bool = False,
                    seed: int = 0) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_static_features > 0`, then each serie gets static features with random values.
    If `equal_ends == True` then all series end at the same date.

    **Parameters:**<br>
    `n_series`: int, number of series for synthetic panel.<br>
    `min_length`: int, minimal length of synthetic panel's series.<br>
    `max_length`: int, minimal length of synthetic panel's series.<br>
    `n_static_features`: int, default=0, number of static exogenous variables for synthetic panel's series.<br>
    `equal_ends`: bool, if True, series finish in the same date stamp `ds`.<br>
    `freq`: str, frequency of the data, [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).<br>

    **Returns:**<br>
    `freq`: pandas.DataFrame, synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    seasonalities = {'D': 7, 'M': 12}
    season = seasonalities[freq]
    
    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    dates = pd.date_range('2000-01-01', periods=max_length, freq=freq).values
    uids = [
        np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)
    ]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]
    y = np.arange(total_length) % season + rng.rand(total_length) * 0.5
    series = pd.DataFrame(
        {
            'unique_id': chain.from_iterable(uids),
            'ds': chain.from_iterable(ds),
            'y': y,
        }
    )
    for i in range(n_static_features):
        random.seed(seed)
        static_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        series[f'static_{i}'] = np.hstack(static_values)
        series[f'static_{i}'] = series[f'static_{i}'].astype('category')
        if i == 0:
            series['y'] = series['y'] * (1 + series[f'static_{i}'].cat.codes)
    series['unique_id'] = series['unique_id'].astype('category')
    series['unique_id'] = series['unique_id'].cat.as_ordered()
    series = series.set_index('unique_id')
    return series

# %% ../nbs/utils.ipynb 10
AirPassengers = np.array([112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
                          118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
                          114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
                          162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
                          209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
                          272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
                          302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
                          315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
                          318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
                          348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
                          362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
                          342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
                          417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
                          432.])

# %% ../nbs/utils.ipynb 11
AirPassengersDF = pd.DataFrame({'unique_id': np.ones(len(AirPassengers)),
                                'ds': pd.date_range(start='1949-01-01',
                                                    periods=len(AirPassengers), freq='M'),
                                'y': AirPassengers})

# %% ../nbs/utils.ipynb 15
@njit
def _seasonal_naive(
        y: np.ndarray, # time series
        h: int, # forecasting horizon
        fitted: bool, #fitted values
        season_length: int, # season length
    ): 
    if y.size < season_length:
        return {'mean': np.full(h, np.nan, np.float32)}
    season_vals = np.empty(season_length, np.float32)
    fitted_vals = np.full(y.size, np.nan, np.float32)
    for i in range(season_length):
        s_naive = _naive(y[i::season_length], h=1, fitted=fitted)
        season_vals[i] = s_naive['mean'].item()
        if fitted:
            fitted_vals[i::season_length] = s_naive['fitted']
    out = _repeat_val_seas(season_vals=season_vals, h=h, season_length=season_length)
    fcst = {'mean': out}
    if fitted:
        fcst['fitted'] = fitted_vals
    return fcst

@njit
def _repeat_val_seas(season_vals: np.ndarray, h: int, season_length: int):
    out = np.empty(h, np.float32)
    for i in range(h):
        out[i] = season_vals[i % season_length]
    return out

@njit
def _naive(
        y: np.ndarray, # time series
        h: int, # forecasting horizon
        fitted: bool, # fitted values
    ): 
    mean = _repeat_val(val=y[-1], h=h)
    if fitted:
        fitted_vals = np.full(y.size, np.nan, np.float32)
        fitted_vals[1:] = np.roll(y, 1)[1:]
        return {'mean': mean, 'fitted': fitted_vals}
    return {'mean': mean}

@njit
def _repeat_val(val: float, h: int):
    return np.full(h, val, np.float32)
