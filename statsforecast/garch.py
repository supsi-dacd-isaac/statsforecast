# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/garch.ipynb.

# %% auto 0
__all__ = ['NOGIL', 'CACHE']

# %% ../nbs/garch.ipynb 3
import os
import numpy as np
from numba import njit
from scipy.optimize import minimize

# %% ../nbs/garch.ipynb 4
NOGIL = os.environ.get("NUMBA_RELEASE_GIL", "False").lower() in ["true"]
CACHE = os.environ.get("NUMBA_CACHE", "False").lower() in ["true"]

# %% ../nbs/garch.ipynb 7
@njit(nogil=NOGIL, cache=CACHE)
def generate_garch_data(n, w, alpha, beta):

    np.random.seed(1)

    y = np.zeros(n)
    sigma2 = np.zeros(n)

    p = len(alpha)
    q = len(beta)

    w_vals = w < 0
    alpha_vals = np.any(alpha < 0)
    beta_vals = np.any(beta < 0)

    if np.any(np.array([w_vals, alpha_vals, beta_vals])):
        raise ValueError("Coefficients must be nonnegative")

    if np.sum(alpha) + np.sum(beta) >= 1:
        raise ValueError(
            "Sum of coefficients of lagged versions of the series and lagged versions of volatility must be less than 1"
        )

    # initialization
    if q != 0:
        sigma2[0:q] = 1

    for k in range(p):
        y[k] = np.random.normal(loc=0, scale=1)

    for k in range(max(p, q), n):
        psum = np.flip(alpha) * (y[k - p : k] ** 2)
        psum = np.nansum(psum)
        if q != 0:
            qsum = np.flip(beta) * (sigma2[k - q : k])
            qsum = np.nansum(qsum)
            sigma2[k] = w + psum + qsum
        else:
            sigma2[k] = w + psum
        y[k] = np.random.normal(loc=0, scale=np.sqrt(sigma2[k]))

    return y

# %% ../nbs/garch.ipynb 12
@njit(nogil=NOGIL, cache=CACHE)
def garch_sigma2(x0, x, p, q):

    w = x0[0]
    alpha = x0[1 : (p + 1)]
    beta = x0[(p + 1) :]

    sigma2 = np.full((len(x),), np.nan)
    sigma2[0] = np.var(x)  # sigma2 can be initialized with the unconditional variance

    for k in range(max(p, q), len(x)):
        psum = np.flip(alpha) * (x[k - p : k] ** 2)
        psum = np.nansum(psum)
        if q != 0:
            qsum = np.flip(beta) * (sigma2[k - q : k])
            qsum = np.nansum(qsum)
            sigma2[k] = w + psum + qsum
        else:
            sigma2[k] = w + psum

    return sigma2

# %% ../nbs/garch.ipynb 14
@njit(nogil=NOGIL, cache=CACHE)
def garch_cons(x0):
    # Constraints for GARCH model
    # alpha+beta < 1
    return 1 - (x0[1:].sum())

# %% ../nbs/garch.ipynb 16
@njit(nogil=NOGIL, cache=CACHE)
def garch_loglik(x0, x, p, q):

    sigma2 = garch_sigma2(x0, x, p, q)
    z = x - np.nanmean(x)
    loglik = 0

    for k in range(max(p, q), len(z)):
        if sigma2[k] == 0:
            sigma2[k] = 1e-10
        loglik = loglik - 0.5 * (
            np.log(2 * np.pi) + np.log(sigma2[k]) + (z[k] ** 2) / sigma2[k]
        )

    return -loglik

# %% ../nbs/garch.ipynb 18
def garch_model(x, p, q):

    np.random.seed(1)
    x0 = np.repeat(0.1, p + q + 1)
    bnds = ((0, None),) * len(x0)
    cons = {"type": "ineq", "fun": garch_cons}
    opt = minimize(
        garch_loglik, x0, args=(x, p, q), method="SLSQP", bounds=bnds, constraints=cons
    )

    coeff = opt.x
    sigma2 = garch_sigma2(coeff, x, p, q)
    fitted = np.full((len(x),), np.nan)

    for k in range(p, len(x)):
        error = np.random.normal(loc=0, scale=1)
        fitted[k] = error * np.sqrt(sigma2[k])

    res = {
        "p": p,
        "q": q,
        "coeff": coeff,
        "message": opt.message,
        "y_vals": x[-p:],
        "sigma2_vals": sigma2[-q:],
        "fitted": fitted,
    }

    return res

# %% ../nbs/garch.ipynb 22
def garch_forecast(mod, h):

    np.random.seed(1)

    p = mod["p"]
    q = mod["q"]

    w = mod["coeff"][0]
    alpha = mod["coeff"][1 : (p + 1)]
    beta = mod["coeff"][(p + 1) :]

    y_vals = np.full((h + p,), np.nan)
    sigma2_vals = np.full((h + q,), np.nan)

    y_vals[0:p] = mod["y_vals"]

    if q != 0:
        sigma2_vals[0:q] = mod["sigma2_vals"]

    for k in range(0, h):
        error = np.random.normal(loc=0, scale=1)
        psum = np.flip(alpha) * (y_vals[k : p + k] ** 2)
        psum = np.nansum(psum)
        if q != 0:
            qsum = np.flip(beta) * (sigma2_vals[k : q + k])
            qsum = np.nansum(qsum)
            sigma2hat = w + psum + qsum
        else:
            sigma2hat = w + psum
        yhat = error * np.sqrt(sigma2hat)
        y_vals[p + k] = yhat
        sigma2_vals[q + k] = sigma2hat

    res = {"mean": y_vals[-h:], "sigma2": sigma2_vals[-h:], "fitted": mod["fitted"]}

    return res
