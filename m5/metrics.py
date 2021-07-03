"""Metrics calculation for the M5 competition.

Currently only point accuracy metrics are defined:
https://www.kaggle.com/c/m5-forecasting-accuracy/overview/evaluation

There are also quantile metrics for the Uncertainty competition, but
those are not currently implemented:
https://www.kaggle.com/c/m5-forecasting-uncertainty/overview/evaluation
"""

import warnings
import xarray as xr
import numpy as np


def mse_naive(
    pds: xr.Dataset, target: str = "sales", dim_date: str = "date"
) -> xr.Dataset:
    """Calculates mean squared error, per-series, for a 'naive' forecast.

    This is used as the denominator for RMSSE calculations.
    """
    naive = pds[target].shift(date=1).fillna(0)

    # Whether the product was sold at this time,
    # make sure we count from when we have naive "predictions"
    sale_ind = (naive > 0).astype(int)

    # 1 if active (starting from idx_first_nz), 0 if first sale or before
    is_active = sale_ind.where(sale_ind).ffill(dim=dim_date).fillna(0).astype(int)

    n_obs = is_active.sum(dim=dim_date)

    err = pds[target] - naive
    result = 1 / (n_obs - 1) * (err * err * is_active).sum(dim=dim_date)
    return result


def mse_pred(
    pds: xr.Dataset,
    target: str = "sales",
    t_hat: str = "sales_hat",
    dim_date: str = "date",
) -> xr.Dataset:
    """This calculates the MSE of the prediction, per-series.

    We assume the entire period is
    """
    err = pds[target] - pds[t_hat]
    h = len(pds[dim_date])
    result = 1 / h * (err * err).sum(dim=dim_date)
    return result


def get_weights(
    pds: xr.Dataset,
    last: int = 28,
    target: str = "sales",
    price: str = "price",
    dim_date: str = "date",
) -> xr.Dataset:
    """Gets the weights, per-series.

    This is based on the value (price * qty) during the last `last` days of sales.
    The value is divided by the total value of all series.
    """
    x = pds.isel({dim_date: slice(-last, None)})
    raw = (x[target] * x[price]).sum(dim=dim_date)
    return raw / raw.sum()


def get_rmsse(
    train: xr.Dataset,
    valid: xr.Dataset,
    target: str = "sales",
    t_hat: str = "sales_hat",
    dim_date: str = "date",
) -> xr.Dataset:
    """Calculaes RMSSE metric per series. It may be nan, if a series had target==0."""
    denominator = mse_naive(train, target=target, dim_date=dim_date)
    numerator = mse_pred(valid, target=target, t_hat=t_hat, dim_date=dim_date)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rmsse: xr.DataArray = np.sqrt(numerator / denominator)
    rmsse = rmsse.where(np.isfinite(rmsse))
    return rmsse


def get_wrmsse(
    train: xr.Dataset,
    valid: xr.Dataset,
    target: str = "sales",
    t_hat: str = "sales_hat",
    price: str = "price",
    dim_date: str = "date",
    last: int = 28,
) -> float:
    """Calculates Weighted RMSSE metric (a float) over all series."""

    rmsse = get_rmsse(
        train=train, valid=valid, target=target, t_hat=t_hat, dim_date=dim_date
    )
    weights = get_weights(
        train, last=last, target=target, price=price, dim_date=dim_date
    )
    wrmsse = (weights * rmsse).sum().item()

    return wrmsse
