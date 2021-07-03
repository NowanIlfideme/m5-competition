"""Metrics calculation for the M5 competition.

Currently only point accuracy metrics are defined:
https://www.kaggle.com/c/m5-forecasting-accuracy/overview/evaluation

There are also quantile metrics for the Uncertainty competition, but
those are not currently implemented:
https://www.kaggle.com/c/m5-forecasting-uncertainty/overview/evaluation
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr


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


def get_weights(
    pds: xr.Dataset,
    last: int = 28,
    target: str = "sales",
    price: str = "price",
    dim_date: str = "date",
) -> xr.DataArray:
    """Gets the weights, per-series.

    This is based on the value (price * qty) during the last `last` days of sales.
    The value is divided by the total value of all series.

    FIXME: These aren't the full weights!
    It turns out, we also need the weights of aggregate series!
    """
    x = pds.isel({dim_date: slice(-last, None)})
    raw = (x[target] * x[price]).sum(dim=dim_date)
    return raw / raw.sum()


def get_wrmsse(
    train: xr.Dataset,
    valid: xr.Dataset,
    target: str = "sales",
    t_hat: str = "sales_hat",
    price: str = "price",
    dim_date: str = "date",
    last: int = 28,
) -> float:
    """Calculates Weighted RMSSE metric (a float) over all series.

    FIXME: These aren't using the full weights!
    It turns out, we also need the weights of aggregate series!
    """

    rmsse = get_rmsse(
        train=train, valid=valid, target=target, t_hat=t_hat, dim_date=dim_date
    )
    weights = get_weights(
        train, last=last, target=target, price=price, dim_date=dim_date
    )
    wrmsse = (weights * rmsse).sum().item()
    return wrmsse


DEFAULT_LEVELS = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "cat": ["cat_id"],
    "dept": ["dept_id"],
    "state-cat": ["state_id", "cat_id"],
    "state-dept": ["state_id", "dept_id"],
    "store-cat": ["store_id", "cat_id"],
    "store-dept": ["store_id", "dept_id"],
    "product": ["item_id"],
    "product-state": ["item_id", "state_id"],
    # "product-store": ["item_id", "store_id"],  # we don't need to groupby
    "product-store": None,
}


def wrmsse_per_level(
    train: xr.Dataset,
    valid: xr.Dataset,
    levels: Dict[str, Optional[List[str]]] = None,
    target: str = "sales",
    t_hat: str = "sales_hat",
    price: str = "price",
    dim_date: str = "date",
    last: int = 28,
) -> pd.Series:
    """Gets Weighted RMSSE per each level."""

    if levels is None:
        levels = DEFAULT_LEVELS
    weights = get_weights(
        train, last=last, target=target, price=price, dim_date=dim_date
    )
    errors_per_level = pd.Series(dtype=float)

    def _d(x: xr.Dataset) -> List[str]:
        return list(set(x.dims).difference(["date"]))

    for lname, ldef in levels.items():
        gb = xr.DataArray("")
        for l in ldef:
            gb = gb + train[l] + "|"
        if ldef is None:
            train_group, valid_group, weights_group = train, valid, weights
        elif len(ldef) == 0:
            train_group = train[[target]].sum(_d(train[[target]]))
            weights_group = weights.sum(_d(weights))
            valid_group = valid[[target, t_hat]].sum(_d(valid[[target]]))
        else:
            train_group = train[[target]].groupby(gb).sum()
            weights_group = weights.groupby(gb).sum()
            valid_group = valid[[target, t_hat]].groupby(gb).sum()
        rmsse_group = get_rmsse(
            train=train_group,
            valid=valid_group,
            target=target,
            t_hat=t_hat,
            dim_date=dim_date,
        )
        err_level = (weights_group * rmsse_group).sum().item()
        errors_per_level[lname] = err_level
    return errors_per_level


def wrmsse_total(
    train: xr.Dataset,
    valid: xr.Dataset,
    levels: Dict[str, Optional[List[str]]] = None,
    target: str = "sales",
    t_hat: str = "sales_hat",
    price: str = "price",
    dim_date: str = "date",
    last: int = 28,
) -> float:
    """Gets the average error over all levels."""

    epl = wrmsse_per_level(
        train=train,
        valid=valid,
        levels=levels,
        target=target,
        t_hat=t_hat,
        price=price,
        dim_date=dim_date,
        last=last,
    )
    return float(epl.mean())
