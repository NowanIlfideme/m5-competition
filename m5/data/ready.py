from m5 import __data__
from .raw import calendar, sell_prices, sales_train_eval
from .prep import get_ds_full

import xarray as xr

path_ds = __data__ / "ds.nc"

__all__ = ["ds", "path_ds"]

if path_ds.exists():
    ds = xr.load_dataset(path_ds)
else:
    ds = get_ds_full(
        calendar=calendar, sales=sales_train_eval, sell_prices=sell_prices
    )
    ds.to_netcdf(path_ds)
