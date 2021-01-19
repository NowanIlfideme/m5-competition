from m5 import __data__

import xarray as xr

path_ds = __data__ / "ds.nc"

__all__ = ["load_ds", "path_ds"]


def load_ds(force_recalc: bool = False, cache: bool = True) -> xr.Dataset:
    if force_recalc or path_ds.exists():
        ds = xr.load_dataset(path_ds)
    else:
        from .raw import calendar, sell_prices, sales_train_eval
        from .prep import get_ds_full

        ds = get_ds_full(
            calendar=calendar, sales=sales_train_eval, sell_prices=sell_prices
        )
        if cache:
            ds.to_netcdf(path_ds)
    return ds
