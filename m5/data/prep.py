import pandas as pd
import xarray as xr


def get_ds_sales(calendar: pd.DataFrame, sales: pd.DataFrame) -> xr.Dataset:
    """Merges sales into a readable format."""

    #
    dd = [c for c in sales.columns if c in calendar["d"].values]

    z = sales[["item_id", "store_id"] + dd].set_index(["item_id", "store_id"])
    z.columns.name = "d"
    z = z.stack().rename("sales").reset_index()
    df_sales = pd.merge(z, calendar[["d", "date"]], on="d").set_index(
        ["item_id", "store_id", "date"]
    )[["sales"]]
    ds_sales = df_sales.to_xarray()
    return ds_sales


def get_ds_prices(
    calendar: pd.DataFrame, sell_prices: pd.DataFrame
) -> xr.Dataset:
    """Merges sale prices """
    df_prices = (
        pd.merge(sell_prices, calendar[["date", "wm_yr_wk"]], on="wm_yr_wk")[
            ["store_id", "item_id", "date", "sell_price"]
        ]
        .rename({"sell_price": "price"})
        .set_index(["item_id", "store_id", "date"])
    )
    ds_prices = df_prices.to_xarray()
    return ds_prices


def get_ds_coords(sales: pd.DataFrame) -> xr.Dataset:
    df_items = (
        sales[["cat_id", "dept_id", "item_id"]]
        .drop_duplicates()
        .set_index("item_id")
    )
    df_stores = (
        sales[["state_id", "store_id"]].drop_duplicates().set_index("store_id")
    )
    ds_coords = xr.merge(
        [
            df_items.to_xarray().set_coords(["cat_id", "dept_id"]),
            df_stores.to_xarray().set_coords(["state_id"]),
        ]
    )
    return ds_coords


def get_ds_full(
    calendar: pd.DataFrame, sales: pd.DataFrame, sell_prices: pd.DataFrame
) -> xr.Dataset:
    """Prepares full dataset."""

    ds_sales = get_ds_sales(calendar, sales)
    ds_prices = get_ds_prices(calendar, sell_prices)
    ds_coords = get_ds_coords(sales)
    ds = xr.merge([ds_coords, ds_sales, ds_prices])

    # Add time coord
    is_future = ds["sales"].isnull().all(dim=["item_id", "store_id"])
    ds.coords["historic"] = ~is_future
    ds.coords["future"] = is_future

    return ds
