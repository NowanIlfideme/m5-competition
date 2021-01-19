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


def get_cal_indices(calendar: pd.DataFrame) -> xr.Dataset:
    """Dataset of indicies for workday, month, year"""
    cc = calendar.set_index("date")
    c1 = cc[["wday", "month", "year"]].to_xarray()
    return c1


def get_cal_events(calendar: pd.DataFrame) -> xr.Dataset:
    """Dataset of calendar events."""

    cc = calendar.set_index("date")
    s_names = pd.concat([cc["event_name_1"], cc["event_name_2"]])
    s_types = pd.concat([cc["event_type_1"], cc["event_type_2"]])
    e_names = pd.Index(s_names.dropna().unique(), name="event_name")

    event_map = (
        pd.DataFrame({"event_name": s_names, "event_type": s_types})
        .dropna()
        .drop_duplicates()
        .set_index("event_name")["event_type"]
    )

    x = pd.DataFrame(0, index=cc.index, columns=e_names)
    for enc in [c for c in cc.columns if "event_name_" in c]:
        x += pd.get_dummies(cc[enc]).reindex(columns=e_names, fill_value=0)

    c2 = xr.DataArray(x).to_dataset(name="event")
    c2.coords["event_type"] = event_map
    return c2


def get_snap(calendar: pd.DataFrame, sales: pd.DataFrame) -> xr.Dataset:
    """Gets dataframe of SNAP sales."""

    cc = calendar.set_index("date")
    cols_snap = {c: c.strip("snap_") for c in cc.columns if "snap_" in c}
    df_snap = cc[list(cols_snap)].rename(columns=cols_snap)
    df_snap.columns.name = "state_id"
    w1 = df_snap.unstack().rename("snap").reset_index()
    w2 = sales[["store_id", "state_id"]].drop_duplicates()
    c3 = (
        pd.merge(w1, w2, on=["state_id"])
        .set_index(["date", "store_id"])[["snap"]]
        .to_xarray()
    )
    return c3


def get_ds_full(
    calendar: pd.DataFrame, sales: pd.DataFrame, sell_prices: pd.DataFrame
) -> xr.Dataset:
    """Prepares full dataset."""

    ds_sales = get_ds_sales(calendar, sales)
    ds_prices = get_ds_prices(calendar, sell_prices)
    ds_coords = get_ds_coords(sales)
    c1 = get_cal_indices(calendar)
    c2 = get_cal_events(calendar)
    ds_snap = get_snap(calendar, sales)
    ds = xr.merge([ds_coords, ds_sales, ds_prices, c1, c2, ds_snap])

    # Add time coord
    is_future = ds["sales"].isnull().all(dim=["item_id", "store_id"])
    ds.coords["historic"] = ~is_future
    ds.coords["future"] = is_future

    return ds
