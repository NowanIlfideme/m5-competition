with pm.Model(coords=d.coords) as m:
    for k, v in d.data_vars.items():
        pm.Data(k, v, dims=v.dims)

    # Scale per store
    pm.Normal("store_scale", mu=0, sd=2, dims=["store_id"])
    # pm.Normal("store_trend", mu=0, sd=1, dims=["store_id"])
    pm.Normal("store_snap", mu=0, sd=1, dims=["store_id"])
    pm.Normal("store_wday", mu=0, sd=1, dims=["day_of_week", "store_id"])

    # Random walk
    # pm.GaussianRandomWalk("rw", mu=, sigma=, init=)
    pm.Normal("rw_drift", mu=0, sd=0.5, dims=["store_id"])
    pm.HalfCauchy("rw_sigma", beta=0.5, dims=["store_id"])
    pm.Normal(
        "rw_innov",
        mu=m["rw_drift"],
        sigma=m["rw_sigma"],
        dims=["date", "store_id"],
    )
    pm.Deterministic(
        "rw_vals", tt.cumsum(m["rw_innov"], axis=0), dims=["date", "store_id"]
    )

    pm.HalfCauchy("sigma_store", beta=2, dims=["store_id"])
    # pm.Exponential("sigma_store", lam=2, dims=['store_id'])

    pm.Lognormal(
        "sales_store_pred",
        mu=(
            m["store_scale"][np.newaxis, :]
            # + m["store_trend"][np.newaxis, :] * m["date_f"][:, np.newaxis]
            + m["store_wday"][m["wday"] - 1, :]
            + m["store_snap"][np.newaxis, :] * m["snap"]
            + m["rw_vals"]
        ),
        sigma=m["sigma_store"],
        observed=m["sales_store"],
        dims=["date", "store_id"],
    )

gv = pm.model_to_graphviz(m)
display(gv)

with m:
    trace = pm.sample(
        draws=500,
        tune=500,
        chains=4,
        target_accept=0.8,
        return_inferencedata=True,
        # init="adapt_diag",
    )