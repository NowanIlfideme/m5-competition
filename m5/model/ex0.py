with pm.Model(coords=d.coords) as m:
    for k, v in d.data_vars.items():
        pm.Data(k, v, dims=v.dims)

    # Scale per store
    pm.Normal("store_scale", mu=0, sd=2, dims=["store_id"])
    # pm.Normal("store_trend", mu=0, sd=1, dims=["store_id"])
    pm.Normal("store_snap", mu=0, sd=1, dims=["store_id"])
    pm.Normal("store_wday", mu=0, sd=1, dims=["day_of_week", "store_id"])

    pm.Deterministic(
        "store_traffic",
        tt.exp(
            m["store_scale"][np.newaxis, :]
            # + m["store_trend"][np.newaxis, :] * m["date_f"][:, np.newaxis]
            + m["store_wday"][m["wday"] - 1, :]
            # + m["store_snap"][np.newaxis, :] * m["snap"]
        ),
        dims=["date", "store_id"],
    )

    pm.Uniform("zero_frac", 0, 1, dims=["item_id", "store_id"])
    psi = m["zero_frac"][np.newaxis, :, :]

    # pm.Normal("c_wday", mu=1, sigma=3, dims=["day_of_week", "item_id"])
    # f_wday = m["c_wday"][m["wday"] - 1, :]  # 'date', 'store_id'

    pm.Normal("c_event", mu=0, sigma=1, dims=["event_name", "item_id"])
    f_event = m["event"] @ m["c_event"]  # 'date', 'item_id'
    #

    pm.HalfNormal("item_scale", sigma=2, dims=["item_id", "store_id"])
    f_base = (
        m["store_traffic"][:, np.newaxis, :] * m["item_scale"][np.newaxis, :, :]
    )

    theta = (
        f_base
        # f_wday[:, :, np.newaxis]
        + f_event[:, :, np.newaxis]
    )  # 'date', 'item_id', 'store_id'

    pm.ZeroInflatedPoisson(
        "y",
        psi=psi,
        theta=theta,
        dims=["date", "item_id", "store_id"],
        observed=m["sales"],
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