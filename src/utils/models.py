"""Model builders (spec §5). Every model exposes fit_predict(train_df, val_df) -> np.ndarray."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET


def abs_pearson(X, y):
    return np.nan_to_num(np.abs(r_regression(X, y)))


def linear_pipeline(kind: str = "ridge", alpha: float = 1.0, l1_ratio: float = 0.5,
                    topk: int | None = None, log_target: bool = False):
    """StandardScaler → [top-k by |Pearson r|, fitted on the training fold only] → linear model."""
    if kind == "ridge":
        est = Ridge(alpha=alpha)
    elif kind == "lasso":
        est = Lasso(alpha=alpha, max_iter=50_000)
    elif kind == "enet":
        est = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000)
    elif kind == "ols":
        est = LinearRegression()
    else:
        raise ValueError(kind)
    steps = [("scale", StandardScaler())]
    if topk is not None:
        steps.append(("topk", SelectKBest(abs_pearson, k=topk)))
    steps.append(("model", est))
    pipe = Pipeline(steps)
    if log_target:  # fit on log(y); predictions are back-transformed to minutes before scoring
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def fit_predict_linear(features: list[str], **kw):
    def fp(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        m = linear_pipeline(**kw).fit(train[features].values, train[TARGET].values)
        return m.predict(val[features].values)
    return fp


# --------------------------------------------------------------------------- baselines (§5.2)

def fp_mean(train, val):
    return np.full(len(val), train[TARGET].mean())


def fp_persistence(train, val):
    return val["tt_now"].values


def fp_profile(train, val):
    """Mean y in the training fold by (is_workday, 15-min slot of T)."""
    prof = train.groupby(["is_workday", "slot"])[TARGET].mean()
    idx = pd.MultiIndex.from_arrays([val["is_workday"], val["slot"]])
    out = prof.reindex(idx).values
    return np.where(np.isnan(out), train[TARGET].mean(), out)


def fit_lgbm(train: pd.DataFrame, features: list[str], params: dict, cfg: dict):
    """LightGBM with n_estimators from early stopping on the last 10% (time-ordered) of `train`."""
    import lightgbm as lgb

    c = cfg["cv"]["lgbm"]
    n_es = int(round(len(train) * c["early_stopping_frac"]))
    fit, es = train.iloc[:-n_es], train.iloc[-n_es:]
    model = lgb.LGBMRegressor(
        n_estimators=c["max_estimators"], num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"], random_state=cfg["seed"],
        deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
    )
    with warnings.catch_warnings():  # lightgbm ≥ 4.7 deprecates eval_set in favour of eval_X / eval_y
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        model.fit(fit[features], fit[TARGET], eval_set=[(es[features], es[TARGET])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(c["early_stopping_rounds"], verbose=False)])
    return model


def fit_predict_lgbm(features: list[str], params: dict, cfg: dict, record: list | None = None):
    def fp(train, val):
        m = fit_lgbm(train, features, params, cfg)
        if record is not None:
            record.append(int(m.best_iteration_))
        return m.predict(val[features])
    return fp
