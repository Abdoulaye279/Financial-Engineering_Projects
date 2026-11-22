# Allocation across industry portfolios: covariance estimators, weighting schemes, out-of-sample backtest
# and factor regressions. Data: Kenneth R. French Data Library (monthly, in percent).
import numpy as np
import pandas as pd
from scipy import optimize


def load_french(path):
    # French csv files: first column YYYYMM, values in percent (or raw for size / number of firms)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.PeriodIndex(pd.to_datetime(df.index.astype(str), format="%Y%m"), freq="M")
    df.columns = df.columns.str.strip()
    return df


def ledoit_wolf_cc(X):
    # Ledoit and Wolf (2004) shrinkage towards a constant-correlation matrix.
    # X: T x N returns. returns the shrunk covariance and the shrinkage intensity in [0, 1]
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / T
    var = np.diag(S)
    sd = np.sqrt(var)
    corr = S / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    # pi: sum of the asymptotic variances of the sample covariances
    phi = (Y ** 2).T @ (Y ** 2) / T - 2 * (Y.T @ Y) * S / T + S ** 2
    pi = phi.sum()
    # rho: asymptotic covariances between the target and the sample covariances
    help_ = Y.T @ Y / T
    theta = (Y ** 3).T @ Y / T - np.diag(help_)[:, None] * S - help_ * var[:, None] + var[:, None] * S
    np.fill_diagonal(theta, 0)
    rho = np.trace(phi) + r_bar * ((sd[None, :] / sd[:, None]) * theta).sum()
    gamma = ((F - S) ** 2).sum()
    delta = max(0.0, min(1.0, (pi - rho) / gamma / T))
    return delta * F + (1 - delta) * S, delta


def min_variance_weights(Sigma):
    # long-only global minimum variance portfolio
    n = len(Sigma)
    res = optimize.minimize(lambda w: w @ Sigma @ w, np.full(n, 1 / n), jac=lambda w: 2 * Sigma @ w, method="SLSQP",
                            bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                            options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance_unconstrained(Sigma):
    # global minimum variance with short selling allowed: w = S^-1 1 / 1' S^-1 1
    w = np.linalg.solve(Sigma, np.ones(len(Sigma)))
    return w / w.sum()


def risk_parity_weights(Sigma):
    # equal contribution to the portfolio variance (Spinu 2013): minimise w'Sw / 2 - sum(log w) / n, then rescale
    n = len(Sigma)
    w0 = 1 / np.sqrt(np.diag(Sigma))
    res = optimize.minimize(lambda w: 0.5 * w @ Sigma @ w - np.log(w).sum() / n, w0 / w0.sum(),
                            jac=lambda w: Sigma @ w - 1 / (n * w), method="L-BFGS-B", bounds=[(1e-10, None)] * n,
                            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
    return res.x / res.x.sum()


def backtest(R, weights_fn, window=60, cost=0.0):
    # monthly out-of-sample backtest. at the start of month t the target weights come from weights_fn(past, t),
    # past = the `window` previous months of R. holdings drift during the month; the move back to the target
    # costs `cost` per unit traded (0.001 = 10 bp). returns net monthly returns and turnover
    X = R.values
    T, n = X.shape
    net, turn = np.empty(T - window), np.zeros(T - window)
    w_drift = None
    for k, t in enumerate(range(window, T)):
        w = weights_fn(X[t - window:t], t)
        if w_drift is not None:
            turn[k] = np.abs(w - w_drift).sum()
        gross = w @ X[t]
        net[k] = gross - cost * turn[k]
        w_drift = w * (1 + X[t]) / (1 + gross)
    idx = R.index[window:]
    return pd.Series(net, index=idx), pd.Series(turn, index=idx)


def ols_newey_west(y, X, lags=6):
    # OLS with a constant, Newey-West (Bartlett) standard errors. returns coefficients and t-stats
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta = ZtZ_inv @ Z.T @ y
    u = y - Z @ beta
    g = Z * u[:, None]
    S = g.T @ g
    for L in range(1, lags + 1):
        G = g[L:].T @ g[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(ZtZ_inv @ S @ ZtZ_inv))
    return beta, beta / se


def sharpe_test(e1, e2):
    # Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction, on two series of monthly
    # excess returns over the same months. returns the z statistic and the two-sided p-value
    from scipy import stats as st
    e1, e2 = np.asarray(e1), np.asarray(e2)
    T = len(e1)
    s1, s2 = e1.mean() / e1.std(), e2.mean() / e2.std()
    rho = np.corrcoef(e1, e2)[0, 1]
    v = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    z = (s1 - s2) / np.sqrt(v)
    return z, 2 * st.norm.sf(abs(z))


def stats(r, rf):
    # annualised statistics of monthly simple returns r, rf the monthly risk-free rate
    ex = r - rf
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {"return": 12 * r.mean(), "vol": np.sqrt(12) * r.std(), "sharpe": np.sqrt(12) * ex.mean() / ex.std(),
            "max drawdown": dd.min(), "worst month": r.min()}
