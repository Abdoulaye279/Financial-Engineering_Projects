import numpy as np
import pandas as pd
from alloc import min_variance_unconstrained, sharpe_test, ledoit_wolf_cc, min_variance_weights, risk_parity_weights, backtest, ols_newey_west, stats

rng = np.random.default_rng(0)


def true_cov(n=30, rho=0.5):
    vol = np.linspace(0.03, 0.09, n)
    C = np.full((n, n), rho)
    np.fill_diagonal(C, 1)
    return C * np.outer(vol, vol)


def block_cov(n=30):
    # two groups of industries: correlation 0.7 inside a group, 0.2 across -> not a constant-correlation matrix
    g = np.arange(n) < n // 2
    C = np.where(g[:, None] == g[None, :], 0.7, 0.2)
    np.fill_diagonal(C, 1)
    vol = np.linspace(0.03, 0.09, n)
    return C * np.outer(vol, vol)


def test_ledoit_wolf_right_target():
    # truth = constant correlation: the target is right, shrinkage ~1 and far better than the sample with 60 months
    Sig = true_cov()
    err_lw, err_s = [], []
    for _ in range(50):
        X = rng.multivariate_normal(np.zeros(30), Sig, 60)
        L, d = ledoit_wolf_cc(X)
        err_lw.append(((L - Sig) ** 2).sum())
        err_s.append(((np.cov(X.T, bias=True) - Sig) ** 2).sum())
        assert 0 <= d <= 1 and np.linalg.eigvalsh(L).min() > 0
    assert np.mean(err_lw) < 0.7 * np.mean(err_s)
    assert ledoit_wolf_cc(rng.multivariate_normal(np.zeros(30), Sig, 2000))[1] > 0.9


def test_ledoit_wolf_wrong_target():
    # truth = two blocks (0.7 inside, 0.2 across): the target is wrong, so shrinkage must fade as data grows
    # and must not make the estimate clearly worse than the sample
    Sig = block_cov()
    d = [ledoit_wolf_cc(rng.multivariate_normal(np.zeros(30), Sig, T))[1] for T in (60, 1000, 20_000)]
    assert d[0] > d[1] > d[2] and d[2] < 0.01
    err_lw, err_s = [], []
    for _ in range(50):
        X = rng.multivariate_normal(np.zeros(30), Sig, 60)
        err_lw.append(((ledoit_wolf_cc(X)[0] - Sig) ** 2).sum())
        err_s.append(((np.cov(X.T, bias=True) - Sig) ** 2).sum())
    assert np.mean(err_lw) < 1.1 * np.mean(err_s)


def test_min_variance_and_risk_parity():
    Sig = true_cov(5)
    w = min_variance_weights(Sig)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert w @ Sig @ w <= np.full(5, 0.2) @ Sig @ np.full(5, 0.2)
    # two uncorrelated assets: unconstrained min variance is w ~ 1 / var, long-only gives the same here
    D = np.diag([0.01, 0.04])
    assert np.allclose(min_variance_weights(D), [0.8, 0.2], atol=1e-5)
    rp = risk_parity_weights(Sig)
    contrib = rp * (Sig @ rp) / (rp @ Sig @ rp)
    assert np.allclose(contrib, 0.2, atol=1e-5)


def test_backtest_drift_turnover_and_costs():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    R = pd.DataFrame({"a": [0.0, 0.0, 0.10, 0.0], "b": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    half = lambda past, t: np.array([0.5, 0.5])
    net, turn = backtest(R, half, window=2, cost=0.01)
    # month 3: 50/50, +5%. month 4: a drifted to 0.55 / 1.05, back to 0.5 costs 2 * (0.55 / 1.05 - 0.5) * 1%
    assert abs(net.iloc[0] - 0.05) < 1e-12 and turn.iloc[0] == 0
    to = 2 * (0.55 / 1.05 - 0.5)
    assert abs(turn.iloc[1] - to) < 1e-12 and abs(net.iloc[1] + 0.01 * to) < 1e-12


def test_ols_newey_west_and_stats():
    x = rng.normal(size=(5000, 2))
    y = 0.01 + x @ np.array([1.0, -0.5]) + rng.normal(0, 0.1, 5000)
    b, t = ols_newey_west(y, x)
    assert np.allclose(b, [0.01, 1.0, -0.5], atol=0.01) and abs(t[1]) > 100
    r = pd.Series([0.1, -0.5, 0.2])
    assert abs(stats(r, 0.0)["max drawdown"] + 0.5) < 1e-12


def test_unconstrained_min_variance_and_sharpe_test():
    D = np.diag([0.01, 0.04])
    assert np.allclose(min_variance_unconstrained(D), [0.8, 0.2])
    Sig = block_cov(5)
    w = min_variance_unconstrained(Sig)
    assert abs(w.sum() - 1) < 1e-12 and w @ Sig @ w <= min_variance_weights(Sig) @ Sig @ min_variance_weights(Sig) + 1e-12
    # same true Sharpe ratio: the test should reject at 5% about 5% of the time
    rej = 0
    for _ in range(400):
        a, b = rng.normal(0.01, 0.05, 600), rng.normal(0.01, 0.05, 600)
        rej += sharpe_test(a, b)[1] < 0.05
    assert 0.02 < rej / 400 < 0.09
    assert sharpe_test(rng.normal(0.03, 0.05, 600), rng.normal(0.0, 0.05, 600))[1] < 0.001
