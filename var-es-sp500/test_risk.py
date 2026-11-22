import numpy as np
import pandas as pd
from scipy import stats
from risk import stress_correlation, rebalanced_returns, backtest_weights, risk_parity_weights, monthly_risk_parity, rolling_equal_risk_weights, gjr_skt_paths, backtest_10d, gjr_skt_fit, skt_logpdf, skt_ppf, skt_var_es, garch_skt_fit, gjr_filter, gjr_t_fit, garch_t_fit, t_var_es, dcc_filter, dcc_fit, t_copula_fit, es_contributions, equal_risk_weights, portfolio_backtest, evt_tail, var_es_normal, var_es_hist, garch_filter, garch_fit, kupiec, christoffersen, backtest

rng = np.random.default_rng(0)


def simulate_garch(n, omega=2e-6, a=0.08, b=0.9):
    r, s2 = np.empty(n), omega / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * r[t] ** 2 + b * s2
    return r


def test_normal_and_hist_on_normal_data():
    r = rng.normal(0, 0.01, 1_000_000)
    var_n, es_n = var_es_normal(r)
    var_h, es_h = var_es_hist(r)
    assert abs(var_n - 0.02326) < 1e-4 and abs(es_n - 0.02665) < 1e-4   # 2.326 sd and 2.665 sd
    assert abs(var_h - var_n) < 2e-4 and abs(es_h - es_n) < 2e-4


def test_garch_recovers_parameters():
    omega, a, b = garch_fit(simulate_garch(20_000))
    assert abs(a - 0.08) < 0.02 and abs(b - 0.9) < 0.03
    assert abs(omega / (1 - a - b) - 1e-4) < 3e-5   # long-run variance


def test_evt_on_student_t():
    # fat tailed data: EVT should find the true 99% quantile and ES of a Student-t(4)
    x = stats.t.rvs(4, size=200_000, random_state=1)
    var, es = evt_tail(x)
    true_var = stats.t.ppf(0.99, 4)
    true_es = stats.t.expect(lambda y: y, args=(4,), lb=true_var) / 0.01
    assert abs(var / true_var - 1) < 0.03 and abs(es / true_es - 1) < 0.05


def test_kupiec():
    good = np.zeros(10_000, bool); good[::100] = True       # exactly 1%
    bad = np.zeros(10_000, bool); bad[::40] = True          # 2.5%
    assert kupiec(good)[1] > 0.5 and kupiec(bad)[1] < 1e-6


def test_christoffersen():
    spread = np.zeros(10_000, bool); spread[::100] = True
    clustered = np.zeros(10_000, bool)
    for s in range(0, 10_000, 500):
        clustered[s:s + 5] = True                          # breaches come in groups of 5
    assert christoffersen(spread)[1] > 0.05 and christoffersen(clustered)[1] < 1e-6


def test_backtest_on_garch_data():
    r = pd.Series(simulate_garch(3000), index=pd.bdate_range("2000-01-03", periods=3000))
    bt = backtest(r, window=1000, refit=250)
    assert len(bt) == 2000 and not bt.isna().any().any()
    rate = (bt["ret"] < -bt["var_garch"]).mean()
    assert 0.003 < rate < 0.02    # close to 1% when the model is right
    for m in ["normal", "hist", "garch", "fhs", "evt", "gt", "gjr", "skt", "gjrskt"]:
        assert (bt[f"es_{m}"] >= bt[f"var_{m}"]).all()


def test_portfolio_backtest():
    # two correlated GARCH assets: both models should give about 1% breaches and ES above VaR
    n = 3000
    a, b = simulate_garch(n), simulate_garch(n)
    R = pd.DataFrame({"x": a, "y": 0.6 * a + 0.8 * b}, index=pd.bdate_range("2000-01-03", periods=n))
    bt = portfolio_backtest(R, [0.5, 0.5], window=1000, refit=250)
    assert len(bt) == 2000 and not bt.isna().any().any()
    for m in ["ewma", "mfhs", "dcc"]:
        assert 0.003 < (bt["ret"] < -bt[f"var_{m}"]).mean() < 0.02
        assert (bt[f"es_{m}"] > bt[f"var_{m}"]).all()


def test_dcc_follows_a_change_in_correlation():
    # correlation 0.2 for 2000 days then 0.8: the DCC correlation should move from one to the other
    def block(rho, n):
        return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    Z = np.vstack([block(0.2, 2000), block(0.8, 2000)])
    a, b = dcc_fit(Z)
    r = dcc_filter(Z, a, b)[:-1, 0, 1]
    assert a > 0 and a + b < 1
    assert abs(r[1000:2000].mean() - 0.2) < 0.15 and abs(r[3000:].mean() - 0.8) < 0.1


def test_t_copula_recovers_parameters():
    C = np.array([[1, 0.5], [0.5, 1]])
    X = stats.multivariate_t(shape=C, df=4).rvs(20_000, random_state=2)
    rho, nu = t_copula_fit(X)
    assert abs(rho - 0.5) < 0.03 and 3 < nu < 5.5


def test_equal_risk_weights():
    # asset 2 is twice as volatile and independent: it should get about a third of the money
    S = np.column_stack([rng.normal(0, 0.01, 200_000), rng.normal(0, 0.02, 200_000)])
    w = equal_risk_weights(S)
    assert abs(w[1] - 1 / 3) < 0.03
    assert np.allclose(es_contributions(S, w), 0.5, atol=0.01)


def test_garch_t_recovers_parameters():
    n, omega, a, b, nu = 20_000, 2e-6, 0.08, 0.9, 5
    z = stats.t.rvs(nu, size=n, random_state=3) * np.sqrt((nu - 2) / nu)   # variance 1
    r, s2 = np.empty(n), omega / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + a * r[t] ** 2 + b * s2
    o_hat, a_hat, b_hat, nu_hat = garch_t_fit(r)
    assert abs(a_hat - a) < 0.02 and abs(b_hat - b) < 0.03 and 4 < nu_hat < 6.5


def test_t_var_es_matches_simulation():
    x = stats.t.rvs(5, size=2_000_000, random_state=4) * np.sqrt(3 / 5)
    var, es = t_var_es(5)
    q = np.quantile(x, 0.01)
    assert abs(var + q) < 0.01 and abs(es + x[x <= q].mean()) < 0.02


def test_gjr_recovers_asymmetry():
    n, omega, a, g, b, nu = 20_000, 2e-6, 0.02, 0.12, 0.9, 6
    z = stats.t.rvs(nu, size=n, random_state=5) * np.sqrt((nu - 2) / nu)
    r, s2 = np.empty(n), omega / (1 - a - g / 2 - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + (a + g * (r[t] < 0)) * r[t] ** 2 + b * s2
    o_hat, a_hat, g_hat, b_hat, nu_hat = gjr_t_fit(r)
    assert abs(a_hat - a) < 0.02 and abs(g_hat - g) < 0.03 and abs(b_hat - b) < 0.03 and 4.5 < nu_hat < 8
    # with g = 0 it is the plain GARCH filter
    x = rng.normal(0, 0.01, 500)
    assert np.allclose(gjr_filter(x, 1e-6, 0.1, 0.0, 0.85), garch_filter(x, 1e-6, 0.1, 0.85))


def test_skewed_t_is_a_standardised_density():
    from scipy.integrate import quad
    for eta, lam in [(5, -0.3), (8, 0.2), (30, 0.0)]:
        f = lambda z: np.exp(skt_logpdf(np.array([z]), eta, lam))[0]
        m0, m1, m2 = (quad(lambda z: z ** k * f(z), -np.inf, np.inf, limit=200)[0] for k in range(3))
        assert abs(m0 - 1) < 1e-6 and abs(m1) < 1e-6 and abs(m2 - 1) < 1e-4
        # ppf inverts the cdf and VaR / ES match a direct integration
        q = skt_ppf(0.01, eta, lam)
        assert abs(quad(f, -np.inf, q)[0] - 0.01) < 1e-7
        var, es = skt_var_es(eta, lam)
        assert abs(var + q) < 1e-12 and abs(es + quad(lambda z: z * f(z), -np.inf, q)[0] / 0.01) < 1e-6
    # lam = 0 is the plain standardised t
    assert abs(skt_var_es(6, 0.0)[0] - t_var_es(6)[0]) < 1e-10


def test_garch_skt_recovers_parameters():
    n, omega, a, b, eta, lam = 20_000, 2e-6, 0.08, 0.9, 6, -0.25
    z = skt_ppf(np.random.default_rng(6).random(n), eta, lam)
    r, s2 = np.empty(n), omega / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + a * r[t] ** 2 + b * s2
    o_hat, a_hat, b_hat, eta_hat, lam_hat = garch_skt_fit(r)
    assert abs(a_hat - a) < 0.02 and abs(b_hat - b) < 0.03 and 4.5 < eta_hat < 8 and abs(lam_hat - lam) < 0.05


def test_gjr_skt_recovers_parameters():
    n, omega, a, g, b, eta, lam = 20_000, 2e-6, 0.02, 0.12, 0.9, 6, -0.25
    z = skt_ppf(np.random.default_rng(7).random(n), eta, lam)
    r, s2 = np.empty(n), omega / (1 - a - g / 2 - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + (a + g * (r[t] < 0)) * r[t] ** 2 + b * s2
    o_h, a_h, g_h, b_h, eta_h, lam_h = gjr_skt_fit(r)
    assert abs(a_h - a) < 0.02 and abs(g_h - g) < 0.03 and abs(b_h - b) < 0.03
    assert 4.5 < eta_h < 8 and abs(lam_h - lam) < 0.05


def test_10_day_paths_match_the_normal_case():
    # no vol dynamics (a = g = b = 0) and a nearly normal skewed t: 10-day VaR = 2.326 * sigma * sqrt(10)
    sig = 0.01
    var, es = gjr_skt_paths((sig ** 2, 0, 0, 0, 200, 0.0), sig ** 2, 0.0, h=10, n=400_000)
    assert abs(var / (2.326 * sig * np.sqrt(10)) - 1) < 0.02
    assert abs(es / (2.665 * sig * np.sqrt(10)) - 1) < 0.03


def test_backtest_10d_shapes():
    r = pd.Series(simulate_garch(3000), index=pd.bdate_range("2000-01-03", periods=3000))
    bt = backtest_10d(r, window=1000, refit=250, n_sims=2000)
    assert len(bt) == 200 and not bt.isna().any().any()
    for m in ["normal", "hist", "hist10", "sqrt", "mc"]:
        assert (bt[f"es_{m}"] > bt[f"var_{m}"]).all()


def test_dcc_and_weights_with_three_assets():
    # 3 assets, the third one moves against the first two on bad days (like bonds)
    C = np.array([[1, 0.6, -0.3], [0.6, 1, -0.3], [-0.3, -0.3, 1]])
    Z = rng.multivariate_normal(np.zeros(3), C, 4000)
    a, b = dcc_fit(Z)
    R = dcc_filter(Z, a, b)
    assert R.shape == (4001, 3, 3) and np.allclose(R[:, range(3), range(3)], 1)
    assert np.allclose(R[-1], C, atol=0.15)
    S = Z * np.array([0.01, 0.02, 0.005])
    w = equal_risk_weights(S)
    assert (w > 0).all() and abs(w.sum() - 1) < 1e-9
    assert np.allclose(es_contributions(S, w), 1 / 3, atol=0.03)


def test_rolling_weights_use_only_the_past():
    # asset 2 is calm for 3 years then 4 times more volatile: its weight must only drop the year after the change
    idx = pd.bdate_range("2000-01-03", periods=2000)
    vol2 = np.where(idx < "2003-01-01", 0.01, 0.04)
    S = pd.DataFrame({"a": rng.normal(0, 0.01, 2000), "b": rng.normal(0, 1, 2000) * vol2}, index=idx)
    W = rolling_equal_risk_weights(S, window=500)
    assert W.loc[:"2001-12-31"].isna().all().all()        # first year only after the window
    assert np.allclose(W.dropna().sum(axis=1), 1)
    assert W.loc["2002-06-03"].nunique() == 2 and abs(W.loc["2002-06-03", "b"] - 0.5) < 0.05   # same vol so far
    assert W.loc["2004-06-01", "b"] < 0.3                    # the change is seen once it is in the past window
    assert (W.loc["2003-01-02":"2003-12-31", "b"] == W.loc["2003-01-02", "b"]).all()   # held all year


def test_risk_parity_weights():
    C = np.array([[1, 0.5, -0.2], [0.5, 1, -0.2], [-0.2, -0.2, 1]])
    vol = np.array([0.2, 0.3, 0.1])
    Sigma = C * np.outer(vol, vol)
    w = risk_parity_weights(Sigma)
    contrib = w * (Sigma @ w) / (w @ Sigma @ w)
    assert np.allclose(contrib, 1 / 3, atol=1e-5) and abs(w.sum() - 1) < 1e-12
    # uncorrelated assets: w proportional to 1 / vol
    w2 = risk_parity_weights(np.diag(vol ** 2))
    assert np.allclose(w2, (1 / vol) / (1 / vol).sum(), atol=1e-6)


def test_monthly_risk_parity_follows_a_vol_change():
    idx = pd.bdate_range("2000-01-03", periods=2000)
    vol2 = np.where(idx < "2005-01-01", 0.01, 0.03)
    R = pd.DataFrame({"a": rng.normal(0, 0.01, 2000), "b": rng.normal(0, 1, 2000) * vol2}, index=idx)
    for method in ["sample", "dcc"]:
        W = monthly_risk_parity(R, method=method, window=1000, refit=250)
        assert np.allclose(W.dropna().sum(axis=1), 1)
        assert abs(W.loc["2004-06-01", "b"] - 0.5) < 0.1 and W.loc["2006-06-01", "b"] < 0.35


def test_rebalanced_returns():
    idx = pd.bdate_range("2000-01-03", periods=60)
    S = pd.DataFrame({"a": np.full(60, 0.01), "b": np.zeros(60)}, index=idx)
    W = pd.DataFrame(0.5, index=idx, columns=["a", "b"])
    net, turn = rebalanced_returns(S, W)
    # day 1: 50/50 -> +0.5%. then a grows, so its weight drifts up until the month changes
    assert abs(net.iloc[0] - 0.005) < 1e-12 and net.iloc[1] > 0.005
    first_of_feb = idx[idx.month == 2][0]
    assert turn[turn > 0].index.tolist() == [first_of_feb, idx[idx.month == 3][0]]
    # drifted weight of a after the January days, then back to 0.5: turnover = 2 * (wa - 0.5)
    n_jan = (idx.month == 1).sum()
    wa = 0.5 * 1.01 ** n_jan / (0.5 * 1.01 ** n_jan + 0.5)
    assert abs(turn[first_of_feb] - 2 * (wa - 0.5)) < 1e-12
    net_c, _ = rebalanced_returns(S, W, cost=0.001)
    assert abs((net - net_c)[first_of_feb] - 0.001 * turn[first_of_feb]) < 1e-12


def test_backtest_weights_matches_fixed_weights():
    # with constant weights it must give the same VaR as backtest() on the portfolio, block by block
    idx = pd.bdate_range("2000-01-03", periods=1500)
    S = pd.DataFrame({"a": rng.normal(0, 0.01, 1500), "b": rng.normal(0, 0.02, 1500)}, index=idx)
    W = pd.DataFrame(0.5, index=idx, columns=["a", "b"])
    W.iloc[1200:] = [0.7, 0.3]
    bt = backtest_weights(S, W, window=1000)
    assert len(bt) == 500 and not bt.isna().any().any()
    ref = backtest(np.log1p(S.iloc[200:] @ np.array([0.7, 0.3])), window=1000, refit=300)
    assert np.allclose(bt["var_hist"].iloc[200:], ref["var_hist"]) and np.allclose(bt["ret"].iloc[200:], ref["ret"])


def test_stress_correlation():
    C = np.array([[1, 0.8, -0.4], [0.8, 1, -0.4], [-0.4, -0.4, 1]])
    D = stress_correlation(C, [0, 1], [2], 0.6)
    assert np.allclose(np.diag(D), 1) and np.linalg.eigvalsh(D).min() > -1e-10
    assert np.allclose(D[0, 2], 0.6) and np.allclose(D[0, 1], 0.8)   # feasible: no change needed
    # infeasible target: 0 and 1 perfectly correlated but asked to be +0.9 and -0.9 with 2 -> projected, still valid
    E = stress_correlation(np.array([[1, 1.0, 0], [1.0, 1, 0], [0, 0, 1]]), [0], [2], 0.9)
    E = stress_correlation(E, [1], [2], -0.9)
    assert np.allclose(np.diag(E), 1) and np.linalg.eigvalsh(E).min() > -1e-10
