import numpy as np
import pandas as pd
from signals import johansen_trace, critical_value, hmm_loglik, hmm_fit_mle, mcnemar_exact, walk_forward, hmm2_fit, hmm2_filter, engle_granger, eg_critical, sharpe_test

rng = np.random.default_rng(0)


def test_mcnemar_counts_only_disagreements():
    a = [1, 1, 1, 0, 0, 1]
    b = [1, 0, 0, 1, 0, 1]
    n_a, n_b, p = mcnemar_exact(a, b)
    assert (n_a, n_b) == (2, 1) and abs(p - 1.0) < 1e-12


def test_walk_forward_uses_only_past_years():
    idx = pd.date_range("2010-01-01", "2013-12-31", freq="D")
    X = pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)
    y = pd.Series(idx.year.values, index=idx)
    seen = []

    class LastYear:
        # predicts the last label it was trained on, and records the newest training year
        def fit(self, X, y):
            seen.append(X.index.year.max())
            self.v = y.iloc[-1]
            return self

        def predict(self, X):
            return np.full(len(X), self.v)

    p = walk_forward(X, y, LastYear, 2012)
    assert seen == [2011, 2012]
    assert (p.loc["2012"] == 2011).all() and (p.loc["2013"] == 2012).all()


def test_hmm_recovers_simulated_regimes():
    P = np.array([[0.99, 0.01], [0.03, 0.97]])
    s, states = 0, []
    for _ in range(6000):
        s = rng.choice(2, p=P[s])
        states.append(s)
    states = np.array(states)
    r = rng.normal(np.where(states, -0.1, 0.05), np.where(states, 2.0, 0.7))
    h = hmm2_fit(r)
    assert np.allclose(h["sd"], [0.7, 2.0], rtol=0.06)
    assert np.allclose(np.diag(h["P"]), [0.99, 0.97], atol=0.01)
    assert ((h["smoothed"][:, 1] > 0.5) == states).mean() > 0.95
    f = hmm2_filter(r, h["mu"], h["sd"], h["P"])
    assert np.allclose(f.sum(1), 1) and ((f[:, 1] > 0.5) == states).mean() > 0.9


def test_engle_granger_size_and_power():
    cv = eg_critical(252, sims=2000)
    assert -3.6 < cv < -3.2  # MacKinnon: about -3.37 for two series and 250 observations
    b = rng.normal(size=252).cumsum()
    e = np.zeros(252)
    for t in range(1, 252):
        e[t] = 0.8 * e[t - 1] + rng.normal()  # stationary spread
    assert engle_granger(2 + 0.5 * b + e, b)[2] < cv


def test_sharpe_test_size():
    rej = sum(sharpe_test(rng.normal(0.01, 0.05, 600), rng.normal(0.01, 0.05, 600))[1] < 0.05 for _ in range(400))
    assert 0.02 < rej / 400 < 0.09


def test_hmm_loglik_and_mle():
    from scipy import stats
    r = rng.standard_t(5, 3000)
    # two identical states = one distribution, whatever the transitions
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert abs(hmm_loglik(r, [0, 0], [1, 1], P) - stats.norm.logpdf(r).sum()) < 1e-6
    assert abs(hmm_loglik(r, [0, 0], [1, 1], P, nu=[5, 5]) - stats.t.logpdf(r, 5).sum()) < 1e-6
    # direct maximum likelihood and EM are two separate codes: same fit on regime data
    s = (np.arange(4000) // 400) % 2
    x = rng.normal(0, np.where(s, 2.0, 0.7))
    em, ml = hmm2_fit(x), hmm_fit_mle(x, 2)
    assert np.allclose(ml["sd"], em["sd"], rtol=0.02) and abs(ml["loglik"] - em["loglik"]) < 3


def test_johansen_size_and_power():
    cv = critical_value(johansen_trace, 252, sims=1000, upper=True)
    assert 13 < cv < 18  # Osterwald-Lenum / MacKinnon-Haug-Michelis: about 15.5 for r = 0 with a constant
    b = rng.normal(size=252).cumsum()
    e = np.zeros(252)
    for t in range(1, 252):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    assert johansen_trace(np.column_stack([2 + 0.5 * b + e, b])) > cv
