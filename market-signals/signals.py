# Is there anything to predict in daily prices? Direction forecasts, regime models and pairs,
# each checked with a test that compares it to the right baseline.
import numpy as np
import pandas as pd
from scipy import stats


def load_close(path):
    # daily close from the csv files of this repo (dates have a time zone, keep only the day)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"].str[:10])
    return df.set_index("Date")["Close"]


def features(r):
    # predictors known at the close of day t for the sign of r[t+1]
    X = pd.concat({f"lag{k}": r.shift(k - 1) for k in range(1, 6)}, axis=1)
    X["vol21"] = r.rolling(21).std()
    X["mom21"] = r.rolling(21).sum()
    X["mom252"] = r.rolling(252).sum()
    X["dd252"] = np.log(np.exp(r.cumsum()) / np.exp(r.cumsum()).rolling(252).max())
    return X


def walk_forward(X, y, make_model, first_test_year):
    # refit every January on all the data before it, predict that year: no look-ahead
    pred = pd.Series(np.nan, index=X.index)
    for year in range(first_test_year, X.index.year.max() + 1):
        tr, te = X.index.year < year, X.index.year == year
        if te.any():
            pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred.dropna().astype(int)


def mcnemar_exact(right_a, right_b):
    # paired test of two classifiers on the same days: only the days where exactly one is right count.
    # under H0 each of those days goes to a or b with probability 1/2. returns (a only, b only, p two-sided)
    a, b = np.asarray(right_a, bool), np.asarray(right_b, bool)
    n_a, n_b = int((a & ~b).sum()), int((~a & b).sum())
    return n_a, n_b, stats.binomtest(n_a, n_a + n_b, 0.5).pvalue


def hmm2_fit(r, iters=500, tol=1e-8):
    # two-state Gaussian hidden Markov model fitted by EM (Baum-Welch), scaled forward-backward.
    # returns dict with mu, sd, P (transition matrix), filtered and smoothed probabilities of each state, loglik
    r = np.asarray(r, float)
    T = len(r)
    q = np.quantile(np.abs(r), 0.8)
    mu, sd = np.array([r.mean(), r.mean()]), np.array([r[np.abs(r) < q].std(), r[np.abs(r) >= q].std()])
    P, pi0, ll_old = np.array([[0.98, 0.02], [0.05, 0.95]]), np.array([0.5, 0.5]), -np.inf
    for _ in range(iters):
        dens = stats.norm.pdf(r[:, None], mu, sd)
        alpha, c = np.empty((T, 2)), np.empty(T)
        a = pi0 * dens[0]
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ P) * dens[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            beta[t] = P @ (dens[t + 1] * beta[t + 1]) / c[t + 1]
        g = alpha * beta
        xi = (alpha[:-1, :, None] * P[None] * (dens[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        ll = np.log(c).sum()
        pi0 = g[0]
        P = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        mu = (g * r[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (r[:, None] - mu) ** 2).sum(0) / g.sum(0))
        if ll - ll_old < tol:
            break
        ll_old = ll
    k = np.argsort(sd)  # state 0 = calm (low vol), state 1 = turbulent
    return {"mu": mu[k], "sd": sd[k], "P": P[np.ix_(k, k)], "filtered": alpha[:, k], "smoothed": g[:, k], "loglik": ll}


def df_tstat(e):
    # Dickey-Fuller t-statistic on a residual series (no constant, no lags): de[t] = b e[t-1] + u
    e = np.asarray(e, float)
    de, x = np.diff(e), e[:-1] - e[:-1].mean()
    b = x @ de / (x @ x)
    u = de - b * x
    return b / np.sqrt(u.var() / (x @ x))


def engle_granger(ya, yb):
    # regress log price a on log price b, DF t-stat of the residual. returns (beta, intercept, t)
    beta, c = np.polyfit(yb, ya, 1)
    return beta, c, df_tstat(ya - beta * yb - c)


def eg_critical(n, alpha=0.05, sims=4000, seed=0):
    # critical value of the Engle-Granger test for n days, by simulation: two independent random walks
    rng = np.random.default_rng(seed)
    t = [engle_granger(*rng.normal(size=(2, n)).cumsum(1))[2] for _ in range(sims)]
    return np.quantile(t, alpha)


def hmm2_filter(r, mu, sd, P):
    # real-time probability of each state given returns up to t, with fixed parameters (e.g. fitted on a past sample)
    r = np.asarray(r, float)
    dens = stats.norm.pdf(r[:, None], mu, sd)
    w, vecs = np.linalg.eig(P.T)
    a = np.real(vecs[:, np.argmax(np.real(w))])
    a = a / a.sum()  # start from the stationary distribution
    out = np.empty((len(r), 2))
    for t in range(len(r)):
        a = (a @ P if t else a) * dens[t]
        a = a / a.sum()
        out[t] = a
    return out



def sharpe(r, rf):
    # annualised Sharpe ratio of daily returns over the daily risk-free rate
    ex = np.asarray(r) - np.asarray(rf)
    return np.sqrt(252) * ex.mean() / ex.std()

def sharpe_test(e1, e2):
    # Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction, same days.
    # returns the z statistic and the two-sided p-value
    e1, e2 = np.asarray(e1), np.asarray(e2)
    s1, s2 = e1.mean() / e1.std(), e2.mean() / e2.std()
    rho = np.corrcoef(e1, e2)[0, 1]
    z = (s1 - s2) / np.sqrt((2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / len(e1))
    return z, 2 * stats.norm.sf(abs(z))


def hmm_loglik(r, mu, sd, P, nu=None):
    # log-likelihood of a K-state hidden Markov model by the scaled forward pass.
    # Gaussian states, or Student-t states with nu degrees of freedom (sd is then the scale)
    r = np.asarray(r, float)
    dens = stats.norm.pdf(r[:, None], mu, sd) if nu is None else stats.t.pdf((r[:, None] - mu) / sd, nu) / sd
    w, vecs = np.linalg.eig(P.T)
    a = np.real(vecs[:, np.argmax(np.real(w))])
    a = a / a.sum()
    ll = 0.0
    for t in range(len(r)):
        a = (a @ P if t else a) * dens[t]
        c = a.sum()
        ll += np.log(c)
        a = a / c
    return ll


def hmm_fit_mle(r, k=2, student=False, start=None, n_starts=1, seed=0):
    # direct maximum likelihood (L-BFGS on unconstrained parameters) for k states, Gaussian or Student-t.
    # transition rows through a softmax, sd through exp, nu = 2 + exp(.). n_starts > 1 adds random perturbations
    # of the start and keeps the best fit (the likelihood has local optima). returns params, loglik, n_params, all logliks
    r = np.asarray(r, float)

    def unpack(th):
        mu, sd = th[:k], np.exp(th[k:2 * k])
        logits = np.zeros((k, k))
        logits[~np.eye(k, dtype=bool)] = th[2 * k:2 * k + k * (k - 1)]
        P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        nu = 2 + np.exp(th[2 * k + k * (k - 1):]) if student else None
        return mu, sd, P, nu

    if start is None:
        qs = np.quantile(np.abs(r - r.mean()), np.linspace(0.3, 0.95, k))
        start = np.r_[np.full(k, r.mean()), np.log(np.sort(qs) * 1.2), np.full(k * (k - 1), -4.0),
                      np.full(k, np.log(6.0)) if student else []]
    from scipy import optimize
    rng = np.random.default_rng(seed)
    fits = [optimize.minimize(lambda th: -hmm_loglik(r, *unpack(th)), start + (rng.normal(0, 0.5, len(start)) if i else 0),
                              method="L-BFGS-B", options={"maxiter": 500}) for i in range(n_starts)]
    res = min(fits, key=lambda f: f.fun)
    mu, sd, P, nu = unpack(res.x)
    order = np.argsort(sd)
    return {"mu": mu[order], "sd": sd[order], "P": P[np.ix_(order, order)], "nu": None if nu is None else nu[order],
            "loglik": -res.fun, "n_params": len(start), "all_logliks": sorted(-f.fun for f in fits)}


def johansen_trace(Y):
    # Johansen trace statistic for "no cointegration" (r = 0) in a VAR(1) on levels with a constant:
    # regress dY and Y[t-1] on a constant, then -T sum log(1 - eigenvalues of the canonical correlations)
    Y = np.asarray(Y, float)
    dY, Yl = np.diff(Y, axis=0), Y[:-1]
    dY, Yl = dY - dY.mean(0), Yl - Yl.mean(0)
    T = len(dY)
    S00, S11, S01 = dY.T @ dY / T, Yl.T @ Yl / T, dY.T @ Yl / T
    lam = np.linalg.eigvals(np.linalg.solve(S11, S01.T) @ np.linalg.solve(S00, S01)).real
    return -T * np.log(1 - np.clip(lam, 0, 1 - 1e-12)).sum()


def critical_value(stat, n, alpha=0.05, sims=2000, seed=0, upper=False):
    # critical value of a two-series cointegration statistic for n days under the null: two independent random walks
    rng = np.random.default_rng(seed)
    v = [stat(rng.normal(size=(n, 2)).cumsum(0)) for _ in range(sims)]
    return np.quantile(v, 1 - alpha if upper else alpha)
