# VaR and ES models for daily returns: normal, historical, GARCH(1,1), filtered historical
# simulation and GARCH-EVT, plus the Kupiec and Christoffersen backtests.
# VaR and ES are given as positive losses.
import numpy as np
from scipy import stats, optimize
from scipy.special import xlogy


def load_close(path):
    # daily close from the csv files of this repo (dates have a time zone, keep only the day)
    import pandas as pd
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"].str[:10])
    return df.set_index("Date")["Close"]


def var_es_normal(r, alpha=0.99):
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / (1 - alpha))


def var_es_hist(r, alpha=0.99):
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()


def garch_filter(r, omega, a, b):
    # conditional variances; s2[t] uses returns up to t-1, the last one is the forecast for the next day
    s2 = np.empty(len(r) + 1)
    s2[0] = r.var()
    for t in range(len(r)):
        s2[t + 1] = omega + a * r[t] ** 2 + b * s2[t]
    return s2


def garch_fit(r):
    # GARCH(1,1) fitted by maximum likelihood (normal), returns omega, alpha, beta
    x = 100 * (r - r.mean())   # in percent, the optimizer works better with this scale

    def nll(p):
        s2 = garch_filter(x, *p)[:-1]
        return 0.5 * np.sum(np.log(s2) + x ** 2 / s2)

    res = optimize.minimize(nll, [0.05 * x.var(), 0.08, 0.9], method="SLSQP",
                            bounds=[(1e-8, None), (0, 1), (0, 1)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}])
    omega, a, b = res.x
    return omega / 1e4, a, b


def _t_nll(x, s2, nu):
    # minus log-likelihood of x with variances s2 and standardised Student-t errors
    from scipy.special import gammaln
    return -np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2) * s2)
                   - (nu + 1) / 2 * np.log1p(x ** 2 / ((nu - 2) * s2)))


def garch_t_fit(r):
    # GARCH(1,1) with standardised Student-t errors, maximum likelihood. returns omega, alpha, beta, nu
    x = 100 * (r - r.mean())
    res = optimize.minimize(lambda p: _t_nll(x, garch_filter(x, *p[:3])[:-1], p[3]), [0.05 * x.var(), 0.08, 0.9, 8.0],
                            method="SLSQP", bounds=[(1e-8, None), (0, 1), (0, 1), (2.1, 100)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}])
    omega, a, b, nu = res.x
    return omega / 1e4, a, b, nu


def _skt_consts(eta, lam):
    # Hansen (1994) skewed t, mean 0 and variance 1. lam < 0 -> heavier left tail
    from scipy.special import gammaln
    c = np.exp(gammaln((eta + 1) / 2) - gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def skt_logpdf(z, eta, lam):
    a, b, c = _skt_consts(eta, lam)
    side = np.where(z < -a / b, 1 - lam, 1 + lam)
    return np.log(b * c) - (eta + 1) / 2 * np.log1p(((b * z + a) / side) ** 2 / (eta - 2))


def skt_ppf(p, eta, lam):
    a, b, _ = _skt_consts(eta, lam)
    s = np.sqrt((eta - 2) / eta)
    p = np.asarray(p, dtype=float)
    left = p < (1 - lam) / 2
    x = np.where(left, stats.t.ppf(p / (1 - lam), eta), stats.t.ppf(0.5 + (p - (1 - lam) / 2) / (1 + lam), eta))
    return (np.where(left, 1 - lam, 1 + lam) * s * x - a) / b


def skt_var_es(eta, lam, alpha=0.99):
    # VaR and ES (positive losses). on the left side z = ((1 - lam) s x - a) / b with x a plain t,
    # so E[z ; z < q] comes from the t formula E[x ; x < xq] = -(eta + xq^2) / (eta - 1) * f(xq)
    a, b, _ = _skt_consts(eta, lam)
    p = 1 - alpha
    s = np.sqrt((eta - 2) / eta)
    xq = stats.t.ppf(p / (1 - lam), eta)
    tail_x = -(eta + xq ** 2) / (eta - 1) * stats.t.pdf(xq, eta)
    tail_z = (1 - lam) / b * ((1 - lam) * s * tail_x - a * p / (1 - lam))
    return -skt_ppf(p, eta, lam), -tail_z / p


def garch_skt_fit(r):
    # GARCH(1,1) with Hansen skewed-t errors. returns omega, alpha, beta, eta, lam
    x = 100 * (r - r.mean())

    def nll(p):
        s2 = garch_filter(x, *p[:3])[:-1]
        return -np.sum(skt_logpdf(x / np.sqrt(s2), p[3], p[4]) - 0.5 * np.log(s2))

    res = optimize.minimize(nll, [0.05 * x.var(), 0.08, 0.9, 8.0, -0.1], method="SLSQP",
                            bounds=[(1e-8, None), (0, 1), (0, 1), (2.1, 100), (-0.99, 0.99)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}])
    omega, a, b, eta, lam = res.x
    return omega / 1e4, a, b, eta, lam


def gjr_filter(r, omega, a, g, b):
    # GJR-GARCH(1,1): a down day adds g * r^2 on top of a * r^2. same convention as garch_filter
    s2 = np.empty(len(r) + 1)
    s2[0] = r.var()
    for t in range(len(r)):
        s2[t + 1] = omega + (a + g * (r[t] < 0)) * r[t] ** 2 + b * s2[t]
    return s2


def gjr_t_fit(r):
    # GJR-GARCH(1,1) with Student-t errors. returns omega, alpha, gamma, beta, nu
    x = 100 * (r - r.mean())
    res = optimize.minimize(lambda p: _t_nll(x, gjr_filter(x, *p[:4])[:-1], p[4]), [0.05 * x.var(), 0.03, 0.08, 0.9, 8.0],
                            method="SLSQP", bounds=[(1e-8, None), (0, 1), (0, 1), (0, 1), (2.1, 100)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2] / 2 - p[3]}])
    omega, a, g, b, nu = res.x
    return omega / 1e4, a, g, b, nu


def t_var_es(nu, alpha=0.99):
    # VaR and ES (positive losses) of a Student-t rescaled to variance 1
    k = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(1 - alpha, nu)
    es = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q ** 2) / (nu - 1)
    return -q * k, es * k


def evt_tail(losses, alpha=0.99, tail=0.1):
    # fit a generalized Pareto distribution to the largest 10% of losses (peaks over threshold)
    # and use it for the VaR and ES quantile instead of the empirical one
    u = np.quantile(losses, 1 - tail)
    excess = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(excess, floc=0)
    var = u + beta / xi * (((1 - alpha) / tail) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    return var, es


def gjr_skt_fit(r):
    # GJR-GARCH(1,1) with Hansen skewed-t errors. returns omega, alpha, gamma, beta, eta, lam
    x = 100 * (r - r.mean())

    def nll(p):
        s2 = gjr_filter(x, *p[:4])[:-1]
        return -np.sum(skt_logpdf(x / np.sqrt(s2), p[4], p[5]) - 0.5 * np.log(s2))

    res = optimize.minimize(nll, [0.05 * x.var(), 0.03, 0.08, 0.9, 8.0, -0.1], method="SLSQP",
                            bounds=[(1e-8, None), (0, 1), (0, 1), (0, 1), (2.1, 100), (-0.99, 0.99)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2] / 2 - p[3]}])
    omega, a, g, b, eta, lam = res.x
    return omega / 1e4, a, g, b, eta, lam


def kupiec(breaches, alpha=0.99):
    # is the breach rate equal to 1 - alpha? returns LR stat and p-value
    T, x, p = len(breaches), breaches.sum(), 1 - alpha
    lr = -2 * (xlogy(T - x, 1 - p) + xlogy(x, p) - xlogy(T - x, 1 - x / T) - xlogy(x, x / T))
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(breaches):
    # is a breach more likely the day after a breach? returns LR stat and p-value
    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00, n01 = np.sum((prev == 0) & (cur == 0)), np.sum((prev == 0) & (cur == 1))
    n10, n11 = np.sum((prev == 1) & (cur == 0)), np.sum((prev == 1) & (cur == 1))
    p0, p1 = n01 / (n00 + n01), n11 / max(n10 + n11, 1)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_h0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll_h1 = xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
    lr = -2 * (ll_h0 - ll_h1)
    return lr, stats.chi2.sf(lr, 1)


def backtest(r, alpha=0.99, window=1000, refit=250):
    # VaR and ES for the next day, for every day after the first `window` days.
    # each model only sees the past `window` returns, GARCH is refitted every `refit` days
    import pandas as pd
    v = r.values
    out = {k: np.full(len(v), np.nan) for k in
           ["var_normal", "es_normal", "var_hist", "es_hist", "var_garch", "es_garch", "var_fhs", "es_fhs", "var_evt", "es_evt", "var_gt", "es_gt", "var_gjr", "es_gjr", "var_skt", "es_skt", "var_gjrskt", "es_gjrskt"]}
    zq = stats.norm.ppf(1 - alpha)
    for start in range(window, len(v), refit):
        past = v[start - window:start]
        mu = past.mean()
        omega, a, b = garch_fit(past)
        # filter through the fitting window and the next block with the same parameters
        block = v[start - window:min(start + refit, len(v))] - mu
        s = np.sqrt(garch_filter(block, omega, a, b))
        z = (past - mu) / s[:window]            # standardised residuals of the fitting window
        q = np.quantile(z, 1 - alpha)
        z_tail = z[z <= q].mean()
        evt_var, evt_es = evt_tail(-z, alpha)
        o_t, a_t, b_t, nu = garch_t_fit(past)
        s_t = np.sqrt(garch_filter(block, o_t, a_t, b_t))
        t_var, t_es = t_var_es(nu, alpha)
        o_j, a_j, g_j, b_j, nu_j = gjr_t_fit(past)
        s_j = np.sqrt(gjr_filter(block, o_j, a_j, g_j, b_j))
        j_var, j_es = t_var_es(nu_j, alpha)
        o_k, a_k, b_k, eta_k, lam_k = garch_skt_fit(past)
        s_k = np.sqrt(garch_filter(block, o_k, a_k, b_k))
        k_var, k_es = skt_var_es(eta_k, lam_k, alpha)
        o_m, a_m, g_m, b_m, eta_m, lam_m = gjr_skt_fit(past)
        s_m = np.sqrt(gjr_filter(block, o_m, a_m, g_m, b_m))
        m_var, m_es = skt_var_es(eta_m, lam_m, alpha)
        for t in range(start, min(start + refit, len(v))):
            w = v[t - window:t]
            out["var_normal"][t], out["es_normal"][t] = var_es_normal(w, alpha)
            out["var_hist"][t], out["es_hist"][t] = var_es_hist(w, alpha)
            st = s[t - start + window]            # GARCH vol forecast for day t
            out["var_garch"][t] = -(mu + st * zq)
            out["es_garch"][t] = -(mu - st * stats.norm.pdf(zq) / (1 - alpha))
            out["var_fhs"][t] = -(mu + st * q)
            out["es_fhs"][t] = -(mu + st * z_tail)
            out["var_evt"][t] = -mu + st * evt_var
            out["es_evt"][t] = -mu + st * evt_es
            out["var_gt"][t] = -mu + s_t[t - start + window] * t_var
            out["es_gt"][t] = -mu + s_t[t - start + window] * t_es
            out["var_gjr"][t] = -mu + s_j[t - start + window] * j_var
            out["es_gjr"][t] = -mu + s_j[t - start + window] * j_es
            out["var_skt"][t] = -mu + s_k[t - start + window] * k_var
            out["es_skt"][t] = -mu + s_k[t - start + window] * k_es
            out["var_gjrskt"][t] = -mu + s_m[t - start + window] * m_var
            out["es_gjrskt"][t] = -mu + s_m[t - start + window] * m_es
    df = pd.DataFrame(out, index=r.index)
    df.insert(0, "ret", v)
    return df.iloc[window:]


def dcc_filter(Z, a, b):
    # DCC(1,1) correlation matrices from standardised residuals Z (days x assets); out[t] uses Z up to t-1,
    # the last one is the forecast for the next day
    Qbar = np.corrcoef(Z.T)
    Q = Qbar.copy()
    out = np.empty((len(Z) + 1,) + Qbar.shape)
    for t in range(len(Z) + 1):
        d = np.sqrt(np.diag(Q))
        out[t] = Q / np.outer(d, d)
        if t < len(Z):
            Q = (1 - a - b) * Qbar + a * np.outer(Z[t], Z[t]) + b * Q
    return out


def dcc_fit(Z):
    # maximum likelihood of the correlation part of a DCC(1,1)
    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        R = dcc_filter(Z, a, b)[:-1]
        logdet = np.linalg.slogdet(R)[1]
        q = np.einsum("ti,ti->t", Z, np.linalg.solve(R, Z[..., None])[..., 0])
        return 0.5 * np.sum(logdet + q - (Z ** 2).sum(axis=1))
    return optimize.minimize(nll, [0.03, 0.95], method="Nelder-Mead").x


def t_copula_fit(X):
    # Student-t copula on the ranks of X (days x 2): rho from Kendall's tau, degrees of freedom by maximum likelihood
    U = stats.rankdata(X, axis=0) / (len(X) + 1)
    rho = np.sin(np.pi / 2 * stats.kendalltau(X[:, 0], X[:, 1])[0])
    C = np.array([[1, rho], [rho, 1]])

    def nll(nu):
        x = stats.t.ppf(U, nu)
        return -(stats.multivariate_t(shape=C, df=nu).logpdf(x).sum() - stats.t.logpdf(x, nu).sum())
    nu = optimize.minimize_scalar(nll, bounds=(2.1, 100), method="bounded").x
    return rho, nu


def es_contributions(S, w, alpha=0.99):
    # share of the historical ES brought by each asset: its average weighted loss on the worst portfolio days
    S, w = np.asarray(S), np.asarray(w)
    p = S @ w
    tail = S[p <= np.quantile(p, 1 - alpha)]
    c = -(tail * w).mean(axis=0)
    return c / c.sum()


def equal_risk_weights(S, alpha=0.99):
    # weights where every asset brings the same share of the historical ES (risk budgeting, Bruder & Roncalli):
    # minimise ES(w) - c * sum(log w). the log keeps all weights positive and the optimum has equal contributions,
    # even when an asset gains on the bad days (bonds, gold) and would have a negative contribution at equal weights
    S = np.asarray(S)
    n = S.shape[1]

    def es(w):
        p = S @ w
        return -p[p <= np.quantile(p, 1 - alpha)].mean()

    w0 = 1 / S.std(axis=0)
    w0 /= w0.sum()
    c = es(w0) / n
    res = optimize.minimize(lambda y: es(np.exp(y)) - c * y.sum(), np.log(w0), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 20000 * n})
    w = np.exp(res.x)
    return w / w.sum()


def rolling_equal_risk_weights(S, window=1000, alpha=0.99):
    # out of sample: on the first trading day of each year, equal-risk weights from the past `window` days only,
    # then kept for the whole year. returns the daily weights (days x assets), NaN before the first year
    import pandas as pd
    S = pd.DataFrame(S)
    W = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    first_days = [i for i in range(window, len(S)) if S.index[i].year != S.index[i - 1].year]
    for k, i in enumerate(first_days):
        j = first_days[k + 1] if k + 1 < len(first_days) else len(S)
        W.iloc[i:j] = equal_risk_weights(S.values[i - window:i], alpha)
    return W


def risk_parity_weights(Sigma):
    # equal contribution to the portfolio variance (Spinu 2013): minimise w'Sw / 2 - sum(log w) / n, then rescale
    n = len(Sigma)
    w0 = 1 / np.sqrt(np.diag(Sigma))
    res = optimize.minimize(lambda w: 0.5 * w @ Sigma @ w - np.log(w).sum() / n, w0 / w0.sum(),
                            jac=lambda w: Sigma @ w - 1 / (n * w), method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    return res.x / res.x.sum()


def monthly_risk_parity(R, method="dcc", window=1000, refit=250, cov_days=250):
    # risk-parity weights set on the first trading day of each month, kept for the month. R: daily log returns
    # method "sample": covariance of the last `cov_days` days. method "dcc": next-day covariance forecast
    # D R D from one GARCH(1,1) per asset and a DCC(1,1), refitted every `refit` days on the last `window` days
    import pandas as pd
    X = R.values
    T, n = X.shape
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    month_starts = [i for i in range(window, T) if R.index[i].month != R.index[i - 1].month]
    covs = {}
    if method == "sample":
        covs = {i: np.cov(X[i - cov_days:i].T) for i in month_starts}
    else:
        for start in range(window, T, refit):
            end = min(start + refit, T)
            past = X[start - window:start]
            mu = past.mean(axis=0)
            sig = np.empty((end - start + window, n))
            for k in range(n):
                sig[:, k] = np.sqrt(garch_filter(X[start - window:end, k] - mu[k], *garch_fit(past[:, k]))[:-1])
            a, b = dcc_fit((past - mu) / sig[:window])
            Rt = dcc_filter((X[start - window:end] - mu) / sig, a, b)
            for i in month_starts:
                if start <= i < end:
                    d = sig[i - start + window]
                    covs[i] = Rt[i - start + window] * np.outer(d, d)
    for k, i in enumerate(month_starts):
        j = month_starts[k + 1] if k + 1 < len(month_starts) else T
        W.iloc[i:j] = risk_parity_weights(covs[i])
    return W


def rebalanced_returns(S, W, cost=0.0):
    # daily returns of a portfolio that drifts with prices and goes back to the target weights W
    # on the first trading day of each month, paying `cost` per unit of value traded (0.001 = 10 bp)
    # S: daily simple returns, W: target weights (same index). returns net simple returns and turnover
    import pandas as pd
    start = W.dropna().index[0]
    S, W = S.loc[start:], W.loc[start:]
    w = W.iloc[0].values.copy()
    net, turn = np.empty(len(S)), np.zeros(len(S))
    for i in range(len(S)):
        if i > 0 and S.index[i].month != S.index[i - 1].month:
            target = W.iloc[i].values
            turn[i] = np.abs(target - w).sum()
            w = target.copy()
        r = S.iloc[i].values
        gross = w @ r
        net[i] = gross - cost * turn[i]
        w = w * (1 + r) / (1 + gross)   # drift until the next rebalancing
    return pd.Series(net, index=S.index), pd.Series(turn, index=S.index)


def backtest_weights(S, W, window=1000, alpha=0.99):
    # VaR / ES backtest of a portfolio whose weights change over time. for each period with constant target
    # weights, the past `window` days are rebuilt with those weights and every model of backtest() is
    # refitted on that history, then used for the period. S: daily simple returns, W: target weights
    import pandas as pd
    v = W.values
    first = int(np.argmax(~np.isnan(v).any(axis=1)))
    starts = [i for i in range(max(first, window), len(W)) if i == max(first, window) or not np.allclose(v[i], v[i - 1])]
    out = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(W)
        hist = np.log1p(S.iloc[i - window:j] @ v[i])
        out.append(backtest(hist, alpha=alpha, window=window, refit=j - i))
    return pd.concat(out)


def stress_correlation(C, group_a, group_b, rho):
    # set every correlation between the assets of group_a and group_b to rho, then go back to the nearest
    # valid correlation matrix (negative eigenvalues clipped, diagonal rescaled to 1)
    D = np.array(C, dtype=float)
    for i in group_a:
        for j in group_b:
            D[i, j] = D[j, i] = rho
    vals, vecs = np.linalg.eigh(D)
    D = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(D))
    return D / np.outer(d, d)


def portfolio_backtest(R, w, alpha=0.99, window=1000, refit=250, lam=0.94):
    # two portfolio models on a matrix of daily log returns R (days x assets) with weights w:
    # - ewma: RiskMetrics covariance (lambda 0.94) and a normal VaR/ES
    # - mfhs: one GARCH per asset, then the past vectors of standardised residuals are
    #   rescaled by today's volatilities and turned into portfolio returns (keeps the joint tails)
    # - dcc: same as mfhs, but the residuals are first decorrelated with their own DCC correlation
    #   and then given today's DCC correlation, so the dependence also changes every day
    import pandas as pd
    X, w = np.asarray(R), np.asarray(w)
    T, n = X.shape
    out = {k: np.full(T, np.nan) for k in ["var_ewma", "es_ewma", "var_mfhs", "es_mfhs", "var_dcc", "es_dcc"]}
    zq = stats.norm.ppf(1 - alpha)
    S = np.cov(X[:250].T)
    for t in range(T):
        if t >= window:
            sp = np.sqrt(w @ S @ w)
            out["var_ewma"][t], out["es_ewma"][t] = -sp * zq, sp * stats.norm.pdf(zq) / (1 - alpha)
        S = lam * S + (1 - lam) * np.outer(X[t], X[t])
    for start in range(window, T, refit):
        end = min(start + refit, T)
        past = X[start - window:start]
        mu = past.mean(axis=0)
        sig = np.empty((end - start + window, n))
        for i in range(n):
            sig[:, i] = np.sqrt(garch_filter(X[start - window:end, i] - mu[i], *garch_fit(past[:, i]))[:-1])
        Z = (past - mu) / sig[:window]
        a, b = dcc_fit(Z)
        Rt = dcc_filter((X[start - window:end] - mu) / sig, a, b)
        # remove each past day's own correlation (Cholesky), leaving uncorrelated shocks
        E = np.array([np.linalg.solve(np.linalg.cholesky(Rt[s]), Z[s]) for s in range(window)])
        for t in range(start, end):
            st = sig[t - start + window]
            port = np.log1p((np.exp(mu + Z * st) - 1) @ w)
            q = np.quantile(port, 1 - alpha)
            out["var_mfhs"][t], out["es_mfhs"][t] = -q, -port[port <= q].mean()
            Zt = E @ np.linalg.cholesky(Rt[t - start + window]).T   # give today's correlation
            port = np.log1p((np.exp(mu + Zt * st) - 1) @ w)
            q = np.quantile(port, 1 - alpha)
            out["var_dcc"][t], out["es_dcc"][t] = -q, -port[port <= q].mean()
    df = pd.DataFrame(out, index=R.index if hasattr(R, "index") else None)
    df.insert(0, "ret", np.log1p((np.exp(X) - 1) @ w))
    return df.iloc[window:]


def gjr_skt_paths(params, s2_next, mu, h=10, alpha=0.99, n=20_000, rng=None):
    # VaR / ES of the h-day log return, by simulating h days of GJR-GARCH with skewed-t shocks
    # s2_next: variance forecast for the first day. the vol moves along each path
    omega, a, g, b, eta, lam = params
    rng = rng or np.random.default_rng(0)
    s2 = np.full(n, s2_next)
    total = np.zeros(n)
    for _ in range(h):
        x = np.sqrt(s2) * skt_ppf(rng.random(n), eta, lam)
        total += mu + x
        s2 = omega + (a + g * (x < 0)) * x ** 2 + b * s2
    q = np.quantile(total, 1 - alpha)
    return -q, -total[total <= q].mean()


def backtest_10d(r, alpha=0.99, window=1000, refit=250, h=10, n_sims=20_000):
    # h-day VaR / ES on non-overlapping h-day periods after the first `window` days
    # sqrt rule: 1-day VaR scaled by sqrt(h). hist10: quantile of the past overlapping h-day returns.
    # mc: h-day paths of GJR-GARCH skewed-t
    import pandas as pd
    v = r.values
    rng = np.random.default_rng(0)
    rows = []
    for start in range(window, len(v) - h + 1, refit):
        past = v[start - window:start]
        mu = past.mean()
        p = gjr_skt_fit(past)
        end = min(start + refit, len(v) - h + 1)
        s2 = gjr_filter(v[start - window:end] - mu, *p[:4])
        one_var, one_es = skt_var_es(p[4], p[5], alpha)
        for t in range(start, end):
            if (t - window) % h:
                continue
            w = v[t - window:t]
            sums = np.convolve(w, np.ones(h), "valid")    # overlapping h-day returns of the window
            q10 = np.quantile(sums, 1 - alpha)
            nv, ne = var_es_normal(w, alpha)
            hv, he = var_es_hist(w, alpha)
            st = np.sqrt(s2[t - start + window])
            mv, me = gjr_skt_paths(p, s2[t - start + window], mu, h, alpha, n_sims, rng)
            rows.append({"date": r.index[t], "ret": v[t:t + h].sum(),
                         "var_normal": -h * mu + np.sqrt(h) * (nv + mu), "es_normal": -h * mu + np.sqrt(h) * (ne + mu),
                         "var_hist": np.sqrt(h) * hv, "es_hist": np.sqrt(h) * he,
                         "var_hist10": -q10, "es_hist10": -sums[sums <= q10].mean(),
                         "var_sqrt": -h * mu + np.sqrt(h) * st * one_var, "es_sqrt": -h * mu + np.sqrt(h) * st * one_es,
                         "var_mc": mv, "es_mc": me})
    return pd.DataFrame(rows).set_index("date")
