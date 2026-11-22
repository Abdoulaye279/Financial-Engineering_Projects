# European option pricing: Black-Scholes, Monte Carlo and CRR binomial tree
import numpy as np
from scipy import stats


def black_scholes(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def simulate_terminal(S, T, r, sigma, n, seed=0):
    # final price of a geometric Brownian motion under the risk-neutral measure (antithetic draws)
    z = np.random.default_rng(seed).standard_normal(n // 2)
    z = np.concatenate([z, -z])
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def monte_carlo(S, K, T, r, sigma, kind="call", n=1_000_000, seed=0):
    ST = simulate_terminal(S, T, r, sigma, n, seed)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


def binomial_crr(S, K, T, r, sigma, kind="call", steps=2000):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    ST = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    v = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    for _ in range(steps):
        v = np.exp(-r * dt) * (p * v[:-1] + (1 - p) * v[1:])
    return v[0]


def implied_vol(price, S, K, T, r, kind="call"):
    # vol to put in BS to get back the given price
    from scipy.optimize import brentq
    return brentq(lambda v: black_scholes(S, K, T, r, v, kind) - price, 1e-4, 5.0)
