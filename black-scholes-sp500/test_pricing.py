from pricing import implied_vol, black_scholes, monte_carlo, binomial_crr

P = dict(S=100.0, T=1.0, r=0.03, sigma=0.2)


def test_known_value():  # known value: S=K=100, T=1, r=0, sigma=0.2 -> 7.9656
    assert abs(black_scholes(100, 100, 1, 0, 0.2) - 7.9656) < 1e-4


def test_matches_references():
    for K in (80, 100, 120):
        for kind in ("call", "put"):
            bs = black_scholes(K=K, kind=kind, **P)
            mc, se = monte_carlo(K=K, kind=kind, n=400_000, **P)
            assert abs(bs - mc) < 4 * se
            assert abs(bs - binomial_crr(K=K, kind=kind, **P)) < 5e-3


def test_implied_vol_roundtrip():
    for K, kind in ((80, "put"), (100, "call"), (120, "call")):
        price = black_scholes(K=K, kind=kind, **P)
        assert abs(implied_vol(price, K=K, kind=kind, S=P["S"], T=P["T"], r=P["r"]) - P["sigma"]) < 1e-8
