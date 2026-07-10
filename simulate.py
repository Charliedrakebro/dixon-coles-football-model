"""
Generate synthetic seasons from known team parameters.

This has two uses: it lets the test suite check that the fitter recovers the
parameters it was given (the strongest evidence an implementation is correct),
and it lets the demo scripts run with no network access. Matches are drawn
from the same Dixon-Coles data-generating process the model assumes, and
bookmaker-style odds are produced by adding a margin to the true probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_team_params(n_teams: int = 20, seed: int = 0):
    """Draw a plausible set of attack/defence ratings, home advantage and rho."""
    rng = np.random.default_rng(seed)
    attack = rng.normal(0.0, 0.35, n_teams)
    defence = rng.normal(0.0, 0.35, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    return {
        "teams": teams,
        "attack": attack,
        "defence": defence,
        "home_adv": 0.26,
        "rho": -0.04,
    }


def _tau_scalar(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _score_probs(lam, mu, rho, max_goals=12):
    from scipy.stats import poisson

    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def simulate_season(params: dict, seed: int = 0, margin: float = 0.05,
                    start_date: str = "2023-08-01") -> pd.DataFrame:
    """Simulate a double round-robin season and attach margined 1X2 odds.

    Returns a frame with the same columns the real data loader produces:
    Date, HomeTeam, AwayTeam, FTHG, FTAG, OddsH, OddsD, OddsA.
    """
    rng = np.random.default_rng(seed)
    teams = params["teams"]
    n = len(teams)
    idx = {t: k for k, t in enumerate(teams)}
    rows = []
    date = pd.Timestamp(start_date)
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    rng.shuffle(fixtures)

    for gw, (h, a) in enumerate(fixtures):
        i, j = idx[h], idx[a]
        lam = np.exp(params["home_adv"] + params["attack"][i] - params["defence"][j])
        mu = np.exp(params["attack"][j] - params["defence"][i])
        probs = _score_probs(lam, mu, params["rho"])
        flat = probs.ravel()
        draw = rng.choice(len(flat), p=flat)
        hg, ag = divmod(draw, probs.shape[1])

        home_p = np.tril(probs, -1).sum()
        away_p = np.triu(probs, 1).sum()
        draw_p = np.trace(probs)
        true = np.array([home_p, draw_p, away_p])
        # apply a multiplicative margin then convert to decimal odds
        book = true * (1.0 + margin)
        odds = 1.0 / book
        rows.append(
            {
                "Date": date + pd.Timedelta(days=gw // (n // 2)),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": int(hg),
                "FTAG": int(ag),
                "OddsH": round(float(odds[0]), 3),
                "OddsD": round(float(odds[1]), 3),
                "OddsA": round(float(odds[2]), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
