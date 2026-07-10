"""
Dixon-Coles bivariate goals model.

Home and away goals are modelled as Poisson counts driven by team-specific
attack and defence ratings plus a home-advantage term, with the Dixon-Coles
low-score dependence correction and optional exponential time-decay weighting.

Reference: Dixon, M. and Coles, S. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market", Applied Statistics.

Parametrisation
---------------
For a match between home team i and away team j:

    log(lambda) = home_adv + attack_i - defence_j        # expected home goals
    log(mu)     =           attack_j - defence_i         # expected away goals

Higher `attack` means more goals scored; higher `defence` means fewer goals
conceded. Ratings are identified by a sum-to-zero constraint on both the
attack and defence vectors (the nth team is minus the sum of the rest), which
removes the attack/defence translation ridge and stabilises the optimiser.

The joint probability of a scoreline (x, y) is

    P(x, y) = tau(x, y; lambda, mu, rho) * Poisson(x; lambda) * Poisson(y; mu)

where tau applies the Dixon-Coles correction to the four lowest scorelines and
equals 1 elsewhere. rho captures the empirical dependence between low scores
(typically small and negative).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def _log_poisson_pmf(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """log P(K = k) for a Poisson with the given rate, computed directly so we
    avoid repeated scipy.stats overhead inside the optimiser."""
    rate = np.maximum(rate, 1e-10)
    return k * np.log(rate) - rate - gammaln(k + 1.0)


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction, vectorised over matches."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles goals model.

    Parameters
    ----------
    max_goals : int
        Upper goal count used when building scoreline probability matrices.
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.team_index: dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defence_: np.ndarray | None = None
        self.home_adv_: float | None = None
        self.rho_: float | None = None
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _pack_teams(self, home, away):
        teams = sorted(set(home) | set(away))
        self.teams = teams
        self.team_index = {t: k for k, t in enumerate(teams)}
        hi = np.array([self.team_index[t] for t in home])
        ai = np.array([self.team_index[t] for t in away])
        return hi, ai

    def _unpack_params(self, params, n):
        # free attack/defence have length n-1; nth is minus their sum.
        a_free = params[: n - 1]
        d_free = params[n - 1 : 2 * (n - 1)]
        home_adv = params[2 * (n - 1)]
        rho = params[2 * (n - 1) + 1]
        attack = np.concatenate([a_free, [-a_free.sum()]])
        defence = np.concatenate([d_free, [-d_free.sum()]])
        return attack, defence, home_adv, rho

    def _neg_log_likelihood(self, params, n, hi, ai, hg, ag, weights):
        attack, defence, home_adv, rho = self._unpack_params(params, n)
        log_lam = home_adv + attack[hi] - defence[ai]
        log_mu = attack[ai] - defence[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        ll = _log_poisson_pmf(hg, lam) + _log_poisson_pmf(ag, mu)
        tau = _tau(hg, ag, lam, mu, rho)
        # floor tau to keep the log finite if the optimiser probes an
        # invalid corner of the rho space.
        ll = ll + np.log(np.maximum(tau, 1e-10))
        return -np.sum(weights * ll)

    def fit(
        self,
        home_team,
        away_team,
        home_goals,
        away_goals,
        weights=None,
        verbose: bool = False,
    ) -> "DixonColesModel":
        """Fit the model by weighted maximum likelihood.

        `weights` lets you pass exponential time-decay weights (see
        `time_decay_weights`); if omitted every match is weighted equally.
        """
        home_team = np.asarray(home_team)
        away_team = np.asarray(away_team)
        hg = np.asarray(home_goals, dtype=float)
        ag = np.asarray(away_goals, dtype=float)
        hi, ai = self._pack_teams(home_team, away_team)
        n = len(self.teams)
        if weights is None:
            weights = np.ones(len(hg))
        weights = np.asarray(weights, dtype=float)

        # sensible starting point: zero ratings, mild home advantage, small rho
        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n - 1), [0.25], [-0.05]])
        bounds = (
            [(-3, 3)] * (n - 1)
            + [(-3, 3)] * (n - 1)
            + [(-1, 2)]           # home advantage
            + [(-0.2, 0.2)]       # rho
        )
        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(n, hi, ai, hg, ag, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "disp": verbose},
        )
        attack, defence, home_adv, rho = self._unpack_params(res.x, n)
        self.attack_ = attack
        self.defence_ = defence
        self.home_adv_ = float(home_adv)
        self.rho_ = float(rho)
        self.fitted_ = True
        self._opt_result = res
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _rates(self, home_team: str, away_team: str):
        i = self.team_index[home_team]
        j = self.team_index[away_team]
        lam = np.exp(self.home_adv_ + self.attack_[i] - self.defence_[j])
        mu = np.exp(self.attack_[j] - self.defence_[i])
        return lam, mu

    def scoreline_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return an (max_goals+1) x (max_goals+1) matrix of scoreline
        probabilities, rows indexed by home goals and columns by away goals."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")
        lam, mu = self._rates(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        home_pmf = np.exp(_log_poisson_pmf(goals, np.full_like(goals, lam, dtype=float)))
        away_pmf = np.exp(_log_poisson_pmf(goals, np.full_like(goals, mu, dtype=float)))
        matrix = np.outer(home_pmf, away_pmf)

        # Dixon-Coles correction on the four lowest scorelines
        rho = self.rho_
        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho
        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()
        return matrix

    def predict_1x2(self, home_team: str, away_team: str) -> dict:
        """Return {'home', 'draw', 'away'} outcome probabilities."""
        m = self.scoreline_matrix(home_team, away_team)
        home = np.tril(m, -1).sum()   # home goals > away goals
        away = np.triu(m, 1).sum()    # away goals > home goals
        draw = np.trace(m)
        return {"home": float(home), "draw": float(draw), "away": float(away)}

    def predict_over_under(self, home_team: str, away_team: str, line: float = 2.5) -> dict:
        """Return over/under probabilities for a total-goals line."""
        m = self.scoreline_matrix(home_team, away_team)
        totals = np.add.outer(
            np.arange(self.max_goals + 1), np.arange(self.max_goals + 1)
        )
        over = m[totals > line].sum()
        return {"over": float(over), "under": float(1.0 - over)}

    # ------------------------------------------------------------------ #
    # Ratings table
    # ------------------------------------------------------------------ #
    def ratings(self) -> "pd.DataFrame":
        import pandas as pd

        return (
            pd.DataFrame(
                {"team": self.teams, "attack": self.attack_, "defence": self.defence_}
            )
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )


def time_decay_weights(match_dates, reference_date, half_life_days: float = 180.0):
    """Exponential decay weights for matches played before a reference date.

    A match played `half_life_days` before the reference date gets weight 0.5,
    so recent form counts for more. Pass the result to `DixonColesModel.fit`.
    """
    import pandas as pd

    match_dates = pd.to_datetime(pd.Series(match_dates)).reset_index(drop=True)
    reference_date = pd.to_datetime(reference_date)
    days_before = (reference_date - match_dates).dt.total_seconds() / 86400.0
    days_before = days_before.clip(lower=0.0)
    xi = np.log(2.0) / half_life_days
    return np.exp(-xi * days_before.to_numpy())
