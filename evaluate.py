"""
Out-of-sample evaluation for the goals model.

Everything here is built to avoid look-ahead. The headline test is a
walk-forward backtest: at each round we refit on matches played strictly
before it, predict the upcoming fixtures, and score those predictions against
what actually happened and against the bookmaker's closing prices. Beating the
closing line is the real bar, so the market is always carried through as the
benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import DixonColesModel, time_decay_weights
from odds import implied_proportional


OUTCOMES = ("home", "draw", "away")


def ranked_probability_score(probs: np.ndarray, outcome_idx: int) -> float:
    """RPS for a single ordered 3-way forecast (lower is better).

    RPS respects the ordering home > draw > away, so being wrong by a little
    (predicting a draw when it was a home win) is penalised less than being
    wrong by a lot. It is the standard scoring rule for 1X2 forecasts.
    """
    probs = np.asarray(probs, dtype=float)
    obs = np.zeros(3)
    obs[outcome_idx] = 1.0
    cum_p = np.cumsum(probs)
    cum_o = np.cumsum(obs)
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(probs) - 1))


def log_loss(probs: np.ndarray, outcome_idx: int) -> float:
    p = np.clip(np.asarray(probs, dtype=float)[outcome_idx], 1e-15, 1.0)
    return float(-np.log(p))


def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_matches: int = 150,
    half_life_days: float | None = 180.0,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Refit round by round and score each prediction out of sample.

    `df` needs columns Date, HomeTeam, AwayTeam, FTHG, FTAG and, if you want the
    market benchmark, OddsH, OddsD, OddsA. Returns one row per predicted match
    with the model and market probabilities and their scores.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    has_odds = {"OddsH", "OddsD", "OddsA"}.issubset(df.columns)
    records = []

    # predict in date order, refitting on everything strictly earlier.
    unique_dates = df["Date"].drop_duplicates().sort_values().to_list()
    for d in unique_dates:
        train = df[df["Date"] < d]
        if len(train) < min_train_matches:
            continue
        today = df[df["Date"] == d]

        model = DixonColesModel(max_goals=max_goals)
        weights = (
            time_decay_weights(train["Date"], d, half_life_days)
            if half_life_days
            else None
        )
        try:
            model.fit(
                train["HomeTeam"], train["AwayTeam"],
                train["FTHG"], train["FTAG"], weights=weights,
            )
        except Exception:
            continue

        for _, row in today.iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            if h not in model.team_index or a not in model.team_index:
                continue  # newly promoted team with no history yet
            pred = model.predict_1x2(h, a)
            mp = np.array([pred["home"], pred["draw"], pred["away"]])
            oi = outcome_index(row["FTHG"], row["FTAG"])

            rec = {
                "Date": d, "HomeTeam": h, "AwayTeam": a,
                "outcome": OUTCOMES[oi],
                "model_home": mp[0], "model_draw": mp[1], "model_away": mp[2],
                "model_rps": ranked_probability_score(mp, oi),
                "model_logloss": log_loss(mp, oi),
            }
            if has_odds:
                book = implied_proportional(
                    [row["OddsH"], row["OddsD"], row["OddsA"]]
                )
                rec.update(
                    market_home=book[0], market_draw=book[1], market_away=book[2],
                    market_rps=ranked_probability_score(book, oi),
                    market_logloss=log_loss(book, oi),
                    edge_home=mp[0] - book[0],
                    edge_draw=mp[1] - book[1],
                    edge_away=mp[2] - book[2],
                )
            records.append(rec)

    return pd.DataFrame(records)


def summarise_backtest(bt: pd.DataFrame) -> dict:
    """Aggregate mean scores for model and (if present) market."""
    out = {
        "n_matches": len(bt),
        "model_rps": bt["model_rps"].mean(),
        "model_logloss": bt["model_logloss"].mean(),
    }
    if "market_rps" in bt.columns:
        out["market_rps"] = bt["market_rps"].mean()
        out["market_logloss"] = bt["market_logloss"].mean()
        out["rps_gap_vs_market"] = out["model_rps"] - out["market_rps"]
    return out


def reliability_table(bt: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Bin model probabilities and compare them to realised frequencies.

    A well-calibrated model has mean predicted probability close to the
    observed rate in every bin. Stacks all three outcomes together.
    """
    preds, hits = [], []
    for outcome, col in zip(OUTCOMES, ["model_home", "model_draw", "model_away"]):
        preds.append(bt[col].to_numpy())
        hits.append((bt["outcome"] == outcome).to_numpy().astype(float))
    p = np.concatenate(preds)
    y = np.concatenate(hits)

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def value_bet_backtest(
    bt: pd.DataFrame, edge_threshold: float = 0.02, kelly_fraction: float = 0.25,
) -> dict:
    """Flat-stake and fractional-Kelly returns on model value bets.

    A value bet is any outcome where the model probability exceeds the market
    implied probability by more than `edge_threshold`. This is deliberately
    naive (it bets into the closing line, the hardest price to beat) and is
    here to show the plumbing and to be honest about how thin real edges are,
    not to promise a profit.
    """
    if "market_home" not in bt.columns:
        raise ValueError("Backtest has no odds columns; cannot value-bet.")

    flat_pnl, kelly_pnl, kelly_bankroll = 0.0, 0.0, 1.0
    n_bets = 0
    outcome_cols = {
        "home": ("model_home", "market_home"),
        "draw": ("model_draw", "market_draw"),
        "away": ("model_away", "market_away"),
    }
    for _, row in bt.iterrows():
        for outcome, (mcol, bcol) in outcome_cols.items():
            p_model = row[mcol]
            p_market = row[bcol]
            if p_model - p_market <= edge_threshold:
                continue
            odds = 1.0 / p_market  # de-vigged fair price implied by the market
            won = row["outcome"] == outcome
            n_bets += 1
            # flat stake of 1 unit
            flat_pnl += (odds - 1.0) if won else -1.0
            # fractional Kelly on the de-vigged price
            b = odds - 1.0
            f = max(0.0, (p_model * b - (1.0 - p_model)) / b) * kelly_fraction
            stake = kelly_bankroll * f
            kelly_bankroll += stake * ((odds - 1.0) if won else -1.0)

    return {
        "n_bets": n_bets,
        "flat_pnl_units": flat_pnl,
        "flat_roi": flat_pnl / n_bets if n_bets else float("nan"),
        "kelly_final_bankroll": kelly_bankroll,
    }
