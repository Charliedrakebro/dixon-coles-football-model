"""
Fit the model on real Premier League data and report.

Pulls results and closing odds from football-data.co.uk, fits the Dixon-Coles
model on the full history, prints the current team ratings, then runs a
walk-forward backtest against the bookmaker's closing line and saves a
reliability plot.

Usage
-----
    python scripts/fit_and_report.py                 # Premier League, last 3 seasons
    python scripts/fit_and_report.py SP1 2223 2324 2425   # La Liga

League codes follow football-data.co.uk (E0, E1, SP1, I1, D1, F1).
Needs network access; if you are offline, run scripts/demo_offline.py instead.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_seasons
from model import DixonColesModel, time_decay_weights
from evaluate import (
    walk_forward_backtest, summarise_backtest, reliability_table, value_bet_backtest,
)


def main(argv):
    league = argv[0] if argv else "E0"
    seasons = tuple(argv[1:]) if len(argv) > 1 else ("2223", "2324", "2425")

    print(f"Downloading {league} seasons {seasons} from football-data.co.uk ...")
    df = load_seasons(league=league, seasons=seasons)
    print(f"Loaded {len(df)} matches.\n")

    # current ratings on the full history, with recency weighting
    ref = df["Date"].max()
    w = time_decay_weights(df["Date"], ref, half_life_days=180)
    model = DixonColesModel().fit(df["HomeTeam"], df["AwayTeam"], df["FTHG"], df["FTAG"], weights=w)
    print("Home advantage: %.3f   rho: %.3f\n" % (model.home_adv_, model.rho_))
    print("Team ratings (attack descending)")
    print(model.ratings().to_string(index=False))

    print("\nRunning walk-forward backtest against the closing line ...")
    bt = walk_forward_backtest(df, min_train_matches=200, half_life_days=180)
    s = summarise_backtest(bt)
    print(f"\nmatches scored : {s['n_matches']}")
    print(f"model RPS      : {s['model_rps']:.4f}")
    if "market_rps" in s:
        print(f"market RPS     : {s['market_rps']:.4f}")
        print(f"gap vs market  : {s['rps_gap_vs_market']:+.4f}")

    if "market_home" in bt.columns:
        vb = value_bet_backtest(bt, edge_threshold=0.03, kelly_fraction=0.25)
        print(f"\nvalue bets     : {vb['n_bets']}")
        print(f"flat-stake ROI : {vb['flat_roi']:+.3%}")
        print(f"quarter-Kelly  : {vb['kelly_final_bankroll']:.3f} (from 1.000)")

    rel = reliability_table(bt, bins=10)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.scatter(rel["mean_predicted"], rel["observed_rate"],
               s=rel["n"] / rel["n"].max() * 200, alpha=0.8, label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability plot ({league})")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    out = f"calibration_{league}.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved reliability plot to {os.path.relpath(out)}")


if __name__ == "__main__":
    main(sys.argv[1:])
