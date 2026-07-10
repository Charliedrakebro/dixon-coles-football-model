"""
Fit the model on real Premier League data and report.

By default this pulls real results and market 1X2 odds from the GitHub mirror
of football-data.co.uk (see data.py), fits the Dixon-Coles model on the full
history, prints the current team ratings, then runs a weekly walk-forward
backtest against the market and saves a reliability plot.

Usage
-----
    python fit_and_report.py                 # Premier League (E0), since 2021-08
    python fit_and_report.py SP1 2020-08-01  # La Liga from a chosen start date

League codes follow football-data.co.uk (E0, E1, SP1, I1, D1, F1). The first
run downloads and caches the mirror (a large file), so it takes a minute.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_github_mirror
from model import DixonColesModel, time_decay_weights
from evaluate import (
    walk_forward_backtest, summarise_backtest, reliability_table, value_bet_backtest,
)


def main(argv):
    league = argv[0] if argv else "E0"
    start = argv[1] if len(argv) > 1 else "2021-08-01"

    print(f"Loading real {league} data (since {start}) from the GitHub mirror ...")
    df = load_github_mirror(league=league, start=start)
    print(f"Loaded {len(df)} matches with results and market odds.\n")

    # current ratings on the full history, with recency weighting
    ref = df["Date"].max()
    w = time_decay_weights(df["Date"], ref, half_life_days=180)
    model = DixonColesModel().fit(df["HomeTeam"], df["AwayTeam"], df["FTHG"], df["FTAG"], weights=w)
    print("Home advantage: %.3f   rho: %.3f\n" % (model.home_adv_, model.rho_))
    print("Team ratings (attack descending)")
    print(model.ratings().to_string(index=False))

    print("\nRunning weekly walk-forward backtest against the market ...")
    bt = walk_forward_backtest(df, min_train_matches=300, half_life_days=180, refit_every_days=7)
    s = summarise_backtest(bt)
    print(f"\nmatches scored : {s['n_matches']}")
    print(f"model RPS      : {s['model_rps']:.4f}")
    if "market_rps" in s:
        print(f"market RPS     : {s['market_rps']:.4f}")
        print(f"gap vs market  : {s['rps_gap_vs_market']:+.4f}")
        print(f"model log loss : {s['model_logloss']:.4f}")
        print(f"market logloss : {s['market_logloss']:.4f}")

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
    print(f"\nSaved reliability plot to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
