"""
End-to-end demo on simulated data (no network needed).

Runs the whole pipeline: simulate a few seasons, fit the model, run a
walk-forward backtest against the (margined) market, print the score
comparison and calibration, run a naive value-bet backtest, and save a
reliability plot to calibration_demo.png.

Use this to sanity-check the code anywhere. For the real thing, run
scripts/fit_and_report.py, which pulls actual Premier League data and odds.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate import make_team_params, simulate_season
from evaluate import (
    walk_forward_backtest, summarise_backtest, reliability_table, value_bet_backtest,
)


def main():
    params = make_team_params(n_teams=20, seed=3)
    df = pd.concat(
        [simulate_season(params, seed=s) for s in range(3)], ignore_index=True
    ).sort_values("Date").reset_index(drop=True)
    print(f"Simulated {len(df)} matches across 3 seasons, {len(params['teams'])} teams.\n")

    bt = walk_forward_backtest(df, min_train_matches=200, half_life_days=240)
    summary = summarise_backtest(bt)
    print("Walk-forward backtest (out of sample)")
    print("-------------------------------------")
    print(f"matches scored     : {summary['n_matches']}")
    print(f"model RPS          : {summary['model_rps']:.4f}")
    if "market_rps" in summary:
        print(f"market RPS         : {summary['market_rps']:.4f}")
        print(f"gap vs market      : {summary['rps_gap_vs_market']:+.4f}  "
              f"({'model better' if summary['rps_gap_vs_market'] < 0 else 'market better'})")
    print(f"model log loss     : {summary['model_logloss']:.4f}")

    print("\nCalibration (model probabilities vs realised rate)")
    print("--------------------------------------------------")
    rel = reliability_table(bt, bins=10)
    print(rel.to_string(index=False))

    vb = value_bet_backtest(bt, edge_threshold=0.02, kelly_fraction=0.25)
    print("\nNaive value-bet backtest (into the fair de-vigged price)")
    print("--------------------------------------------------------")
    print(f"bets placed        : {vb['n_bets']}")
    print(f"flat-stake ROI     : {vb['flat_roi']:+.3%}")
    print(f"quarter-Kelly bank : {vb['kelly_final_bankroll']:.3f} (from 1.000)")

    # reliability plot

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.scatter(rel["mean_predicted"], rel["observed_rate"], s=rel["n"] / rel["n"].max() * 200,
               alpha=0.8, label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability plot (simulated demo)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    out = "calibration_demo.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved reliability plot to {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
