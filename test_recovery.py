"""
Parameter-recovery test.

If you simulate a season from known ratings and the fitter cannot recover
them, nothing downstream can be trusted. This test simulates several seasons
from fixed parameters, fits the model, and checks that the recovered attack
and defence ratings line up with the truth and that the home-advantage and
rho parameters land in the right region.

Run directly (`python tests/test_recovery.py`) or under pytest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DixonColesModel
from simulate import make_team_params, simulate_season


def _fit_on_simulated(n_seasons=4, seed=1):
    params = make_team_params(n_teams=20, seed=7)
    frames = []
    import pandas as pd

    for s in range(n_seasons):
        frames.append(simulate_season(params, seed=seed + s))
    df = pd.concat(frames, ignore_index=True)
    model = DixonColesModel().fit(df["HomeTeam"], df["AwayTeam"], df["FTHG"], df["FTAG"])
    return params, model


def test_parameter_recovery():
    params, model = _fit_on_simulated()
    order = model.teams
    idx = {t: k for k, t in enumerate(params["teams"])}
    true_attack = np.array([params["attack"][idx[t]] for t in order])
    true_defence = np.array([params["defence"][idx[t]] for t in order])

    attack_corr = np.corrcoef(true_attack, model.attack_)[0, 1]
    defence_corr = np.corrcoef(true_defence, model.defence_)[0, 1]

    assert attack_corr > 0.9, f"attack correlation too low: {attack_corr:.3f}"
    assert defence_corr > 0.9, f"defence correlation too low: {defence_corr:.3f}"
    assert abs(model.home_adv_ - params["home_adv"]) < 0.1, model.home_adv_
    assert model.rho_ < 0.05, model.rho_
    return attack_corr, defence_corr, model, params


if __name__ == "__main__":
    ac, dc, model, params = test_parameter_recovery()
    print("attack correlation (true vs fitted): %.3f" % ac)
    print("defence correlation (true vs fitted): %.3f" % dc)
    print("home advantage: true %.3f  fitted %.3f" % (params["home_adv"], model.home_adv_))
    print("rho:            true %.3f  fitted %.3f" % (params["rho"], model.rho_))
    print("\nTop 6 fitted attack ratings:")
    print(model.ratings().head(6).to_string(index=False))
    print("\nAll checks passed.")
