# A Dixon-Coles goals model for football, benchmarked against the market

A from-scratch implementation of the Dixon-Coles bivariate goals model, fitted
by maximum likelihood on real Premier League data and evaluated out of sample
against bookmaker odds. The question it asks is the one that matters
commercially: not "does the model fit?" but "can a transparent statistical
model price a match as well as the market, and where does it fall short?"

The headline result is honest. Across four seasons of real Premier League
matches, on a clean weekly walk-forward backtest, the model lands within about
4% of the market on ranked probability score and is well calibrated through the
middle of the probability range, but it does not beat the market, and naive
betting into it loses money. That is exactly what a model this simple should do
against an efficient market. The value of the project is the framework: a
correct implementation, a look-ahead-free evaluation, and an honest read on
edge.

## Data

Results and odds are real, from [football-data.co.uk](https://www.football-data.co.uk).
The repo pulls them through a public GitHub mirror that aggregates that source,
[xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025),
so it runs anywhere without depending on football-data.co.uk being reachable.
The default run uses the English Premier League (division code `E0`), seasons
2021/22 to 2024/25, which is 1,520 matches with full-time goals and market 1X2
odds.

One honest caveat on the benchmark: these are the market-average (consensus)
odds across bookmakers, not a single sharp book's closing price. The average is
a slightly softer benchmark than, say, Pinnacle's close, so the gap reported
below would likely widen against the sharpest available line. FBref has richer
match stats but no prices, and StatsBomb's open data has events but neither odds
nor deep history, so neither supports a market benchmark on its own.

## Model

Home and away goals are Poisson counts driven by team ratings and a home-field
term. For home team *i* against away team *j*:

```
log(lambda) = home_adv + attack_i - defence_j      (expected home goals)
log(mu)     =           attack_j - defence_i        (expected away goals)
```

Higher `attack` means more goals scored, higher `defence` means fewer conceded.
Ratings are identified by a sum-to-zero constraint on the attack and defence
vectors, which removes the translation ridge between them.

Independent Poisson margins understate the number of draws and low-scoring
games, so the Dixon-Coles correction adjusts the four lowest scorelines:

```
P(x, y) = tau(x, y; lambda, mu, rho) * Poisson(x; lambda) * Poisson(y; mu)
```

with `tau` equal to one everywhere except (0,0), (0,1), (1,0) and (1,1), where
it is governed by a dependence parameter `rho` (small and negative in
practice). Recent matches are weighted more heavily through an exponential
time-decay term with a configurable half-life, so form is allowed to drift.

Estimation is weighted maximum likelihood over all parameters jointly
(`attack` and `defence` per team, home advantage and `rho`) via L-BFGS-B.

## Out-of-sample evaluation

The evaluation is walk-forward and leak-free. The model is refitted weekly on
matches played strictly earlier (with time decay), then used to predict the
upcoming fixtures. Because each match is scored from a model trained only on
prior games, there is no look-ahead. Predictions are scored with ranked
probability score (RPS, the standard 1X2 metric, which respects the ordering
home > draw > away), log loss, and a reliability plot binning predicted
probabilities against realised frequencies. The market's de-vigged probabilities
are carried through as the benchmark at every step.

## Results (Premier League, 2021/22 to 2024/25)

Fitting on the full history with recency weighting, home advantage comes out at
0.13 (notably below the ~0.25 of older eras, consistent with the well-documented
post-pandemic decline), and the low-score dependence `rho` is close to zero. The
strongest recency-weighted attacks at the end of 2024/25 are Liverpool, Man
City, Newcastle and Arsenal, with Arsenal the best defence, which passes the
eyeball test.

The weekly walk-forward backtest scored 1,213 out-of-sample matches:

```
model RPS      : 0.2005
market RPS     : 0.1929
gap vs market  : +0.0076   (market better)
model log loss : 0.9735
market log loss: 0.9462
```

So the model is within roughly 4% of the market on RPS. Calibration is good
through the middle of the range (predicted probabilities of 0.1 to 0.6 match
realised frequencies closely) and drifts to mild overconfidence on heavy
favourites, where samples are thin. See `calibration_E0.png`.

A naive value-bet backtest, staking whenever the model's probability beat the
market's implied probability by more than three points, placed 977 bets and
returned about -4.4% on flat stakes; a quarter-Kelly bankroll fell from 1.00 to
roughly 0.21. Betting into the consensus price with a model that is marginally
worse than the market loses, as it should. Reporting that rather than a
curve-fitted profit is the point. Two de-vig methods are provided, proportional
and Shin (1993), the latter removing proportionally more margin from longshots.

## Validation

`test_recovery.py` simulates seasons from known ratings and checks the fitter
recovers them. On the current settings the recovered attack and defence ratings
correlate with the truth at roughly 0.98 and 0.97, and home advantage and `rho`
land next to their true values. If that test fails, nothing downstream should be
trusted.

## Running it

```bash
pip install -r requirements.txt

# real Premier League data: fit, weekly walk-forward backtest, plot
# (first run downloads and caches the data file, so it takes a minute)
python fit_and_report.py

# another league (La Liga) from a chosen start date
python fit_and_report.py SP1 2020-08-01

# offline demo on simulated data (no network needed)
python demo_offline.py

# parameter-recovery test
python test_recovery.py
```

## Layout

```
model.py            Dixon-Coles model: fit, scoreline matrix, 1X2 and over/under
odds.py             de-vigging: proportional and Shin implied probabilities
data.py             real-data loaders: GitHub mirror and football-data.co.uk
simulate.py         synthetic seasons for tests and offline demos
evaluate.py         RPS, log loss, walk-forward backtest, calibration, value bets
fit_and_report.py   full pipeline on real data (default: Premier League)
demo_offline.py     full pipeline on simulated data (no network)
test_recovery.py    parameter-recovery check
```

## Limitations and where this goes next

- The benchmark is the market average, not the sharpest closing line; the honest
  next step is to backtest against a single sharp book's close, where the gap
  will widen and the real difficulty shows.
- The Dixon-Coles low-score correction handles dependence at 0 and 1 goals; a
  full bivariate Poisson with a shared component is the natural model to compare
  against.
- No promotion or relegation handling: teams with little recent history (for
  example a club relegated early in the window) are poorly identified and their
  ratings can drift to the parameter bounds. A hierarchical prior shrinking such
  teams toward the league mean would fix this.
- Ratings are team-level only. Adding key-player availability, rest days and
  travel would be the first features to test.

## References

- Dixon, M. and Coles, S. (1997). Modelling association football scores and
  inefficiencies in the football betting market. *Applied Statistics*, 46(2).
- Karlis, D. and Ntzoufras, I. (2003). Analysis of sports data by using
  bivariate Poisson models. *The Statistician*, 52(3).
- Shin, H. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *The Economic Journal*, 103(420).

## Data attribution

Match results and odds are sourced from football-data.co.uk via the
[xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
mirror.
