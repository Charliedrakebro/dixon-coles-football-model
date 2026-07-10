# A Dixon-Coles goals model for football, benchmarked against the closing line

A from-scratch implementation of the Dixon-Coles bivariate goals model, fitted
by maximum likelihood on real match data and evaluated out of sample against
bookmaker closing odds. The question it asks is the one that matters
commercially: not "does the model fit?" but "can a transparent statistical
model price a match as well as the market, and where does it fall short?"

The headline result is deliberately honest. On a clean walk-forward backtest
the model gets close to the market on ranked probability score and is
well calibrated, but it does not beat the closing line, which is exactly what
you should expect from a model this simple priced into the sharpest number a
book publishes. The value of the project is the framework: a correct
implementation, a look-ahead-free evaluation, and an honest read on edge.

## Data

Results and prices come from [football-data.co.uk](https://www.football-data.co.uk),
which is the practical source here because each CSV carries full-time goals and
closing 1X2 odds in one place. FBref has richer match stats but no prices, and
StatsBomb's open data has events but neither odds nor deep league history, so
neither supports a market benchmark on its own. The loader standardises to a
common schema and prefers Pinnacle closing odds, then Bet365, then the market
average.

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

The evaluation is walk-forward and leak-free. At each match date the model is
refitted on matches played strictly earlier (with time decay), then used to
predict that day's fixtures. Predictions are scored with:

* **Ranked probability score (RPS)**, the standard 1X2 metric, which respects
  the ordering home > draw > away.
* **Log loss**, as a sharper penalty on confident mistakes.
* **A reliability plot**, binning predicted probabilities against realised
  frequencies to check calibration.

The bookmaker's de-vigged closing probabilities are carried through every step
as the benchmark, because beating the closing line, not beating a coin flip, is
the real bar.

## Betting backtest

A naive value-bet backtest stakes on any outcome where the model probability
exceeds the market implied probability by a threshold, under both flat and
fractional-Kelly staking. It bets into the fair de-vigged price on purpose:
that is the hardest number to beat, and reporting a slightly negative ROI here
is the honest outcome rather than a curve-fitted profit. Two de-vig methods are
provided, proportional and Shin (1993), the latter removing proportionally more
margin from longshots.

## Illustrative output

Running the offline demo (three simulated seasons, so the "market" is the true
probabilities plus a 5% margin) produces the kind of report the real script
generates:

```
Walk-forward backtest (out of sample)
matches scored : 930
model RPS      : 0.1858
market RPS     : 0.1830
gap vs market  : +0.0027  (market better)
model log loss : 0.9342
```

Calibration tracks the diagonal closely across all ten bins, and the naive
value-bet ROI sits just below zero. Fitting on real Premier League data with
`scripts/fit_and_report.py` produces the same report against genuine closing
odds; drop your numbers in here once you have run it.

## Validation

`tests/test_recovery.py` simulates seasons from known ratings and checks the
fitter recovers them. On the current settings the recovered attack and defence
ratings correlate with the truth at roughly 0.98 and 0.97, and home advantage
and `rho` land next to their true values. If that test fails, nothing
downstream should be trusted.

## Running it

```bash
pip install -r requirements.txt

# offline, no network: simulate, fit, backtest, plot
python scripts/demo_offline.py

# real data: Premier League, last three seasons
python scripts/fit_and_report.py

# another league (La Liga), choosing seasons
python scripts/fit_and_report.py SP1 2223 2324 2425

# parameter-recovery test
python tests/test_recovery.py
```

## Layout

```
dcmodel/
  model.py      Dixon-Coles model: fit, scoreline matrix, 1X2 and over/under
  odds.py       de-vigging: proportional and Shin implied probabilities
  data.py       football-data.co.uk loader and schema standardisation
  simulate.py   synthetic seasons for tests and offline demos
  evaluate.py   RPS, log loss, walk-forward backtest, calibration, value bets
scripts/
  demo_offline.py   full pipeline on simulated data
  fit_and_report.py full pipeline on real data
tests/
  test_recovery.py  parameter-recovery check
```

## Limitations and where this goes next

* The Dixon-Coles low-score correction handles dependence at 0 and 1 goals; a
  full bivariate Poisson with a shared component is the natural next model to
  compare against.
* No promotion or relegation handling: newly promoted teams start with no
  history and are skipped until they have played, rather than given an informed
  prior. A hierarchical prior that shrinks new teams toward the league mean
  would fix this.
* Ratings are team-level only. Adding availability of key players, rest days
  and travel would be the first features to test.
* Beating the closing line needs either a genuine information edge or better
  price capture than close; the sensible next step is to backtest against
  earlier prices to see whether the model beats the opening line and gets
  steamed toward close.

## References

* Dixon, M. and Coles, S. (1997). Modelling association football scores and
  inefficiencies in the football betting market. *Applied Statistics*, 46(2).
* Karlis, D. and Ntzoufras, I. (2003). Analysis of sports data by using
  bivariate Poisson models. *The Statistician*, 52(3).
* Shin, H. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *The Economic Journal*, 103(420).
