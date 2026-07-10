"""
Convert bookmaker decimal odds into implied probabilities.

Raw reciprocals of decimal odds sum to more than one; the excess is the
bookmaker's margin (the "overround" or "vig"). To recover a probability
estimate we have to strip that margin out. Two methods are provided:

* `implied_proportional` divides each reciprocal by the overround. Simple and
  transparent, but it spreads the margin evenly and so slightly overstates
  the probability of longshots (the favourite-longshot bias).
* `implied_shin` uses Shin's (1993) model, which assumes the margin reflects a
  proportion of insider trading and removes proportionally more from longshots.
  It is the more defensible closing-line estimate.
"""

from __future__ import annotations

import numpy as np


def implied_proportional(odds: np.ndarray) -> np.ndarray:
    """Normalise reciprocal odds so they sum to one.

    `odds` is an array of decimal odds for a single market (e.g. the three
    1X2 prices). Returns probabilities in the same order.
    """
    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds
    return raw / raw.sum()


def overround(odds: np.ndarray) -> float:
    """The bookmaker margin: reciprocal odds summed, minus one."""
    odds = np.asarray(odds, dtype=float)
    return float((1.0 / odds).sum() - 1.0)


def _shin_probs(z: float, raw: np.ndarray, booksum: float) -> np.ndarray:
    root = np.sqrt(z**2 + 4.0 * (1.0 - z) * raw**2 / booksum)
    return (root - z) / (2.0 * (1.0 - z))


def implied_shin(odds: np.ndarray) -> np.ndarray:
    """Shin (1993) implied probabilities for a single market.

    The insider-trading proportion z is the value for which Shin's implied
    probabilities sum to one; that sum is monotone in z, so we solve for it
    with a bracketed root-find. Falls back to the proportional method for a
    two-way or degenerate market.
    """
    from scipy.optimize import brentq

    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds
    booksum = raw.sum()
    if len(odds) < 3 or booksum <= 1.0:
        return raw / booksum

    f = lambda z: _shin_probs(z, raw, booksum).sum() - 1.0
    # f(0) = booksum - 1 > 0; f grows negative as z rises, so bracket upward.
    lo, hi = 0.0, 0.5
    while f(hi) > 0 and hi < 0.99:
        hi += 0.1
    try:
        z = brentq(f, lo, hi)
    except ValueError:
        return raw / booksum
    p = _shin_probs(z, raw, booksum)
    return p / p.sum()
