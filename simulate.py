"""Dixon-Coles football goals model: fitting, evaluation and odds tools."""

from .model import DixonColesModel, time_decay_weights
from .odds import implied_proportional, implied_shin, overround
from .evaluate import (
    walk_forward_backtest,
    summarise_backtest,
    reliability_table,
    value_bet_backtest,
    ranked_probability_score,
)

__all__ = [
    "DixonColesModel",
    "time_decay_weights",
    "implied_proportional",
    "implied_shin",
    "overround",
    "walk_forward_backtest",
    "summarise_backtest",
    "reliability_table",
    "value_bet_backtest",
    "ranked_probability_score",
]
