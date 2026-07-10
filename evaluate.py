"""
Load match results and bookmaker odds from football-data.co.uk.

That site is the pragmatic source for this project because each CSV carries
full-time goals and closing 1X2 odds together, which is exactly what the model
and its market benchmark need. FBref has richer stats but no prices, and
StatsBomb's open data has events but neither odds nor full league history.

The loader standardises whatever is available into a common schema:
    Date, HomeTeam, AwayTeam, FTHG, FTAG, OddsH, OddsD, OddsA
preferring Pinnacle closing odds, then Bet365, then the market average.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281"

# preference order for the 1X2 odds triple
ODDS_SETS = [
    ("PSCH", "PSCD", "PSCA"),  # Pinnacle closing
    ("PSH", "PSD", "PSA"),     # Pinnacle
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),  # market average
    ("BbAvH", "BbAvD", "BbAvA"),
]


def _season_url(league: str, season: str) -> str:
    # e.g. league 'E0' (Premier League), season '2324' -> .../2324/E0.csv
    return f"{BASE}/{season}/{league}.csv"


def _standardise(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not cols.issubset(raw.columns):
        raise ValueError("CSV missing expected result columns.")
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "HomeTeam": raw["HomeTeam"].astype(str),
            "AwayTeam": raw["AwayTeam"].astype(str),
            "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
        }
    )
    for h, d, a in ODDS_SETS:
        if {h, d, a}.issubset(raw.columns):
            out["OddsH"] = pd.to_numeric(raw[h], errors="coerce")
            out["OddsD"] = pd.to_numeric(raw[d], errors="coerce")
            out["OddsA"] = pd.to_numeric(raw[a], errors="coerce")
            break
    return out.dropna(subset=["Date", "FTHG", "FTAG"]).reset_index(drop=True)


def load_seasons(league: str = "E0", seasons=("2223", "2324", "2425")) -> pd.DataFrame:
    """Download and concatenate one or more seasons for a league.

    League codes follow football-data.co.uk (E0 Premier League, E1 Championship,
    SP1 La Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1). Seasons are four-digit,
    e.g. '2324' for 2023/24.
    """
    frames = []
    for season in seasons:
        url = _season_url(league, season)
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = pd.read_csv(io.BytesIO(resp.read()), encoding="latin-1")
        frames.append(_standardise(raw))
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    """Load a football-data.co.uk CSV already saved to disk."""
    raw = pd.read_csv(path, encoding="latin-1")
    return _standardise(raw)
