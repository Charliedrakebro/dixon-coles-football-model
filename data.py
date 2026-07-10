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


# --------------------------------------------------------------------------- #
# GitHub mirror (default real-data source)
#
# football-data.co.uk does not serve CORS-friendly downloads and can be blocked
# on locked-down networks, so by default we pull the same underlying data from a
# public GitHub mirror that aggregates football-data.co.uk results and odds:
#   github.com/xgabora/Club-Football-Match-Data-2000-2025
# It carries full-time goals and the market-average 1X2 odds we need. The file
# is large, so it is cached locally after the first download.
# --------------------------------------------------------------------------- #
MIRROR_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
)


def load_github_mirror(
    league: str = "E0", start: str = "2021-08-01", cache_path: str = "matches_cache.csv"
) -> pd.DataFrame:
    """Load real results and market odds for a league from the GitHub mirror.

    `league` uses football-data.co.uk division codes (E0 Premier League, SP1 La
    Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1). `start` trims to matches on or
    after that date. The full mirror is cached to `cache_path` on first use.
    """
    import os

    if not os.path.exists(cache_path):
        with urllib.request.urlopen(MIRROR_URL, timeout=120) as resp:
            data = resp.read()
        with open(cache_path, "wb") as fh:
            fh.write(data)
    raw = pd.read_csv(cache_path, low_memory=False)

    raw = raw[raw["Division"] == league].copy()
    # the mirror occasionally spells a club two ways across seasons
    # (e.g. "Nott'm Forest" vs "Nottm Forest"); strip apostrophes and
    # surrounding whitespace so those collapse to one team.
    for col in ("HomeTeam", "AwayTeam"):
        raw[col] = raw[col].astype(str).str.replace("'", "", regex=False).str.strip()
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["MatchDate"], errors="coerce"),
            "HomeTeam": raw["HomeTeam"].astype(str),
            "AwayTeam": raw["AwayTeam"].astype(str),
            "FTHG": pd.to_numeric(raw["FTHome"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["FTAway"], errors="coerce"),
            "OddsH": pd.to_numeric(raw["OddHome"], errors="coerce"),
            "OddsD": pd.to_numeric(raw["OddDraw"], errors="coerce"),
            "OddsA": pd.to_numeric(raw["OddAway"], errors="coerce"),
        }
    ).dropna(subset=["Date", "FTHG", "FTAG", "OddsH", "OddsD", "OddsA"])
    out = out[out["Date"] >= pd.to_datetime(start)]
    return out.sort_values("Date").reset_index(drop=True)
