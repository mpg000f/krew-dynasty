"""
Krew Dynasty — Dynasty Model
Pulls live data from Sleeper + KeepTradeCut and outputs:
  1. 2026 rookie draft board (availability + values)
  2. 2026 pick ownership board (including traded picks)
  3. Roster overview by team (dynasty value totals)

Three value columns:
  KTC      — raw KeepTradeCut dynasty value (scraped live)
  Pos Adj  — KTC + positional scarcity bonus (3RB/4WR vs standard SF)
  Model    — from-scratch: recent stats + aging curve projection + pos scarcity

Scoring: 2QB/SUPERFLEX | 0.5 PPR | 0.25 TEP
"""

import os
import json
import re
import requests
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from bs4 import BeautifulSoup
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
LEAGUE_ID  = "1312078949865500672"
DRAFT_ID   = "1312078949873885184"
SEASON     = "2026"
POSITIONS  = {"QB", "RB", "WR", "TE"}

KTC_FORMAT = 2
TEP_BONUS  = 0.25

SLEEPER = "https://api.sleeper.app/v1"

STANDARD_STARTERS = {"QB": 20, "RB": 25, "WR": 25, "TE": 10}
LEAGUE_STARTERS   = {"QB": 20, "RB": 42, "WR": 48, "TE": 10}

STAT_YEARS    = list(range(2021, 2026))
TIME_DISCOUNT = 0.85   # each projected future season worth 85% of prior year
PROJ_YEARS    = 15     # max seasons to project forward
# Year 1 production as fraction of eventual ceiling at that rank
# RBs enter hot; WRs/QBs need 2-3 years; TEs have huge Year-2 spike
YEAR1_SCALING = {"QB": 0.55, "RB": 0.80, "WR": 0.55, "TE": 0.30}

MY_TEAM    = "pltiii"
OUTPUT_DIR = "docs"

# ── Age Curves ────────────────────────────────────────────────────────────────
# Production relative to position peak (1.00).
# Used only in the from-scratch Model column — KTC already bakes in age.
# Sources: age-indexed production articles (RB/WR/TE) + passing consensus (QB).
AGE_CURVES = {
    "RB": {
        21: 1.00, 22: 1.00, 23: 1.00, 24: 1.00, 25: 1.00, 26: 1.00,
        27: 0.92, 28: 0.80, 29: 0.62, 30: 0.48, 31: 0.36, 32: 0.25,
        33: 0.18, 34: 0.12, 35: 0.08,
    },
    "WR": {
        21: 0.80, 22: 0.80, 23: 0.93, 24: 0.91, 25: 0.95, 26: 1.00,
        27: 1.00, 28: 1.00, 29: 0.96, 30: 0.92, 31: 0.87, 32: 0.78,
        33: 0.74, 34: 0.62, 35: 0.50, 36: 0.38,
    },
    "TE": {
        21: 0.50, 22: 0.55, 23: 0.65, 24: 0.78, 25: 0.90, 26: 1.00,
        27: 1.00, 28: 1.00, 29: 1.00, 30: 0.98, 31: 0.93, 32: 0.88,
        33: 0.82, 34: 0.78, 35: 0.65,
    },
    "QB": {
        22: 0.78, 23: 0.83, 24: 0.88, 25: 0.93, 26: 0.96, 27: 0.98,
        28: 1.00, 29: 1.00, 30: 1.00, 31: 0.98, 32: 0.95, 33: 0.90,
        34: 0.84, 35: 0.75, 36: 0.65, 37: 0.55,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get(url, **kwargs):
    r = requests.get(url, timeout=25, **kwargs)
    r.raise_for_status()
    return r.json()


def normalize(name):
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


# ── Fantasy Scoring ───────────────────────────────────────────────────────────
def _league_pts(row):
    """Fantasy points under this league's exact scoring settings."""
    pos = row["position"]
    p  = row.get("passing_yards", 0)            * 0.04
    p += row.get("passing_tds", 0)              * 4
    p += row.get("interceptions", 0)            * -1
    p += row.get("passing_2pt_conversions", 0)  * 2
    p += row.get("rushing_yards", 0)            * 0.1
    p += row.get("rushing_tds", 0)              * 6
    p += row.get("rushing_2pt_conversions", 0)  * 2
    rec_bonus = TEP_BONUS if pos == "TE" else 0
    p += row.get("receptions", 0)               * (0.5 + rec_bonus)
    p += row.get("receiving_yards", 0)          * 0.1
    p += row.get("receiving_tds", 0)            * 6
    p += row.get("receiving_2pt_conversions", 0)* 2
    p += (row.get("sack_fumbles_lost", 0) +
          row.get("rushing_fumbles_lost", 0) +
          row.get("receiving_fumbles_lost", 0)) * -2
    return p


# ── Stats Pipeline ────────────────────────────────────────────────────────────
_NFLVERSE_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download"
    "/stats_player/stats_player_reg_{year}.parquet"
)

def load_stats_data():
    """
    Load seasonal stats from nflverse stats_player release (covers 2021–2025).
    Fetches parquet files directly, skipping any year not yet published.
    Normalises column names to match _league_pts expectations.
    """
    frames = []
    for yr in STAT_YEARS:
        url = _NFLVERSE_STATS_URL.format(year=yr)
        try:
            import io
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(r.content)))
        except Exception:
            pass

    if not frames:
        raise RuntimeError("No seasonal stats available from nflverse for any year in STAT_YEARS.")

    df = pd.concat(frames, ignore_index=True)

    # Normalise column names so _league_pts / build_stats_baseline work unchanged
    df = df.drop(columns=["player_name"], errors="ignore")  # drop abbreviated name
    df = df.rename(columns={
        "player_display_name": "player_name",
        "passing_interceptions": "interceptions",
    })

    loaded = sorted(df["season"].dropna().astype(int).unique())
    print(f"Loading historical stats ({loaded[0]}–{loaded[-1]})…")

    df = df[df["position"].isin(POSITIONS) & (df["games"] >= 4)].copy()
    df["fpts"]    = df.apply(_league_pts, axis=1)
    df["fpts_17"] = df["fpts"] / df["games"] * 17
    return df


def build_scoring_curve(df):
    """
    Returns a curve_pts(pos, rank) function: average fantasy points at each positional rank.
    Averages across all seasons in df, giving a stable baseline for each position slot.
    """
    records = []
    for _, grp in df.groupby("season"):
        for pos, pgrp in grp.groupby("position"):
            ranked = pgrp.sort_values("fpts_17", ascending=False).reset_index(drop=True)
            ranked["pos_rank"] = range(1, len(ranked) + 1)
            records.append(ranked[["pos_rank", "position", "fpts_17"]])

    curve_df = (pd.concat(records)
                  .groupby(["position", "pos_rank"])["fpts_17"]
                  .mean()
                  .reset_index()
                  .rename(columns={"fpts_17": "avg_fpts"}))

    def curve_pts(pos, rank):
        sub = curve_df[(curve_df.position == pos) & (curve_df.pos_rank == rank)]
        if sub.empty:
            sub = curve_df[curve_df.position == pos]
            sub = sub.iloc[(sub["pos_rank"] - rank).abs().argsort()].head(1)
        return float(sub["avg_fpts"].values[0]) if not sub.empty else 0.0

    return curve_pts


def build_stats_baseline(df):
    """
    Per-player weighted fantasy points baseline from recent seasons.
    Uses 60% most-recent season + 40% second-most-recent (falls back if only one available).
    Returns dict: normalize(name) -> fpts_17 (float)
    """
    baseline = {}
    for player_name, grp in df.groupby("player_name"):
        if not isinstance(player_name, str) or not player_name.strip():
            continue
        yr_data   = dict(zip(grp["season"].astype(int), grp["fpts_17"].astype(float)))
        available = sorted(yr_data.keys(), reverse=True)
        if not available:
            continue
        fpts = (0.60 * yr_data[available[0]] + 0.40 * yr_data[available[1]]
                if len(available) >= 2 else yr_data[available[0]])
        key = normalize(player_name)
        if key:
            baseline[key] = fpts
    return baseline


def build_rookie_imputation(curve_pts):
    """
    Expected Year 1 production for rookies by position and positional rank.
    Uses historical average production at each rank × Year 1 scaling factor.
    """
    return {
        pos: {rank: curve_pts(pos, rank) * YEAR1_SCALING[pos] for rank in range(1, 71)}
        for pos in POSITIONS
    }


# ── Positional Adjustment (additive VORP) ────────────────────────────────────
def build_position_adjustments(ktc_players, curve_pts):
    """
    Compute an additive KTC bonus for each positional rank based on how much
    deeper this league's starter pool is vs standard SF.
    Applied to both Pos Adj and Model columns.
    """
    ktc_val = {}
    for p in ktc_players:
        pos  = p.get("position", "")
        rank = p.get("pos_rank")
        if pos in POSITIONS and rank:
            ktc_val.setdefault(pos, {})[rank] = p["value"]

    adjustments = {}
    for pos in POSITIONS:
        std_repl_rank    = STANDARD_STARTERS[pos]
        league_repl_rank = LEAGUE_STARTERS[pos]
        std_repl_pts     = curve_pts(pos, std_repl_rank)
        league_repl_pts  = curve_pts(pos, league_repl_rank)

        hi_rank = min(3,  max(ktc_val.get(pos, {}).keys(), default=3))
        lo_rank = min(15, max(ktc_val.get(pos, {}).keys(), default=15))
        ktc_hi  = ktc_val.get(pos, {}).get(hi_rank)
        ktc_lo  = ktc_val.get(pos, {}).get(lo_rank)
        if ktc_hi and ktc_lo:
            pts_diff = curve_pts(pos, hi_rank) - curve_pts(pos, lo_rank)
            k = (ktc_hi - ktc_lo) / pts_diff if pts_diff > 0 else 35.0
        else:
            k = 35.0

        bonuses = {}
        for rank in range(1, 71):
            pts = curve_pts(pos, rank)
            if pts >= std_repl_pts:
                extra = 0
            elif pts >= league_repl_pts:
                extra = pts - league_repl_pts
            else:
                extra = 0
            bonuses[rank] = round(k * extra)

        adjustments[pos] = bonuses
        print(f"  {pos}: std_repl={std_repl_pts:.0f}pts  league_repl={league_repl_pts:.0f}pts  "
              f"k={k:.1f}  bonus@rank1={bonuses[1]}  bonus@std_repl={bonuses[std_repl_rank]}")

    return adjustments


def apply_position_adjustment(players, adjustments):
    """Add positional scarcity bonus to KTC value → adj_value."""
    for p in players:
        pos  = p.get("position", "")
        rank = p.get("pos_rank")
        if pos not in adjustments or rank is None or p.get("is_pick"):
            p["adj_value"] = p["value"]
            continue
        max_rank   = max(adjustments[pos].keys())
        bonus      = adjustments[pos].get(min(int(rank), max_rank), 0)
        p["adj_value"] = p["value"] + bonus
    return players


# ── From-Scratch Model ────────────────────────────────────────────────────────
def compute_model_values(players, stats_baseline, rookie_imputation):
    """
    Build dynasty value from scratch for each player.

    For each player:
      1. Get a production baseline (fpts per 17-game season):
           - Veterans: weighted avg of last 2 seasons from nfl_data_py
           - Rookies:  imputed from their positional KTC rank × Year 1 scaling
           - No-stats vets: rank-based imputation at full (non-rookie) production
      2. Project forward PROJ_YEARS seasons using AGE_CURVES:
           projected[y] = baseline × (curve[age+y] / curve[age])
      3. Discount each year: × TIME_DISCOUNT^y
      4. Sum to get dynasty_score (in discounted fantasy-point units)
      5. Normalize all scores to 0–9999 scale (same as KTC)
    """
    print("Computing from-scratch dynasty model…")

    for p in players:
        pos     = p.get("position", "")
        age     = p.get("age")
        is_rook = p.get("years_exp") == 0 or p.get("rookie")

        if p.get("is_pick") or pos not in AGE_CURVES or not age:
            p["model_score"] = 0.0
            continue

        curve   = AGE_CURVES[pos]
        age_int = max(min(curve.keys()), min(int(age), max(curve.keys())))
        cur_cv  = curve[age_int]

        # Production baseline
        if is_rook:
            rank     = p.get("pos_rank", 20)
            max_rank = max(rookie_imputation[pos].keys())
            baseline = rookie_imputation[pos].get(min(rank, max_rank), 0.0)
        else:
            baseline = stats_baseline.get(normalize(p["name"]))
            if baseline is None:
                # No recent stats — impute from positional rank at full production
                rank     = p.get("pos_rank", 20)
                max_rank = max(rookie_imputation[pos].keys())
                yr1      = rookie_imputation[pos].get(min(rank, max_rank), 0.0)
                scaling  = YEAR1_SCALING.get(pos, 0.60)
                baseline = yr1 / scaling if scaling > 0 else 0.0

        if not baseline or baseline <= 0 or cur_cv <= 0:
            p["model_score"] = 0.0
            continue

        score = 0.0
        for y in range(PROJ_YEARS):
            fa  = age_int + y
            fcv = curve.get(min(fa, max(curve.keys())), 0.0)
            if fcv < 0.05:
                break
            score += baseline * (fcv / cur_cv) * (TIME_DISCOUNT ** y)

        p["model_score"] = score

    # Normalize to 0–9999
    valid     = [p["model_score"] for p in players if not p.get("is_pick") and p.get("model_score", 0) > 0]
    max_score = max(valid) if valid else 1.0
    for p in players:
        raw = p.get("model_score", 0)
        p["model_value"] = round(9999 * raw / max_score) if raw > 0 else 0

    return players


def apply_model_positional_adjustment(players, adjustments):
    """Add the same positional scarcity bonus to model_value → model_adj."""
    for p in players:
        pos  = p.get("position", "")
        rank = p.get("pos_rank")
        if pos not in adjustments or rank is None or p.get("is_pick"):
            p["model_adj"] = p.get("model_value", 0)
            continue
        max_rank   = max(adjustments[pos].keys())
        bonus      = adjustments[pos].get(min(int(rank), max_rank), 0)
        p["model_adj"] = p.get("model_value", 0) + bonus
    return players


# ── KTC Scraper ───────────────────────────────────────────────────────────────
KTC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

def fetch_ktc():
    """
    Fetch dynasty player values from KeepTradeCut.
    Uses superflexValues.tep.value (native SF + 0.25 TEP).
    """
    url = (
        f"https://keeptradecut.com/dynasty-rankings"
        f"?filters=QB|WR|RB|TE|RDP&format={KTC_FORMAT}"
    )
    print("Fetching KTC dynasty rankings…")
    try:
        r = requests.get(url, headers=KTC_HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  Warning: KTC request failed ({e}). Falling back to Fantasy Calc.")
        return fetch_fantasycalc()

    soup = BeautifulSoup(r.text, "html.parser")
    for script in soup.find_all("script"):
        text = script.string or ""
        if "playersArray" not in text:
            continue
        m = re.search(r"var playersArray\s*=\s*(\[.*?\]);", text, re.DOTALL)
        if m:
            try:
                players_raw = json.loads(m.group(1))
                result = parse_ktc_players(players_raw)
                result = add_positional_ranks(result)
                print(f"  Loaded {len(result)} players from KTC (native TEP+SF values).")
                return result
            except Exception as e:
                print(f"  KTC parse error: {e}. Falling back to Fantasy Calc.")
                return fetch_fantasycalc()

    print("  Warning: Could not find KTC playersArray. Falling back to Fantasy Calc.")
    return fetch_fantasycalc()


def parse_ktc_players(raw):
    """Parse KTC playersArray into standardized dicts."""
    out = []
    for p in raw:
        pos     = (p.get("position") or "").upper()
        is_pick = pos == "RDP"
        if pos not in POSITIONS and not is_pick:
            continue

        sf_vals  = p.get("superflexValues", {})
        tep_vals = sf_vals.get("tep", {})
        value    = tep_vals.get("value") or sf_vals.get("value") or 0
        rank     = tep_vals.get("rank")  or sf_vals.get("rank")  or 999

        out.append({
            "name":      p.get("playerName", ""),
            "position":  "PICK" if is_pick else pos,
            "team":      p.get("team") or "FA",
            "age":       p.get("age"),
            "value":     int(value),
            "rookie":    bool(p.get("rookie")),
            "years_exp": int(p.get("seasonsExperience") or 0),
            "is_pick":   is_pick,
            "ktc_rank":  rank,
            "source":    "KTC",
        })
    return out


def add_positional_ranks(players):
    """Assign pos_rank (rank within position by KTC value, 1 = highest)."""
    by_pos = {}
    for i, p in enumerate(players):
        pos = p.get("position", "")
        if pos in POSITIONS:
            by_pos.setdefault(pos, []).append(i)
    for pos, indices in by_pos.items():
        sorted_idx = sorted(indices, key=lambda i: -players[i]["value"])
        for rank, idx in enumerate(sorted_idx, 1):
            players[idx]["pos_rank"] = rank
    return players


def fetch_fantasycalc():
    """Fallback: fetch from Fantasy Calc (2QB, 0.5 PPR, 0.25 TEP)."""
    print("Fetching dynasty values from Fantasy Calc (fallback)…")
    url = (
        "https://api.fantasycalc.com/values/current"
        "?isDynasty=true&numQbs=2&ppr=0.5&tep=0.25"
    )
    data = get(url)
    out  = []
    for row in data:
        p       = row.get("player", {})
        pos     = (p.get("position") or "").upper()
        is_pick = pos == "PICK" or not pos
        if pos not in POSITIONS and not is_pick:
            continue
        out.append({
            "name":       p.get("name", ""),
            "position":   pos,
            "team":       p.get("maybeTeam") or "FA",
            "age":        p.get("maybeAge"),
            "value":      int(row.get("value", 0)),
            "rookie":     int(p.get("maybeYoe") or 99) == 0,
            "years_exp":  int(p.get("maybeYoe") or 0),
            "is_pick":    is_pick,
            "sleeper_id": str(p.get("sleeperId")) if p.get("sleeperId") else None,
            "source":     "FC",
        })
    return out


def apply_tep(players):
    """Boost TE values for TEP scoring (Fantasy Calc fallback only)."""
    te_players = [(i, p) for i, p in enumerate(players) if p["position"] == "TE"]
    if not te_players:
        return players
    n = len(te_players)
    for rank, (i, p) in enumerate(sorted(te_players, key=lambda x: -x[1]["value"])):
        est_rec     = max(30, 100 - int(70 * rank / max(n - 1, 1)))
        value_boost = int(TEP_BONUS * est_rec * 5)
        players[i]["value"] += value_boost
    return players


# ── Sleeper Data ──────────────────────────────────────────────────────────────
def fetch_sleeper():
    print("Fetching Sleeper league data…")
    users        = get(f"{SLEEPER}/league/{LEAGUE_ID}/users")
    rosters      = get(f"{SLEEPER}/league/{LEAGUE_ID}/rosters")
    traded_picks = get(f"{SLEEPER}/league/{LEAGUE_ID}/traded_picks")
    draft_info   = get(f"{SLEEPER}/draft/{DRAFT_ID}")
    print("Fetching Sleeper player database (may take a moment)…")
    all_players  = get(f"{SLEEPER}/players/nfl")
    return users, rosters, traded_picks, draft_info, all_players


# ── Build Roster / Owner Maps ─────────────────────────────────────────────────
def build_maps(users, rosters):
    user_by_id = {u["user_id"]: u for u in users}
    roster_to_owner = {}
    for r in rosters:
        uid = r["owner_id"]
        u   = user_by_id.get(uid, {})
        roster_to_owner[r["roster_id"]] = (
            u.get("team_name") or u.get("display_name") or f"Roster {r['roster_id']}"
        )
    player_on_roster = {}
    for r in rosters:
        for pid in (r.get("players") or []):
            player_on_roster[pid] = r["roster_id"]
    return roster_to_owner, player_on_roster


# ── 2026 Draft Pick Board ─────────────────────────────────────────────────────
def build_pick_board(draft_info, traded_picks):
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    n_rounds = draft_info["settings"]["rounds"]
    n_teams  = draft_info["settings"]["teams"]
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    board = {}
    for rnd in range(1, n_rounds + 1):
        for slot, roster_id in enumerate(pick_order, start=1):
            board[(rnd, slot)] = {
                "round": rnd, "slot": slot,
                "pick": f"{rnd}.{slot:02d}",
                "original_roster": roster_id,
                "current_owner":   roster_id,
            }

    for t in traded_picks:
        if t["season"] != SEASON:
            continue
        orig = t["roster_id"]
        for slot, rid in enumerate(pick_order, start=1):
            if rid == orig and (t["round"], slot) in board:
                board[(t["round"], slot)]["current_owner"] = t["owner_id"]
                break

    return sorted(board.values(), key=lambda x: (x["round"], x["slot"]))


def slot_to_bucket(slot, n_teams=10):
    third = n_teams / 3
    return "Early" if slot <= third else ("Mid" if slot <= 2 * third else "Late")


def build_pick_value_map(ktc_raw):
    rnd_map   = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    pick_vals = {}
    for p in ktc_raw:
        if not p.get("is_pick"):
            continue
        parts = p["name"].split()
        if len(parts) == 3:
            year, bucket, rnd_str = parts
            rnd = rnd_map.get(rnd_str)
            if rnd:
                pick_vals[(year, rnd, bucket)] = p["value"]
    return pick_vals


def build_team_picks(draft_info, traded_picks, pick_value_map, n_teams=10):
    seasons    = sorted(set([SEASON] + [t["season"] for t in traded_picks]))
    n_rounds   = draft_info["settings"]["rounds"]
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    owner_of = {}
    for season in seasons:
        for rnd in range(1, n_rounds + 1):
            for rid in range(1, n_teams + 1):
                owner_of[(season, rnd, rid)] = rid

    for t in traded_picks:
        owner_of[(t["season"], t["round"], t["roster_id"])] = t["owner_id"]

    rnd_label = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    team_picks = {rid: [] for rid in range(1, n_teams + 1)}

    for (season, rnd, orig_roster), current_owner in sorted(owner_of.items()):
        if season == SEASON:
            try:
                slot   = pick_order.index(orig_roster) + 1
                bucket = slot_to_bucket(slot, n_teams)
            except ValueError:
                bucket = "Mid"
        else:
            bucket = "Mid"

        label = f"{season} {bucket} {rnd_label.get(rnd, f'{rnd}th')}"
        val   = pick_value_map.get((season, rnd, bucket), 0)
        team_picks[current_owner].append({
            "label": label, "season": season,
            "round": rnd, "orig": orig_roster, "value": val,
        })

    return team_picks


# ── Match Value Source to Sleeper Players ────────────────────────────────────
def match_players(value_players, all_players, player_on_roster):
    sleeper_by_name = {}
    for pid, p in all_players.items():
        if p.get("position") not in POSITIONS:
            continue
        key = normalize(p.get("full_name", ""))
        if key:
            sleeper_by_name[key] = pid

    matched = []
    for p in value_players:
        if p.get("is_pick"):
            continue
        sid = p.get("sleeper_id")
        if sid and sid not in all_players:
            sid = None
        if not sid:
            sid = sleeper_by_name.get(normalize(p["name"]))
        sp  = all_players.get(sid, {}) if sid else {}
        rid = player_on_roster.get(sid)
        matched.append({
            **p,
            "sleeper_id":   sid,
            "years_exp":    sp.get("years_exp"),
            "on_roster_id": rid,
        })
    return matched


# ── Console Output ────────────────────────────────────────────────────────────
W = 80

def divider(title=""):
    if title:
        pad = (W - len(title) - 4) // 2
        print("=" * pad + f"  {title}  " + "=" * max(0, W - pad - len(title) - 4))
    else:
        print("=" * W)


def print_rookie_board(players, roster_to_owner):
    divider("2026 ROOKIE DRAFT BOARD — Krew Dynasty")
    print(f"  {'RK':<4} {'NAME':<24} {'POS':<4} {'TEAM':<5} {'AGE':<6} {'MODEL':>7} {'POS ADJ':>8} {'KTC':>7}  STATUS")
    divider()
    rookies = sorted(
        [p for p in players if p.get("years_exp") == 0 or p.get("rookie")],
        key=lambda x: -x.get("model_adj", 0)
    )
    for rk, p in enumerate(rookies[:40], 1):
        rid    = p.get("on_roster_id")
        owner  = roster_to_owner.get(rid, "") if rid else ""
        status = f"STASHED — {owner}" if owner else "AVAILABLE"
        age_s  = f"{p['age']:.1f}" if (p.get("age") and p["age"] > 0) else "  ?"
        print(
            f"  {rk:<4} {p['name']:<24} {p['position']:<4} {p['team']:<5} "
            f"{age_s:<6} {p.get('model_adj', 0):>7} {p.get('adj_value', p['value']):>8} "
            f"{p['value']:>7}  {status}"
        )


def print_pick_board(pick_board, roster_to_owner):
    divider("2026 DRAFT PICK OWNERSHIP")
    print(f"  {'PICK':<8} {'CURRENT OWNER':<28} ORIGINAL OWNER")
    divider()
    for p in pick_board:
        orig   = roster_to_owner.get(p["original_roster"], f"R{p['original_roster']}")
        curr   = roster_to_owner.get(p["current_owner"],   f"R{p['current_owner']}")
        traded = "  ← TRADED" if p["original_roster"] != p["current_owner"] else ""
        print(f"  {p['pick']:<8} {curr:<28} {orig}{traded}")


def print_roster_overview(rosters, all_players, players_matched, roster_to_owner,
                          team_picks=None, my_team=None):
    model_by_sid  = {p["sleeper_id"]: p.get("model_adj", 0)  for p in players_matched if p.get("sleeper_id")}
    model_by_name = {normalize(p["name"]): p.get("model_adj", 0) for p in players_matched}
    ktc_by_sid    = {p["sleeper_id"]: p["value"]              for p in players_matched if p.get("sleeper_id")}
    ktc_by_name   = {normalize(p["name"]): p["value"]         for p in players_matched}

    def get_vals(pid):
        name  = all_players.get(pid, {}).get("full_name", "")
        key   = normalize(name)
        model = model_by_sid.get(pid) or model_by_name.get(key, 0)
        ktc   = ktc_by_sid.get(pid)   or ktc_by_name.get(key,   0)
        return model, ktc

    team_totals = []
    for r in rosters:
        rid        = r["roster_id"]
        owner      = roster_to_owner[rid]
        pids       = [pid for pid in (r.get("players") or [])
                      if all_players.get(pid, {}).get("position") in POSITIONS]
        player_model = sum(get_vals(pid)[0] for pid in pids)
        picks_26     = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] == SEASON)
        picks_fut    = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] != SEASON)
        total        = player_model + picks_26 + picks_fut
        team_totals.append((rid, owner, player_model, picks_26, picks_fut, total))

    team_totals.sort(key=lambda x: -x[5])

    divider("DYNASTY POWER RANKINGS  (Model value: stats + aging + pos scarcity)")
    print(f"  {'RK':<4} {'TEAM':<28} {'PLAYERS':>9} {'26 PICKS':>9} {'FUT PICKS':>10} {'TOTAL':>8}")
    divider()
    for rk, (rid, owner, pv, p26, pf, tot) in enumerate(team_totals, 1):
        marker = " ◄ YOU" if owner == my_team else ""
        print(f"  {rk:<4} {owner:<28} {pv:>9,} {p26:>9,} {pf:>10,} {tot:>8,}{marker}")

    divider("ROSTER DETAIL")
    for rid, owner, pv, p26, pf, tot in team_totals:
        pid_list  = next(r.get("players") or [] for r in rosters if r["roster_id"] == rid)
        team_rows = []
        for pid in pid_list:
            p   = all_players.get(pid, {})
            pos = p.get("position", "")
            if pos not in POSITIONS:
                continue
            model, ktc = get_vals(pid)
            team_rows.append((p.get("full_name", pid), pos, model, ktc))
        team_rows.sort(key=lambda x: -x[2])

        marker = "  ◄ YOU" if owner == my_team else ""
        print(f"\n  {owner}{marker}")
        print(f"  Players: {pv:,}  |  2026 Picks: {p26:,}  |  Future Picks: {pf:,}  |  Total: {tot:,}")
        print(f"  {'POS':<4} {'NAME':<26} {'MODEL':>7} {'KTC':>7}")
        for name, pos, model, ktc in team_rows[:12]:
            print(f"  {pos:<4} {name:<26} {model:>7} {ktc:>7}")
        if len(team_rows) > 12:
            print(f"       … +{len(team_rows) - 12} more players")

        picks = sorted((team_picks or {}).get(rid, []), key=lambda x: (-int(x["season"]), x["round"]))
        if picks:
            print(f"  {'PICK':<32} KTC")
            for pk in picks:
                print(f"  {pk['label']:<32} {pk['value']:>5}")


# ── HTML Data Helpers ─────────────────────────────────────────────────────────
def compute_team_totals(rosters, all_players, players_matched, roster_to_owner, team_picks):
    model_by_sid  = {p["sleeper_id"]: p.get("model_adj", 0)  for p in players_matched if p.get("sleeper_id")}
    model_by_name = {normalize(p["name"]): p.get("model_adj", 0) for p in players_matched}
    ktc_by_sid    = {p["sleeper_id"]: p["value"]              for p in players_matched if p.get("sleeper_id")}
    ktc_by_name   = {normalize(p["name"]): p["value"]         for p in players_matched}

    def get_vals(pid):
        name  = all_players.get(pid, {}).get("full_name", "")
        key   = normalize(name)
        model = model_by_sid.get(pid) or model_by_name.get(key, 0)
        ktc   = ktc_by_sid.get(pid)   or ktc_by_name.get(key,   0)
        return model, ktc

    result = []
    for r in rosters:
        rid   = r["roster_id"]
        owner = roster_to_owner[rid]
        pids  = [pid for pid in (r.get("players") or [])
                 if all_players.get(pid, {}).get("position") in POSITIONS]

        player_rows = []
        for pid in pids:
            p   = all_players.get(pid, {})
            pos = p.get("position", "")
            if pos not in POSITIONS:
                continue
            model, ktc = get_vals(pid)
            player_rows.append({"name": p.get("full_name", pid), "pos": pos, "model": model, "ktc": ktc})
        player_rows.sort(key=lambda x: -x["model"])

        player_model = sum(x["model"] for x in player_rows)
        picks_26     = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] == SEASON)
        picks_fut    = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] != SEASON)
        picks_list   = sorted((team_picks or {}).get(rid, []), key=lambda x: (-int(x["season"]), x["round"]))

        result.append({
            "rid": rid, "owner": owner,
            "player_model": player_model, "picks_26": picks_26, "picks_fut": picks_fut,
            "total": player_model + picks_26 + picks_fut,
            "players": player_rows, "picks": picks_list,
        })

    result.sort(key=lambda x: -x["total"])
    return result


def get_rookie_list(players, roster_to_owner):
    rookies = sorted(
        [p for p in players if p.get("years_exp") == 0 or p.get("rookie")],
        key=lambda x: -x["value"]
    )
    result = []
    for rk, p in enumerate(rookies[:40], 1):
        rid   = p.get("on_roster_id")
        owner = roster_to_owner.get(rid, "") if rid else ""
        result.append({
            "rank": rk, "name": p["name"], "pos": p["position"],
            "team": p["team"], "age": p.get("age"),
            "ktc":   p["value"],
            "adj":   p.get("adj_value", p["value"]),
            "model": p.get("model_adj", 0),
            "status": "STASHED" if owner else "AVAILABLE", "owner": owner,
        })
    return result


def get_all_players_data(players, roster_to_owner):
    result = []
    for p in players:
        if p.get("is_pick"):
            continue
        rid   = p.get("on_roster_id")
        owner = roster_to_owner.get(rid, "") if rid else ""
        ktc   = p["value"]
        adj   = p.get("adj_value", ktc)
        model = p.get("model_adj", 0)
        diff  = round((model - ktc) / ktc * 100) if ktc else 0
        result.append({
            "name":  p["name"],
            "pos":   p.get("position", ""),
            "team":  p.get("team", "FA"),
            "age":   p.get("age"),
            "ktc":   ktc,
            "adj":   adj,
            "model": model,
            "diff":  diff,
            "owner": owner,
        })
    result.sort(key=lambda x: -x["model"])
    return result


# ── HTML Rendering ────────────────────────────────────────────────────────────
def render_html(team_totals, rookies, all_players_data, pick_board, roster_to_owner, updated_at):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def fmt(n):
        return f"{n:,}"

    def diff_badge(d):
        if d > 10:
            return f'<span class="badge bg-success">+{d}%</span>'
        elif d < -10:
            return f'<span class="badge bg-danger">{d}%</span>'
        return f'<span class="text-secondary small">{d:+}%</span>'

    # ── Power Rankings rows ──────────────────────────────────────────────────
    pr_rows = ""
    for rk, t in enumerate(team_totals, 1):
        is_me = t["owner"] == MY_TEAM
        row_class = ' class="you-row"' if is_me else ""
        you = " ◄ YOU" if is_me else ""
        pr_rows += (
            f'<tr{row_class}>'
            f'<td>{rk}</td><td><strong>{t["owner"]}</strong>{you}</td>'
            f'<td class="text-end">{fmt(t["player_model"])}</td>'
            f'<td class="text-end">{fmt(t["picks_26"])}</td>'
            f'<td class="text-end">{fmt(t["picks_fut"])}</td>'
            f'<td class="text-end fw-bold">{fmt(t["total"])}</td>'
            f'</tr>\n'
        )

    # ── Roster accordion ────────────────────────────────────────────────────
    accordion_items = ""
    for rk, t in enumerate(team_totals, 1):
        is_me  = t["owner"] == MY_TEAM
        you    = " ◄ YOU" if is_me else ""
        p_rows = ""
        for p in t["players"][:16]:
            diff = round((p["model"] - p["ktc"]) / p["ktc"] * 100) if p["ktc"] else 0
            p_rows += (
                f'<tr><td>{p["pos"]}</td><td>{p["name"]}</td>'
                f'<td class="text-end">{fmt(p["model"])}</td>'
                f'<td class="text-end text-secondary">{fmt(p["ktc"])}</td>'
                f'<td class="text-end">{diff_badge(diff)}</td></tr>\n'
            )
        if len(t["players"]) > 16:
            p_rows += f'<tr><td colspan="5" class="text-secondary small">… +{len(t["players"]) - 16} more</td></tr>'

        pk_rows = "".join(
            f'<tr><td>{pk["label"]}</td><td class="text-end">{fmt(pk["value"])}</td></tr>'
            for pk in t["picks"]
        )
        picks_section = (
            f'<h6 class="mt-3 mb-1">Draft Picks <span class="text-secondary small fw-normal">(KTC)</span></h6>'
            f'<table class="table table-sm mb-0"><thead><tr><th>Pick</th><th class="text-end">KTC</th></tr></thead>'
            f'<tbody>{pk_rows}</tbody></table>'
        ) if pk_rows else ""

        extra = "fw-bold text-warning" if is_me else ""
        accordion_items += f"""
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button collapsed {extra}" type="button"
                    data-bs-toggle="collapse" data-bs-target="#team-{rk}">
              #{rk}&nbsp;<strong>{t["owner"]}</strong>{you}
              <span class="ms-auto me-3 text-secondary small fw-normal">
                Players {fmt(t["player_model"])} &middot; Picks {fmt(t["picks_26"] + t["picks_fut"])} &middot; Total {fmt(t["total"])}
              </span>
            </button>
          </h2>
          <div id="team-{rk}" class="accordion-collapse collapse">
            <div class="accordion-body p-2">
              <table class="table table-sm mb-0">
                <thead><tr><th>Pos</th><th>Name</th><th class="text-end">Model</th><th class="text-end">KTC</th><th class="text-end">Δ%</th></tr></thead>
                <tbody>{p_rows}</tbody>
              </table>
              {picks_section}
            </div>
          </div>
        </div>"""

    # ── Rookie rows ─────────────────────────────────────────────────────────
    rookie_rows = ""
    for r in rookies:
        badge = (
            f'<span class="badge bg-warning text-dark">Stashed &mdash; {r["owner"]}</span>'
            if r["status"] == "STASHED"
            else '<span class="badge bg-success">Available</span>'
        )
        age_s = f'{r["age"]:.1f}' if r.get("age") and r["age"] > 0 else "?"
        rookie_rows += (
            f'<tr><td>{r["rank"]}</td><td>{r["name"]}</td><td>{r["pos"]}</td>'
            f'<td>{r["team"]}</td><td>{age_s}</td>'
            f'<td class="text-end fw-bold">{fmt(r["model"])}</td>'
            f'<td class="text-end">{fmt(r["adj"])}</td>'
            f'<td class="text-end text-secondary">{fmt(r["ktc"])}</td>'
            f'<td>{badge}</td></tr>\n'
        )

    # ── Pick ownership rows ─────────────────────────────────────────────────
    pick_rows = ""
    for p in pick_board:
        orig   = roster_to_owner.get(p["original_roster"], f'R{p["original_roster"]}')
        curr   = roster_to_owner.get(p["current_owner"],   f'R{p["current_owner"]}')
        traded = p["original_roster"] != p["current_owner"]
        badge  = ' <span class="badge bg-info text-dark">Traded</span>' if traded else ""
        pick_rows += (
            f'<tr><td>{p["pick"]}</td>'
            f'<td>{curr}{badge}</td>'
            f'<td class="text-secondary">{orig}</td></tr>\n'
        )

    # ── All players rows ────────────────────────────────────────────────────
    all_player_rows = ""
    for rk, p in enumerate(all_players_data, 1):
        age_s  = f'{p["age"]:.1f}' if p.get("age") and p["age"] > 0 else "?"
        owner  = p["owner"] or '<span class="text-secondary">—</span>'
        is_me  = p["owner"] == MY_TEAM
        row_cls = ' class="you-row"' if is_me else ""
        all_player_rows += (
            f'<tr data-pos="{p["pos"]}"{row_cls}>'
            f'<td>{rk}</td><td>{p["name"]}</td><td>{p["pos"]}</td>'
            f'<td>{p["team"]}</td><td>{age_s}</td>'
            f'<td class="text-end fw-bold">{fmt(p["model"])}</td>'
            f'<td class="text-end">{fmt(p["adj"])}</td>'
            f'<td class="text-end text-secondary">{fmt(p["ktc"])}</td>'
            f'<td class="text-end">{diff_badge(p["diff"])}</td>'
            f'<td>{owner}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Krew Dynasty</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ background: #0d1117; }}
    .you-row td {{ background-color: rgba(255, 193, 7, 0.12) !important; }}
    .accordion-button {{ background: #161b22; color: #c9d1d9; }}
    .accordion-button:not(.collapsed) {{ background: #1f2937; color: #e6edf3; box-shadow: none; }}
    .accordion-button::after {{ filter: invert(1) brightness(0.7); }}
    .accordion-item {{ border-color: #30363d; background: #0d1117; }}
    th {{ white-space: nowrap; }}
  </style>
</head>
<body>

<nav class="navbar border-bottom border-secondary mb-4" style="background:#161b22">
  <div class="container-fluid">
    <span class="navbar-brand fw-bold fs-5">🏈 Krew Dynasty</span>
    <small class="text-secondary">Updated {updated_at} &nbsp;·&nbsp; 2QB/SF &nbsp;·&nbsp; 0.5 PPR &nbsp;·&nbsp; 0.25 TEP</small>
  </div>
</nav>

<div class="container-xl px-3">

  <div class="alert alert-secondary small py-2 mb-4">
    <strong>Model</strong> = from-scratch dynasty value (recent stats × aging curve projection × positional scarcity) &nbsp;·&nbsp;
    <strong>Pos Adj</strong> = KTC + positional scarcity bonus for 3RB/4WR/2FLEX &nbsp;·&nbsp;
    <strong>KTC</strong> = raw KeepTradeCut market value &nbsp;·&nbsp;
    <strong>Δ%</strong> = how much Model differs from KTC
  </div>

  <ul class="nav nav-tabs mb-4" id="tabs">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-rankings" type="button">Power Rankings</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-all-players" type="button">All Players</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-rookies" type="button">Rookie Board</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-picks" type="button">2026 Pick Ownership</button></li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-rankings">
      <p class="text-secondary small mb-3">Sorted by Model value. Players column uses Model; pick values use KTC.</p>
      <div class="table-responsive mb-4">
        <table class="table table-sm table-hover align-middle">
          <thead class="table-secondary text-dark">
            <tr><th>Rk</th><th>Team</th><th class="text-end">Players</th><th class="text-end">2026 Picks</th><th class="text-end">Future Picks</th><th class="text-end">Total</th></tr>
          </thead>
          <tbody>{pr_rows}</tbody>
        </table>
      </div>
      <h6 class="mb-3">Roster Detail</h6>
      <div class="accordion">{accordion_items}</div>
    </div>

    <div class="tab-pane fade" id="tab-all-players">
      <p class="text-secondary small mb-2">Sorted by Model. <strong>Δ%</strong> = how much our model differs from raw KTC market value.</p>
      <div class="btn-group btn-group-sm mb-3" role="group">
        <button type="button" class="btn btn-outline-secondary active pos-filter" onclick="filterPos(this,'ALL')">All</button>
        <button type="button" class="btn btn-outline-secondary pos-filter" onclick="filterPos(this,'QB')">QB</button>
        <button type="button" class="btn btn-outline-secondary pos-filter" onclick="filterPos(this,'RB')">RB</button>
        <button type="button" class="btn btn-outline-secondary pos-filter" onclick="filterPos(this,'WR')">WR</button>
        <button type="button" class="btn btn-outline-secondary pos-filter" onclick="filterPos(this,'TE')">TE</button>
      </div>
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle" id="players-table">
          <thead class="table-secondary text-dark">
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>Team</th><th>Age</th><th class="text-end">Model</th><th class="text-end">Pos Adj</th><th class="text-end">KTC</th><th class="text-end">Δ%</th><th>Owner</th></tr>
          </thead>
          <tbody>{all_player_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-rookies">
      <p class="text-secondary small mb-3">Top 40 rookies sorted by KTC. All three values shown for comparison.</p>
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle">
          <thead class="table-secondary text-dark">
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>Team</th><th>Age</th><th class="text-end">Model</th><th class="text-end">Pos Adj</th><th class="text-end">KTC</th><th>Status</th></tr>
          </thead>
          <tbody>{rookie_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-picks">
      <p class="text-secondary small mb-3">2026 rookie draft pick ownership after all trades.</p>
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle">
          <thead class="table-secondary text-dark">
            <tr><th>Pick</th><th>Current Owner</th><th>Original Owner</th></tr>
          </thead>
          <tbody>{pick_rows}</tbody>
        </table>
      </div>
    </div>

  </div>
</div>

<footer class="text-center text-secondary small py-4 mt-5">
  Data: KeepTradeCut + Sleeper API &nbsp;&middot;&nbsp; Model: nfl_data_py 2021&ndash;2025 + aging curves
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
function filterPos(btn, pos) {{
  document.querySelectorAll('#players-table tbody tr').forEach(function(tr) {{
    tr.style.display = (pos === 'ALL' || tr.dataset.pos === pos) ? '' : 'none';
  }});
  document.querySelectorAll('.pos-filter').forEach(function(b) {{
    b.classList.toggle('active', b === btn);
  }});
}}
</script>
</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML written → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. KTC (native SF+TEP values, scraped live)
    ktc_raw = fetch_ktc()
    source  = ktc_raw[0].get("source", "KTC") if ktc_raw else "KTC"
    if source == "FC":
        ktc_raw = apply_tep(ktc_raw)
        ktc_raw = add_positional_ranks(ktc_raw)

    # 2. Load historical stats once; build scoring curve for positional adjustment
    df        = load_stats_data()
    curve_pts = build_scoring_curve(df)

    # 3. Positional scarcity adjustment (used by both Pos Adj and Model columns)
    print("Building positional adjustments…")
    adjustments = build_position_adjustments(ktc_raw, curve_pts)

    # 4. Per-player stats baseline + rookie imputation (for from-scratch model)
    stats_baseline    = build_stats_baseline(df)
    rookie_imputation = build_rookie_imputation(curve_pts)

    # 5. Sleeper
    users, rosters, traded_picks, draft_info, all_players = fetch_sleeper()

    # 6. Maps
    roster_to_owner, player_on_roster = build_maps(users, rosters)

    # 7. Build all three value columns
    players = match_players(ktc_raw, all_players, player_on_roster)
    players = apply_position_adjustment(players, adjustments)          # → adj_value
    players = compute_model_values(players, stats_baseline, rookie_imputation)  # → model_value
    players = apply_model_positional_adjustment(players, adjustments)  # → model_adj

    # 8. Pick boards
    pick_board     = build_pick_board(draft_info, traded_picks)
    pick_value_map = build_pick_value_map(ktc_raw)
    team_picks     = build_team_picks(draft_info, traded_picks, pick_value_map)

    # 9. Console output
    print()
    print_roster_overview(rosters, all_players, players, roster_to_owner,
                          team_picks=team_picks, my_team=MY_TEAM)
    print()
    print_rookie_board(players, roster_to_owner)
    print()
    print_pick_board(pick_board, roster_to_owner)
    print()

    # 10. HTML output → docs/index.html
    updated_at       = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    team_totals      = compute_team_totals(rosters, all_players, players, roster_to_owner, team_picks)
    rookie_data      = get_rookie_list(players, roster_to_owner)
    all_players_data = get_all_players_data(players, roster_to_owner)
    render_html(team_totals, rookie_data, all_players_data, pick_board, roster_to_owner, updated_at)
    print()


if __name__ == "__main__":
    main()
