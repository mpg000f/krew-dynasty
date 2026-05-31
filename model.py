"""
Krew Dynasty — Dynasty Model
Pulls live data from Sleeper + KeepTradeCut and outputs:
  1. 2026 rookie draft board (availability + KTC value)
  2. 2026 pick ownership board (including traded picks)
  3. Roster overview by team (dynasty value totals)

Scoring: 2QB/SUPERFLEX | 0.5 PPR | 0.25 TEP
"""

import os
import sys
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
TEP_BONUS  = 0.25  # TEs: 0.5 base + 0.25 TEP = 0.75 PPR; already in KTC tep values

SLEEPER = "https://api.sleeper.app/v1"

# Starters per position across all 10 teams.
# Standard 10-team 2QB/SF: 1QB+1SF=2QB, 2RB+½FLEX=25RB, 2WR+½FLEX=25WR, 1TE=10TE
# This league:              2QB=20QB,    3RB+~1.2FLEX=42RB, 4WR+~0.8FLEX=48WR, 1TE=10TE
STANDARD_STARTERS = {"QB": 20, "RB": 25, "WR": 25, "TE": 10}
LEAGUE_STARTERS   = {"QB": 20, "RB": 42, "WR": 48, "TE": 10}

STAT_YEARS = list(range(2020, 2025))

MY_TEAM    = "pltiii"
OUTPUT_DIR = "docs"

# ── Helpers ───────────────────────────────────────────────────────────────────
def get(url, **kwargs):
    r = requests.get(url, timeout=25, **kwargs)
    r.raise_for_status()
    return r.json()


def normalize(name):
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


# ── Positional Adjustment (additive VORP) ────────────────────────────────────
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


def build_position_adjustments(ktc_players):
    """
    Compute an additive KTC bonus for each positional rank based on how much
    deeper this league's starter pool is vs standard SF.

    Concept
    -------
    KTC calibrates values relative to a standard replacement level (e.g. WR25
    in a 10-team 2QB league).  Your league needs 48 WR starters, so WR25–WR48
    are genuine starters here but worthless in standard.

    Adjustment (additive, not multiplicative)
    -----------------------------------------
    For a player above BOTH replacement levels (rank ≤ standard_repl_rank):
        bonus = (pts[std_repl] - pts[league_repl]) × k
        → same absolute KTC bonus for every starter — small % for elite, larger % for mid-tier

    For a player in the "new starter" zone (standard_repl < rank ≤ league_repl):
        bonus = (pts[rank] - pts[league_repl]) × k
        → proportional to how far above your replacement level they sit

    For players below league replacement: bonus = 0

    k (KTC per fantasy point) is derived position-by-position from the slope
    of live KTC values vs the historical scoring curve.
    """
    print("Loading historical stats for positional adjustment (2020–2024)…")
    players_df = nfl.import_players()[["gsis_id", "display_name", "position"]].rename(
        columns={"gsis_id": "player_id", "display_name": "player_name"})
    stats_df = nfl.import_seasonal_data(STAT_YEARS, s_type="REG")
    df = stats_df.merge(players_df, on="player_id", how="left")
    df = df[df["position"].isin(POSITIONS) & (df["games"] >= 4)].copy()
    df["fpts"] = df.apply(_league_pts, axis=1)

    records = []
    for season, grp in df.groupby("season"):
        for pos, pgrp in grp.groupby("position"):
            ranked = pgrp.sort_values("fpts", ascending=False).reset_index(drop=True)
            ranked["pos_rank"] = range(1, len(ranked) + 1)
            records.append(ranked[["pos_rank", "position", "fpts"]])
    curve = (pd.concat(records)
               .groupby(["position", "pos_rank"])["fpts"]
               .mean()
               .reset_index()
               .rename(columns={"fpts": "avg_fpts"}))

    def curve_pts(pos, rank):
        sub = curve[(curve.position == pos) & (curve.pos_rank == rank)]
        if sub.empty:
            sub = curve[curve.position == pos]
            sub = sub.iloc[(sub["pos_rank"] - rank).abs().argsort()].head(1)
        return float(sub["avg_fpts"].values[0])

    # KTC value at each positional rank (from live data)
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

        # Estimate k: KTC per fantasy point, using ranks 3 and 15 as anchors.
        # Ranks well inside the starter zone give a stable slope estimate.
        hi_rank = min(3,  max(ktc_val.get(pos, {}).keys(), default=3))
        lo_rank = min(15, max(ktc_val.get(pos, {}).keys(), default=15))
        ktc_hi  = ktc_val.get(pos, {}).get(hi_rank)
        ktc_lo  = ktc_val.get(pos, {}).get(lo_rank)
        if ktc_hi and ktc_lo:
            pts_diff = curve_pts(pos, hi_rank) - curve_pts(pos, lo_rank)
            k = (ktc_hi - ktc_lo) / pts_diff if pts_diff > 0 else 35.0
        else:
            k = 35.0

        # Bonus for each positional rank
        bonuses = {}
        for rank in range(1, 71):
            pts = curve_pts(pos, rank)
            if pts >= std_repl_pts:
                extra = std_repl_pts - league_repl_pts      # constant for all clear starters
            elif pts >= league_repl_pts:
                extra = pts - league_repl_pts                # partial for marginal starters
            else:
                extra = 0
            bonuses[rank] = round(k * extra)

        adjustments[pos] = bonuses
        print(f"  {pos}: std_repl={std_repl_pts:.0f}pts, league_repl={league_repl_pts:.0f}pts, "
              f"k={k:.1f}, bonus@rank1={bonuses[1]}, bonus@std_repl={bonuses[std_repl_rank]}")

    return adjustments


def apply_position_adjustment(players, adjustments):
    """Add the positional bonus (in KTC units) to each player's raw KTC value."""
    for p in players:
        pos  = p.get("position", "")
        rank = p.get("pos_rank")
        if pos not in adjustments or rank is None or p.get("is_pick"):
            p["adj_value"]  = p["value"]
            p["adj_factor"] = 1.0
            continue
        max_rank = max(adjustments[pos].keys())
        bonus = adjustments[pos].get(min(int(rank), max_rank), 0)
        p["adj_value"]  = p["value"] + bonus
        p["adj_factor"] = round(p["adj_value"] / p["value"], 3) if p["value"] else 1.0
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
    KTC serves a var playersArray = [...] inline in the page.
    Scores: superflexValues.tep.value (native SF + 0.25 TEP).
    Returns a list of dicts with keys: name, position, team, age, value, rookie, years_exp
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
    """
    Parse KTC playersArray.
    Uses superflexValues.tep.value = Superflex + 0.25 TEP (matches this league exactly).
    """
    out = []
    for p in raw:
        pos = (p.get("position") or "").upper()
        is_pick = pos == "RDP"
        if pos not in POSITIONS and not is_pick:
            continue

        name = p.get("playerName", "")
        team = p.get("team") or "FA"
        age  = p.get("age")

        # Superflex + TEP value (native in KTC — no manual adjustment needed)
        sf_vals  = p.get("superflexValues", {})
        tep_vals = sf_vals.get("tep", {})
        value    = tep_vals.get("value") or sf_vals.get("value") or 0
        rank     = tep_vals.get("rank") or sf_vals.get("rank") or 999

        out.append({
            "name":      name,
            "position":  "PICK" if is_pick else pos,
            "team":      team,
            "age":       age,
            "value":     int(value),
            "rookie":    bool(p.get("rookie")),
            "years_exp": int(p.get("seasonsExperience") or 0),
            "is_pick":   is_pick,
            "ktc_rank":  rank,
            "source":    "KTC",
        })
    return out


def add_positional_ranks(players):
    """
    Assign pos_rank (rank within position by KTC value) to each player.
    This is distinct from ktc_rank (overall dynasty rank) and is what the
    VORP adjustment factors are indexed by.
    """
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
    """Fallback: fetch from Fantasy Calc (2QB, 0.5 PPR, 0.25 TEP).
    FC wraps player info under a 'player' key and includes sleeperId directly.
    """
    print("Fetching dynasty values from Fantasy Calc (fallback)…")
    url = (
        "https://api.fantasycalc.com/values/current"
        "?isDynasty=true&numQbs=2&ppr=0.5&tep=0.25"
    )
    data = get(url)
    out = []
    for row in data:
        p    = row.get("player", {})
        pos  = (p.get("position") or "").upper()
        age  = p.get("maybeAge")
        is_pick = pos == "PICK" or not pos
        if pos not in POSITIONS and not is_pick:
            continue
        out.append({
            "name":       p.get("name", ""),
            "position":   pos,
            "team":       p.get("maybeTeam") or "FA",
            "age":        age,
            "value":      int(row.get("value", 0)),
            "rookie":     int(p.get("maybeYoe") or 99) == 0,
            "years_exp":  int(p.get("maybeYoe") or 0),
            "is_pick":    is_pick,
            "sleeper_id": str(p.get("sleeperId")) if p.get("sleeperId") else None,
            "source":     "FC",
        })
    return out


# ── Apply TEP Adjustment ──────────────────────────────────────────────────────
def apply_tep(players):
    """
    Boost TE values for TEP scoring. KTC doesn't natively support TEP,
    so we add a linear adjustment based on estimated reception counts.
    Top TEs get a larger boost than low-end TEs.
    """
    te_players = [(i, p) for i, p in enumerate(players) if p["position"] == "TE"]
    if not te_players:
        return players

    # Estimate receptions by rank within TE (top TE ≈ 100 rec, bottom ≈ 30)
    n = len(te_players)
    for rank, (i, p) in enumerate(sorted(te_players, key=lambda x: -x[1]["value"])):
        est_rec = max(30, 100 - int(70 * rank / max(n - 1, 1)))
        season_bonus = TEP_BONUS * est_rec
        # Translate to KTC-scale: rough heuristic ~5 value points per point of scoring
        value_boost = int(season_bonus * 5)
        players[i]["value"] += value_boost
        players[i]["tep_boost"] = value_boost

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
        rid = r["roster_id"]
        for pid in (r.get("players") or []):
            player_on_roster[pid] = rid

    return roster_to_owner, player_on_roster


# ── 2026 Draft Pick Board ─────────────────────────────────────────────────────
def build_pick_board(draft_info, traded_picks):
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    n_rounds = draft_info["settings"]["rounds"]
    n_teams  = draft_info["settings"]["teams"]

    # Linear draft: same slot order every round
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    board = {}
    for rnd in range(1, n_rounds + 1):
        for slot, roster_id in enumerate(pick_order, start=1):
            board[(rnd, slot)] = {
                "round":           rnd,
                "slot":            slot,
                "pick":            f"{rnd}.{slot:02d}",
                "original_roster": roster_id,
                "current_owner":   roster_id,
            }

    # Apply traded picks for this season only
    for t in traded_picks:
        if t["season"] != SEASON:
            continue
        orig     = t["roster_id"]
        new_own  = t["owner_id"]
        rnd      = t["round"]
        for slot, rid in enumerate(pick_order, start=1):
            if rid == orig and (rnd, slot) in board:
                board[(rnd, slot)]["current_owner"] = new_own
                break

    return sorted(board.values(), key=lambda x: (x["round"], x["slot"]))


# ── Pick Value Helpers ───────────────────────────────────────────────────────
def slot_to_bucket(slot, n_teams=10):
    """Map draft slot to KTC tier (Early / Mid / Late)."""
    third = n_teams / 3
    if slot <= third:
        return "Early"
    elif slot <= 2 * third:
        return "Mid"
    return "Late"


def build_pick_value_map(ktc_raw):
    """Extract KTC pick values keyed by label e.g. '2026 Early 1st'."""
    rnd_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
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
    """
    Build a dict of roster_id → list of picks owned (all seasons in traded_picks).
    For 2026, uses exact slot → bucket.  For future years, uses 'Mid' as estimate.
    """
    seasons  = sorted(set([SEASON] + [t["season"] for t in traded_picks]))
    n_rounds = draft_info["settings"]["rounds"]
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    # owner_of[(season, round, orig_roster)] = current_owner_roster_id
    owner_of = {}
    for season in seasons:
        for rnd in range(1, n_rounds + 1):
            for rid in range(1, n_teams + 1):
                owner_of[(season, rnd, rid)] = rid  # each team owns their own picks

    for t in traded_picks:
        owner_of[(t["season"], t["round"], t["roster_id"])] = t["owner_id"]

    rnd_label = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    team_picks = {rid: [] for rid in range(1, n_teams + 1)}

    for (season, rnd, orig_roster), current_owner in sorted(owner_of.items()):
        # Determine bucket
        if season == SEASON:
            try:
                slot = pick_order.index(orig_roster) + 1
                bucket = slot_to_bucket(slot, n_teams)
            except ValueError:
                bucket = "Mid"
        else:
            bucket = "Mid"  # future picks: unknown standing → use mid estimate

        label = f"{season} {bucket} {rnd_label.get(rnd, f'{rnd}th')}"
        val   = pick_value_map.get((season, rnd, bucket), 0)

        team_picks[current_owner].append({
            "label":   label,
            "season":  season,
            "round":   rnd,
            "orig":    orig_roster,
            "value":   val,
        })

    return team_picks


# ── Match Value Source to Sleeper Players ────────────────────────────────────
def match_players(value_players, all_players, player_on_roster):
    # Primary: use sleeper_id baked into FC/KTC data
    # Secondary: fall back to normalized name match
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
        # Try direct Sleeper ID first (FC provides this)
        sid = p.get("sleeper_id")
        if sid and sid not in all_players:
            sid = None
        # Fall back to name match
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


# ── Output ────────────────────────────────────────────────────────────────────
W = 72

def divider(title=""):
    if title:
        pad = (W - len(title) - 4) // 2
        print("=" * pad + f"  {title}  " + "=" * (W - pad - len(title) - 4))
    else:
        print("=" * W)


def print_rookie_board(players, roster_to_owner, pick_board):
    src = players[0].get("source", "KTC") if players else "KTC"
    val_label = f"{src} SF+TEP" if src == "KTC" else f"{src} SF+TEP(adj)"
    divider(f"2026 ROOKIE DRAFT BOARD — Krew Dynasty")
    print(f"  Scoring: 2QB/SF  |  0.5 PPR  |  {TEP_BONUS} TEP  |  Values: {val_label}")
    divider()

    rookies = [
        p for p in players
        if p.get("years_exp") == 0 or p.get("rookie")
    ]
    rookies = sorted(rookies, key=lambda x: -x["value"])

    print(f"  {'RK':<4} {'NAME':<24} {'POS':<4} {'TEAM':<5} {'AGE':<6} {'ADJ':>6} {'KTC':>6}  STATUS")
    divider()
    for rk, p in enumerate(rookies, 1):
        rid    = p.get("on_roster_id")
        owner  = roster_to_owner.get(rid, "") if rid else ""
        status = f"STASHED — {owner}" if owner else "AVAILABLE"
        age_s  = f"{p['age']:.1f}" if (p.get("age") and p["age"] > 0) else "  ?"
        adj    = p.get("adj_value", p["value"])
        raw    = p["value"]
        print(
            f"  {rk:<4} {p['name']:<24} {p['position']:<4} {p['team']:<5} "
            f"{age_s:<6} {adj:>6} {raw:>6}  {status}"
        )
        if rk >= 40:
            break


def print_pick_board(pick_board, roster_to_owner):
    divider("2026 DRAFT PICK OWNERSHIP")
    print(f"  {'PICK':<8} {'CURRENT OWNER':<28} ORIGINAL OWNER")
    divider()
    for p in pick_board:
        orig  = roster_to_owner.get(p["original_roster"], f"R{p['original_roster']}")
        curr  = roster_to_owner.get(p["current_owner"],   f"R{p['current_owner']}")
        traded = "  ← TRADED" if p["original_roster"] != p["current_owner"] else ""
        print(f"  {p['pick']:<8} {curr:<28} {orig}{traded}")


def print_roster_overview(rosters, all_players, players_matched, roster_to_owner,
                          team_picks=None, my_team=None):
    # Use adj_value (league-adjusted) with fallback to raw value
    adj_by_sid  = {p["sleeper_id"]: p["adj_value"] for p in players_matched if p.get("sleeper_id")}
    adj_by_name = {normalize(p["name"]): p["adj_value"] for p in players_matched}
    raw_by_sid  = {p["sleeper_id"]: p["value"] for p in players_matched if p.get("sleeper_id")}
    raw_by_name = {normalize(p["name"]): p["value"] for p in players_matched}

    def get_vals(pid):
        name = all_players.get(pid, {}).get("full_name", "")
        key  = normalize(name)
        adj  = adj_by_sid.get(pid) or adj_by_name.get(key, 0)
        raw  = raw_by_sid.get(pid) or raw_by_name.get(key, 0)
        return adj, raw

    # Compute totals for every team
    team_totals = []
    for r in rosters:
        rid   = r["roster_id"]
        owner = roster_to_owner[rid]
        pids  = [pid for pid in (r.get("players") or [])
                 if all_players.get(pid, {}).get("position") in POSITIONS]

        player_adj = sum(get_vals(pid)[0] for pid in pids)
        picks_26   = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] == SEASON)
        picks_fut  = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] != SEASON)
        total      = player_adj + picks_26 + picks_fut
        team_totals.append((rid, owner, player_adj, picks_26, picks_fut, total))

    team_totals.sort(key=lambda x: -x[5])

    # ── Power Rankings ────────────────────────────────────────────────────────
    divider("DYNASTY POWER RANKINGS  (league-adjusted KTC SF+TEP)")
    print("  Values adjusted for 3RB/4WR/2FLEX roster vs standard SF baseline.")
    print(f"  {'RK':<4} {'TEAM':<28} {'PLAYERS':>8} {'26 PICKS':>9} {'FUT PICKS':>10} {'TOTAL':>8}")
    divider()
    for rk, (rid, owner, pv, p26, pf, tot) in enumerate(team_totals, 1):
        marker = " ◄ YOU" if owner == my_team else ""
        print(f"  {rk:<4} {owner:<28} {pv:>8,} {p26:>9,} {pf:>10,} {tot:>8,}{marker}")

    # ── Per-Team Detail ───────────────────────────────────────────────────────
    divider("ROSTER DETAIL")
    for rid, owner, pv, p26, pf, tot in team_totals:
        pid_list  = next(r.get("players") or [] for r in rosters if r["roster_id"] == rid)
        team_rows = []
        for pid in pid_list:
            p    = all_players.get(pid, {})
            pos  = p.get("position", "")
            if pos not in POSITIONS:
                continue
            name = p.get("full_name", pid)
            adj, raw = get_vals(pid)
            team_rows.append((name, pos, adj, raw))
        team_rows.sort(key=lambda x: -x[2])

        marker = "  ◄ YOU" if owner == my_team else ""
        print(f"\n  {owner}{marker}")
        print(f"  Players: {pv:,}  |  2026 Picks: {p26:,}  |  Future Picks: {pf:,}  |  Total: {tot:,}")
        print(f"  {'POS':<4} {'NAME':<26} {'ADJ':>6} {'KTC':>6} {'MULT':>6}")
        for name, pos, adj, raw in team_rows[:12]:
            mult = f"{adj/raw:.2f}x" if raw else "  -"
            unranked = "  (unranked)" if raw == 0 else ""
            print(f"  {pos:<4} {name:<26} {adj:>6} {raw:>6} {mult:>6}{unranked}")
        if len(team_rows) > 12:
            print(f"       … +{len(team_rows) - 12} more players")

        picks = sorted((team_picks or {}).get(rid, []), key=lambda x: (-int(x["season"]), x["round"]))
        if picks:
            print(f"  {'PICK':<32} KTC")
            for pk in picks:
                print(f"  {pk['label']:<32} {pk['value']:>5}")


# ── HTML Rendering ────────────────────────────────────────────────────────────
def compute_team_totals(rosters, all_players, players_matched, roster_to_owner, team_picks):
    adj_by_sid  = {p["sleeper_id"]: p["adj_value"] for p in players_matched if p.get("sleeper_id")}
    adj_by_name = {normalize(p["name"]): p["adj_value"] for p in players_matched}
    raw_by_sid  = {p["sleeper_id"]: p["value"] for p in players_matched if p.get("sleeper_id")}
    raw_by_name = {normalize(p["name"]): p["value"] for p in players_matched}

    def get_vals(pid):
        name = all_players.get(pid, {}).get("full_name", "")
        adj  = adj_by_sid.get(pid) or adj_by_name.get(normalize(name), 0)
        raw  = raw_by_sid.get(pid) or raw_by_name.get(normalize(name), 0)
        return adj, raw

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
            adj, raw = get_vals(pid)
            player_rows.append({"name": p.get("full_name", pid), "pos": pos, "adj": adj, "raw": raw})
        player_rows.sort(key=lambda x: -x["adj"])

        player_adj = sum(x["adj"] for x in player_rows)
        picks_26   = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] == SEASON)
        picks_fut  = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] != SEASON)
        picks_list = sorted((team_picks or {}).get(rid, []), key=lambda x: (-int(x["season"]), x["round"]))

        result.append({
            "rid": rid, "owner": owner,
            "player_adj": player_adj, "picks_26": picks_26, "picks_fut": picks_fut,
            "total": player_adj + picks_26 + picks_fut,
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
            "adj": p.get("adj_value", p["value"]), "raw": p["value"],
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
        result.append({
            "name":   p["name"],
            "pos":    p.get("position", ""),
            "team":   p.get("team", "FA"),
            "age":    p.get("age"),
            "adj":    p.get("adj_value", p["value"]),
            "raw":    p["value"],
            "factor": p.get("adj_factor", 1.0),
            "owner":  owner,
        })
    result.sort(key=lambda x: -x["adj"])
    return result


def render_html(team_totals, rookies, all_players_data, pick_board, roster_to_owner, updated_at):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def fmt(n):
        return f"{n:,}"

    # ── Power Rankings rows ──────────────────────────────────────────────────
    pr_rows = ""
    for rk, t in enumerate(team_totals, 1):
        is_me = t["owner"] == MY_TEAM
        row_class = ' class="you-row"' if is_me else ""
        you = " ◄ YOU" if is_me else ""
        pr_rows += (
            f'<tr{row_class}>'
            f'<td>{rk}</td><td><strong>{t["owner"]}</strong>{you}</td>'
            f'<td class="text-end">{fmt(t["player_adj"])}</td>'
            f'<td class="text-end">{fmt(t["picks_26"])}</td>'
            f'<td class="text-end">{fmt(t["picks_fut"])}</td>'
            f'<td class="text-end fw-bold">{fmt(t["total"])}</td>'
            f'</tr>\n'
        )

    # ── Roster accordion ────────────────────────────────────────────────────
    accordion_items = ""
    for rk, t in enumerate(team_totals, 1):
        is_me   = t["owner"] == MY_TEAM
        you     = " ◄ YOU" if is_me else ""
        p_rows  = ""
        for p in t["players"][:16]:
            mult    = f'{p["adj"]/p["raw"]:.2f}x' if p["raw"] else "—"
            p_rows += (
                f'<tr><td>{p["pos"]}</td><td>{p["name"]}</td>'
                f'<td class="text-end">{fmt(p["adj"])}</td>'
                f'<td class="text-end">{fmt(p["raw"])}</td>'
                f'<td class="text-end text-secondary">{mult}</td></tr>\n'
            )
        if len(t["players"]) > 16:
            p_rows += f'<tr><td colspan="5" class="text-secondary small">… +{len(t["players"]) - 16} more</td></tr>'

        pk_rows = "".join(
            f'<tr><td>{pk["label"]}</td><td class="text-end">{fmt(pk["value"])}</td></tr>'
            for pk in t["picks"]
        )
        picks_section = (
            f'<h6 class="mt-3 mb-1">Draft Picks</h6>'
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
                Players {fmt(t["player_adj"])} &middot; Picks {fmt(t["picks_26"] + t["picks_fut"])} &middot; Total {fmt(t["total"])}
              </span>
            </button>
          </h2>
          <div id="team-{rk}" class="accordion-collapse collapse">
            <div class="accordion-body p-2">
              <table class="table table-sm mb-0">
                <thead><tr><th>Pos</th><th>Name</th><th class="text-end">Adj</th><th class="text-end">KTC</th><th class="text-end">Mult</th></tr></thead>
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
        mult  = f'{r["adj"]/r["raw"]:.2f}x' if r["raw"] else "—"
        rookie_rows += (
            f'<tr><td>{r["rank"]}</td><td>{r["name"]}</td><td>{r["pos"]}</td>'
            f'<td>{r["team"]}</td><td>{age_s}</td>'
            f'<td class="text-end">{fmt(r["adj"])}</td>'
            f'<td class="text-end">{fmt(r["raw"])}</td>'
            f'<td class="text-end text-secondary">{mult}</td>'
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
        mult   = f'{p["factor"]:.2f}x' if p["raw"] else "—"
        owner  = p["owner"] or '<span class="text-secondary">—</span>'
        is_me  = p["owner"] == MY_TEAM
        row_cls = ' class="you-row"' if is_me else ""
        all_player_rows += (
            f'<tr data-pos="{p["pos"]}"{row_cls}>'
            f'<td>{rk}</td><td>{p["name"]}</td><td>{p["pos"]}</td>'
            f'<td>{p["team"]}</td><td>{age_s}</td>'
            f'<td class="text-end">{fmt(p["adj"])}</td>'
            f'<td class="text-end">{fmt(p["raw"])}</td>'
            f'<td class="text-end text-secondary">{mult}</td>'
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
  <ul class="nav nav-tabs mb-4" id="tabs">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-rankings" type="button">Power Rankings</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-all-players" type="button">All Players</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-rookies" type="button">Rookie Board</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-picks" type="button">2026 Pick Ownership</button></li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-rankings">
      <p class="text-secondary small mb-3">Values adjusted for 3RB/4WR/2FLEX roster construction vs standard SF baseline, using 2020&ndash;2024 historical scoring data.</p>
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
      <p class="text-secondary small mb-2">All players ranked by league-adjusted value. <em>Adj</em> = KTC &times; positional multiplier based on 3RB/4WR/2FLEX roster construction.</p>
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
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>NFL Team</th><th>Age</th><th class="text-end">Adj</th><th class="text-end">KTC</th><th class="text-end">Mult</th><th>Owner</th></tr>
          </thead>
          <tbody>{all_player_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-rookies">
      <p class="text-secondary small mb-3">Top 40 rookies by KTC SF+TEP. <em>Adj</em> = league-adjusted value, <em>Mult</em> = adj/raw factor.</p>
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle">
          <thead class="table-secondary text-dark">
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>NFL Team</th><th>Age</th><th class="text-end">Adj</th><th class="text-end">KTC</th><th class="text-end">Mult</th><th>Status</th></tr>
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
  Data: KeepTradeCut + Sleeper API &nbsp;&middot;&nbsp; Historical adjustment: nfl_data_py 2020&ndash;2024
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
    # 1. KTC (native SF+TEP values)
    ktc_raw = fetch_ktc()
    source  = ktc_raw[0].get("source", "KTC") if ktc_raw else "KTC"
    if source == "FC":
        ktc_raw = apply_tep(ktc_raw)
        ktc_raw = add_positional_ranks(ktc_raw)

    # 2. Positional adjustment: additive VORP bonus from historical scoring data
    adjustments = build_position_adjustments(ktc_raw)

    # 3. Sleeper
    users, rosters, traded_picks, draft_info, all_players = fetch_sleeper()

    # 4. Maps
    roster_to_owner, player_on_roster = build_maps(users, rosters)

    # 5. Match KTC → Sleeper, apply additive positional adjustment
    players = match_players(ktc_raw, all_players, player_on_roster)
    players = apply_position_adjustment(players, adjustments)

    # 6. Pick boards
    pick_board     = build_pick_board(draft_info, traded_picks)
    pick_value_map = build_pick_value_map(ktc_raw)
    team_picks     = build_team_picks(draft_info, traded_picks, pick_value_map)

    # 6. Console output
    print()
    print_roster_overview(rosters, all_players, players, roster_to_owner,
                          team_picks=team_picks, my_team=MY_TEAM)
    print()
    print_rookie_board(players, roster_to_owner, pick_board)
    print()
    print_pick_board(pick_board, roster_to_owner)
    print()

    # 7. HTML output → docs/index.html
    updated_at       = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    team_totals      = compute_team_totals(rosters, all_players, players, roster_to_owner, team_picks)
    rookie_data      = get_rookie_list(players, roster_to_owner)
    all_players_data = get_all_players_data(players, roster_to_owner)
    render_html(team_totals, rookie_data, all_players_data, pick_board, roster_to_owner, updated_at)
    print()


if __name__ == "__main__":
    main()
