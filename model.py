"""
Krew Dynasty — Dynasty Model
Pulls live data from Sleeper + KeepTradeCut and outputs:
  1. Roster power rankings (dynasty value totals)
  2. All-player rankings (league-adjusted KTC value)
  3. 2026 rookie draft board
  4. 2026 pick ownership board

Values: KTC SF+TEP with a modest flat multiplier on RB/WR to reflect deeper
roster requirements (3RB/4WR/2FLEX) vs a standard SF league.
Scoring: 2QB/SUPERFLEX | 0.5 PPR | 0.25 TEP
"""

import os
import json
import re
import requests
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

# Modest flat multiplier on RB/WR to reflect deeper roster requirements
# (3RB/4WR/2FLEX with 10 teams) vs the standard SF league KTC calibrates for.
POS_MULTIPLIER = {"QB": 1.0, "RB": 1.06, "WR": 1.08, "TE": 1.0}

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


# ── Positional Adjustment (flat multiplier) ───────────────────────────────────
def apply_position_adjustment(players):
    """Apply flat POS_MULTIPLIER to RB/WR; all other positions stay at 1.0x."""
    for p in players:
        if p.get("is_pick"):
            p["adj_value"] = p["value"]
            continue
        mult = POS_MULTIPLIER.get(p.get("position", ""), 1.0)
        p["adj_value"] = round(p["value"] * mult)
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
    """Fetch dynasty values from KeepTradeCut (superflexValues.tep.value)."""
    url = f"https://keeptradecut.com/dynasty-rankings?filters=QB|WR|RB|TE|RDP&format={KTC_FORMAT}"
    print("Fetching KTC dynasty rankings…")
    try:
        r = requests.get(url, headers=KTC_HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  Warning: KTC failed ({e}). Falling back to Fantasy Calc.")
        return fetch_fantasycalc()

    soup = BeautifulSoup(r.text, "html.parser")
    for script in soup.find_all("script"):
        text = script.string or ""
        if "playersArray" not in text:
            continue
        m = re.search(r"var playersArray\s*=\s*(\[.*?\]);", text, re.DOTALL)
        if m:
            try:
                result = parse_ktc_players(json.loads(m.group(1)))
                result = add_positional_ranks(result)
                print(f"  Loaded {len(result)} players from KTC.")
                return result
            except Exception as e:
                print(f"  KTC parse error: {e}. Falling back to Fantasy Calc.")
                return fetch_fantasycalc()

    print("  Could not find KTC playersArray. Falling back to Fantasy Calc.")
    return fetch_fantasycalc()


def parse_ktc_players(raw):
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
    """Assign pos_rank within each position by KTC value (1 = highest)."""
    by_pos = {}
    for i, p in enumerate(players):
        pos = p.get("position", "")
        if pos in POSITIONS:
            by_pos.setdefault(pos, []).append(i)
    for pos, indices in by_pos.items():
        for rank, idx in enumerate(sorted(indices, key=lambda i: -players[i]["value"]), 1):
            players[idx]["pos_rank"] = rank
    return players


def fetch_fantasycalc():
    """Fallback: Fantasy Calc (2QB, 0.5 PPR, 0.25 TEP)."""
    print("Fetching from Fantasy Calc (fallback)…")
    url  = "https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=2&ppr=0.5&tep=0.25"
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
        players[i]["value"] += int(TEP_BONUS * max(30, 100 - int(70 * rank / max(n-1, 1))) * 5)
    return players


# ── Sleeper ───────────────────────────────────────────────────────────────────
def fetch_sleeper():
    print("Fetching Sleeper league data…")
    users        = get(f"{SLEEPER}/league/{LEAGUE_ID}/users")
    rosters      = get(f"{SLEEPER}/league/{LEAGUE_ID}/rosters")
    traded_picks = get(f"{SLEEPER}/league/{LEAGUE_ID}/traded_picks")
    draft_info   = get(f"{SLEEPER}/draft/{DRAFT_ID}")
    print("Fetching Sleeper player database…")
    all_players  = get(f"{SLEEPER}/players/nfl")
    return users, rosters, traded_picks, draft_info, all_players


def build_maps(users, rosters):
    user_by_id = {u["user_id"]: u for u in users}
    roster_to_owner = {}
    for r in rosters:
        u = user_by_id.get(r["owner_id"], {})
        roster_to_owner[r["roster_id"]] = (
            u.get("team_name") or u.get("display_name") or f"Roster {r['roster_id']}"
        )
    player_on_roster = {}
    for r in rosters:
        for pid in (r.get("players") or []):
            player_on_roster[pid] = r["roster_id"]
    return roster_to_owner, player_on_roster


# ── Pick Boards ───────────────────────────────────────────────────────────────
def build_pick_board(draft_info, traded_picks):
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    n_rounds   = draft_info["settings"]["rounds"]
    n_teams    = draft_info["settings"]["teams"]
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    board = {}
    for rnd in range(1, n_rounds + 1):
        for slot, rid in enumerate(pick_order, 1):
            board[(rnd, slot)] = {
                "round": rnd, "slot": slot, "pick": f"{rnd}.{slot:02d}",
                "original_roster": rid, "current_owner": rid,
            }
    for t in traded_picks:
        if t["season"] != SEASON:
            continue
        for slot, rid in enumerate(pick_order, 1):
            if rid == t["roster_id"] and (t["round"], slot) in board:
                board[(t["round"], slot)]["current_owner"] = t["owner_id"]
                break
    return sorted(board.values(), key=lambda x: (x["round"], x["slot"]))


def slot_to_bucket(slot, n_teams=10):
    third = n_teams / 3
    return "Early" if slot <= third else ("Mid" if slot <= 2 * third else "Late")


def build_pick_value_map(ktc_raw):
    rnd_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    vals    = {}
    for p in ktc_raw:
        if not p.get("is_pick"):
            continue
        parts = p["name"].split()
        if len(parts) == 3:
            year, bucket, rnd_str = parts
            rnd = rnd_map.get(rnd_str)
            if rnd:
                vals[(year, rnd, bucket)] = p["value"]
    return vals


def build_team_picks(draft_info, traded_picks, pick_value_map, n_teams=10):
    seasons    = sorted(set([SEASON] + [t["season"] for t in traded_picks]))
    n_rounds   = draft_info["settings"]["rounds"]
    slot_to_roster = {int(k): v for k, v in draft_info["slot_to_roster_id"].items()}
    pick_order = [slot_to_roster[i] for i in range(1, n_teams + 1)]

    owner_of = {(s, r, rid): rid
                for s in seasons for r in range(1, n_rounds+1) for rid in range(1, n_teams+1)}
    for t in traded_picks:
        owner_of[(t["season"], t["round"], t["roster_id"])] = t["owner_id"]

    rnd_label  = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    team_picks = {rid: [] for rid in range(1, n_teams + 1)}
    for (season, rnd, orig), owner in sorted(owner_of.items()):
        if season == SEASON:
            try:
                bucket = slot_to_bucket(pick_order.index(orig) + 1, n_teams)
            except ValueError:
                bucket = "Mid"
        else:
            bucket = "Mid"
        label = f"{season} {bucket} {rnd_label.get(rnd, f'{rnd}th')}"
        team_picks[owner].append({
            "label": label, "season": season, "round": rnd,
            "orig": orig, "value": pick_value_map.get((season, rnd, bucket), 0),
        })
    return team_picks


# ── Match Players ─────────────────────────────────────────────────────────────
def match_players(value_players, all_players, player_on_roster):
    sleeper_by_name = {
        normalize(p.get("full_name", "")): pid
        for pid, p in all_players.items()
        if p.get("position") in POSITIONS and p.get("full_name")
    }
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
        matched.append({
            **p,
            "sleeper_id":   sid,
            "years_exp":    sp.get("years_exp"),
            "on_roster_id": player_on_roster.get(sid),
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


def print_roster_overview(rosters, all_players, players_matched, roster_to_owner,
                          team_picks=None, my_team=None):
    adj_by_sid  = {p["sleeper_id"]: p["adj_value"] for p in players_matched if p.get("sleeper_id")}
    adj_by_name = {normalize(p["name"]): p["adj_value"] for p in players_matched}
    ktc_by_sid  = {p["sleeper_id"]: p["value"]     for p in players_matched if p.get("sleeper_id")}
    ktc_by_name = {normalize(p["name"]): p["value"] for p in players_matched}

    def get_vals(pid):
        name = all_players.get(pid, {}).get("full_name", "")
        key  = normalize(name)
        return (adj_by_sid.get(pid) or adj_by_name.get(key, 0),
                ktc_by_sid.get(pid) or ktc_by_name.get(key, 0))

    team_totals = []
    for r in rosters:
        rid          = r["roster_id"]
        owner        = roster_to_owner[rid]
        pids         = [pid for pid in (r.get("players") or [])
                        if all_players.get(pid, {}).get("position") in POSITIONS]
        player_adj   = sum(get_vals(pid)[0] for pid in pids)
        picks_26     = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] == SEASON)
        picks_fut    = sum(p["value"] for p in (team_picks or {}).get(rid, []) if p["season"] != SEASON)
        team_totals.append((rid, owner, player_adj, picks_26, picks_fut, player_adj + picks_26 + picks_fut))

    team_totals.sort(key=lambda x: -x[5])

    divider("DYNASTY POWER RANKINGS  (league-adjusted KTC SF+TEP)")
    print("  RB 1.06x · WR 1.08x flat multiplier vs standard SF baseline.")
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
            adj, ktc = get_vals(pid)
            team_rows.append((p.get("full_name", pid), pos, adj, ktc))
        team_rows.sort(key=lambda x: -x[2])

        marker = "  ◄ YOU" if owner == my_team else ""
        print(f"\n  {owner}{marker}")
        print(f"  Players: {pv:,}  |  2026 Picks: {p26:,}  |  Future Picks: {pf:,}  |  Total: {tot:,}")
        print(f"  {'POS':<4} {'NAME':<26} {'ADJ':>7} {'KTC':>7} {'MULT':>6}")
        for name, pos, adj, ktc in team_rows[:12]:
            mult = f"{adj/ktc:.2f}x" if ktc else "—"
            print(f"  {pos:<4} {name:<26} {adj:>7} {ktc:>7} {mult:>6}")
        if len(team_rows) > 12:
            print(f"       … +{len(team_rows) - 12} more players")
        picks = sorted((team_picks or {}).get(rid, []), key=lambda x: (-int(x["season"]), x["round"]))
        if picks:
            print(f"  {'PICK':<32} KTC")
            for pk in picks:
                print(f"  {pk['label']:<32} {pk['value']:>5}")


def print_rookie_board(players, roster_to_owner):
    divider("2026 ROOKIE DRAFT BOARD — Krew Dynasty")
    print(f"  {'RK':<4} {'NAME':<24} {'POS':<4} {'TEAM':<5} {'AGE':<6} {'ADJ':>7} {'KTC':>7}  STATUS")
    divider()
    rookies = sorted(
        [p for p in players if p.get("years_exp") == 0 or p.get("rookie")],
        key=lambda x: -x["value"]
    )
    for rk, p in enumerate(rookies[:40], 1):
        rid    = p.get("on_roster_id")
        owner  = roster_to_owner.get(rid, "") if rid else ""
        status = f"STASHED — {owner}" if owner else "AVAILABLE"
        age_s  = f"{p['age']:.1f}" if (p.get("age") and p["age"] > 0) else "  ?"
        print(
            f"  {rk:<4} {p['name']:<24} {p['position']:<4} {p['team']:<5} "
            f"{age_s:<6} {p.get('adj_value', p['value']):>7} {p['value']:>7}  {status}"
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


# ── HTML Data Helpers ─────────────────────────────────────────────────────────
def compute_team_totals(rosters, all_players, players_matched, roster_to_owner, team_picks):
    adj_by_sid  = {p["sleeper_id"]: p["adj_value"] for p in players_matched if p.get("sleeper_id")}
    adj_by_name = {normalize(p["name"]): p["adj_value"] for p in players_matched}
    ktc_by_sid  = {p["sleeper_id"]: p["value"]     for p in players_matched if p.get("sleeper_id")}
    ktc_by_name = {normalize(p["name"]): p["value"] for p in players_matched}

    def get_vals(pid):
        name = all_players.get(pid, {}).get("full_name", "")
        key  = normalize(name)
        return (adj_by_sid.get(pid) or adj_by_name.get(key, 0),
                ktc_by_sid.get(pid) or ktc_by_name.get(key, 0))

    result = []
    for r in rosters:
        rid   = r["roster_id"]
        owner = roster_to_owner[rid]
        pids  = [pid for pid in (r.get("players") or [])
                 if all_players.get(pid, {}).get("position") in POSITIONS]

        player_rows = []
        for pid in pids:
            p   = all_players.get(pid, {})
            if p.get("position") not in POSITIONS:
                continue
            adj, ktc = get_vals(pid)
            player_rows.append({"name": p.get("full_name", pid), "pos": p["position"], "adj": adj, "ktc": ktc})
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
        adj   = p.get("adj_value", p["value"])
        ktc   = p["value"]
        result.append({
            "name":   p["name"],
            "pos":    p.get("position", ""),
            "team":   p.get("team", "FA"),
            "age":    p.get("age"),
            "adj":    adj,
            "raw":    ktc,
            "factor": round(adj / ktc, 3) if ktc else 1.0,
            "owner":  owner,
        })
    result.sort(key=lambda x: -x["adj"])
    return result


# ── HTML Rendering ────────────────────────────────────────────────────────────
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
        is_me  = t["owner"] == MY_TEAM
        you    = " ◄ YOU" if is_me else ""
        p_rows = ""
        for p in t["players"][:16]:
            mult    = f'{p["adj"]/p["ktc"]:.2f}x' if p["ktc"] else "—"
            p_rows += (
                f'<tr><td>{p["pos"]}</td><td>{p["name"]}</td>'
                f'<td class="text-end">{fmt(p["adj"])}</td>'
                f'<td class="text-end text-secondary">{fmt(p["ktc"])}</td>'
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
            f'<td class="text-end fw-bold">{fmt(r["adj"])}</td>'
            f'<td class="text-end text-secondary">{fmt(r["raw"])}</td>'
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
        age_s   = f'{p["age"]:.1f}' if p.get("age") and p["age"] > 0 else "?"
        mult    = f'{p["factor"]:.2f}x' if p["raw"] else "—"
        owner   = p["owner"] or '<span class="text-secondary">—</span>'
        is_me   = p["owner"] == MY_TEAM
        row_cls = ' class="you-row"' if is_me else ""
        all_player_rows += (
            f'<tr data-pos="{p["pos"]}"{row_cls}>'
            f'<td>{rk}</td><td>{p["name"]}</td><td>{p["pos"]}</td>'
            f'<td>{p["team"]}</td><td>{age_s}</td>'
            f'<td class="text-end fw-bold">{fmt(p["adj"])}</td>'
            f'<td class="text-end text-secondary">{fmt(p["raw"])}</td>'
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
  <ul class="nav nav-tabs mb-4" id="tabs">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-rankings" type="button">Power Rankings</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-all-players" type="button">All Players</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-rookies" type="button">Rookie Board</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-picks" type="button">2026 Pick Ownership</button></li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-rankings">
      <p class="text-secondary small mb-3">Values are KTC SF+TEP with a modest flat multiplier on RB (1.06&times;) and WR (1.08&times;) to reflect deeper roster requirements vs a standard SF league.</p>
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
      <p class="text-secondary small mb-2">All players sorted by league-adjusted value. <em>Adj</em> = KTC &times; positional multiplier (RB 1.06&times;, WR 1.08&times;). <em>Mult</em> = Adj&divide;KTC.</p>
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
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>Team</th><th>Age</th><th class="text-end">Adj</th><th class="text-end">KTC</th><th class="text-end">Mult</th><th>Owner</th></tr>
          </thead>
          <tbody>{all_player_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-rookies">
      <p class="text-secondary small mb-3">Top 40 rookies by KTC SF+TEP. <em>Adj</em> = league-adjusted value.</p>
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle">
          <thead class="table-secondary text-dark">
            <tr><th>Rk</th><th>Name</th><th>Pos</th><th>Team</th><th>Age</th><th class="text-end">Adj</th><th class="text-end">KTC</th><th class="text-end">Mult</th><th>Status</th></tr>
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
  Data: KeepTradeCut + Sleeper API &nbsp;&middot;&nbsp; RB 1.06&times; &middot; WR 1.08&times; positional multiplier
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
    # 1. KTC (scraped live)
    ktc_raw = fetch_ktc()
    source  = ktc_raw[0].get("source", "KTC") if ktc_raw else "KTC"
    if source == "FC":
        ktc_raw = apply_tep(ktc_raw)
        ktc_raw = add_positional_ranks(ktc_raw)

    # 2. Sleeper
    users, rosters, traded_picks, draft_info, all_players = fetch_sleeper()
    roster_to_owner, player_on_roster = build_maps(users, rosters)

    # 3. Match and adjust
    players = match_players(ktc_raw, all_players, player_on_roster)
    players = apply_position_adjustment(players)

    # 5. Pick boards
    pick_board     = build_pick_board(draft_info, traded_picks)
    pick_value_map = build_pick_value_map(ktc_raw)
    team_picks     = build_team_picks(draft_info, traded_picks, pick_value_map)

    # 6. Console output
    print()
    print_roster_overview(rosters, all_players, players, roster_to_owner,
                          team_picks=team_picks, my_team=MY_TEAM)
    print()
    print_rookie_board(players, roster_to_owner)
    print()
    print_pick_board(pick_board, roster_to_owner)
    print()

    # 7. HTML
    updated_at       = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    team_totals      = compute_team_totals(rosters, all_players, players, roster_to_owner, team_picks)
    rookie_data      = get_rookie_list(players, roster_to_owner)
    all_players_data = get_all_players_data(players, roster_to_owner)
    render_html(team_totals, rookie_data, all_players_data, pick_board, roster_to_owner, updated_at)
    print()


if __name__ == "__main__":
    main()
