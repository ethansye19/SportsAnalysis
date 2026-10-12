#!/usr/bin/env python3
"""
rec_yards_analyzer.py — Receiving Yards Prop Analyzer

Looks up a receiver's YPRR splits vs zone and man coverage (exported from the
R analysis to a CSV), blends them by the opponent's zone rate, projects his
receiving yards, and prices the prop against the sportsbook with quarter-Kelly
bet sizing.

SPLITS CSV (one row per player, exported from R)
------------------------------------------------
  player            name as it appears in nflreadr, e.g. "J.Jefferson"
  yprr_zone         yards per route run vs zone
  yprr_man          yards per route run vs man
  routes_per_game   average routes (dropbacks on the field) per game
  sd_yards          game-to-game standard deviation of receiving yards
  Optional: team, routes_zone, routes_man (route counts; used to flag small samples)

THE CALCULATION
---------------
  blended YPRR   = zone% * YPRR_zone + (1 - zone%) * YPRR_man
  expected yards = blended YPRR * projected routes * adjustments
  adjustments    = bounded multipliers for defense strength, shadow CB, QB form
  P(over)        = 1 - NormalCDF(line; mean = expected yards, sd = sd_yards)
  edge           = model probability - de-vigged (fair) market probability
  EV             = p * payout - (1 - p), per $1 at the offered odds
  stake          = 0.25 * full Kelly * bankroll

USAGE
-----
  # Look a player up in the splits CSV
  python rec_yards_analyzer.py lookup Jefferson --splits player_splits.csv

  # Add a prop (zone% is the opponent's zone rate, 0-100)
  python rec_yards_analyzer.py add --splits player_splits.csv --file props.json \\
      --player "J.Jefferson" --line 84.5 --over -115 --under -105 --zone-pct 72 \\
      --def-strength "Below Avg" --shadowed --shadow-tier Good \\
      --qb-recent 7.9 --qb-season 7.3

  # Ranked report
  python rec_yards_analyzer.py report --splits player_splits.csv --file props.json \\
      --bankroll 1000

  # Try it with made-up data
  python rec_yards_analyzer.py demo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

DEF_MAP = {"Elite": 0.90, "Above Avg": 0.95, "Average": 1.00, "Below Avg": 1.05, "Weak": 1.10}
SHADOW_MAP = {"Elite": 0.85, "Good": 0.92, "Average": 0.97}
FACTOR_BOUNDS = (0.80, 1.20)     # each single adjustment
NET_BOUNDS = (0.70, 1.30)        # all adjustments combined
FALLBACK_CV = 0.55               # sd = 0.55 * mean if sd_yards is missing
LOW_SAMPLE_ROUTES = 100          # flag splits built on fewer routes than this

# Accepted column names in the splits CSV (first match wins)
ALIASES = {
    "player": ["player", "player_name", "receiver", "receiver_player_name", "name"],
    "team": ["team", "posteam", "recent_team"],
    "yprr_zone": ["yprr_zone", "yprr_vs_zone", "zone_yprr"],
    "yprr_man": ["yprr_man", "yprr_vs_man", "man_yprr"],
    "routes_per_game": ["routes_per_game", "routes_pg", "avg_routes", "routes_per_gm"],
    "sd_yards": ["sd_yards", "yards_sd", "sd_rec_yards", "game_sd", "sd"],
    "routes_zone": ["routes_zone", "routes_vs_zone", "zone_routes"],
    "routes_man": ["routes_man", "routes_vs_man", "man_routes"],
}
REQUIRED = ["player", "yprr_zone", "yprr_man", "routes_per_game", "sd_yards"]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class PlayerSplits:
    player: str
    yprr_zone: float
    yprr_man: float
    routes_per_game: float
    sd_yards: Optional[float]
    team: str = ""
    routes_zone: Optional[float] = None
    routes_man: Optional[float] = None


@dataclass
class Prop:
    player: str                         # canonical name from the splits CSV
    line: float
    over_odds: int                      # American odds, e.g. -110 / +150
    under_odds: int
    zone_pct: float                     # opponent zone rate, 0-100
    team: str = ""                      # disambiguates duplicate names
    routes: Optional[float] = None      # override; defaults to routes_per_game
    def_strength: str = "Average"
    shadowed: bool = False
    shadow_tier: str = "Average"
    qb_recent: Optional[float] = None   # QB efficiency, last 3-5 games
    qb_season: Optional[float] = None   # QB efficiency, season


@dataclass
class Result:
    player: str
    line: float
    side: str
    odds: int
    blended_yprr: float
    routes: float
    base_yards: float
    exp_yards: float
    sd: float
    net_factor: float
    factors: List[Tuple[str, float]]
    p_model: float
    fair_p: float
    edge: float
    ev: float
    kelly_full: float
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normal_cdf(x: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def american_to_payout(odds: int) -> float:
    """Profit per $1 staked (b in the Kelly formula)."""
    return odds / 100 if odds > 0 else 100 / -odds


# --------------------------------------------------------------------------- #
# Splits CSV
# --------------------------------------------------------------------------- #

def _num(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.upper() in ("NA", "NAN", "NULL"):
        return None
    return float(v)


def load_splits(path: str) -> List[PlayerSplits]:
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(f"Error: splits file '{path}' not found. Export it from R or pass --splits.")
    with f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower(): h for h in (reader.fieldnames or []) if h}
        cols: Dict[str, str] = {}
        for key, names in ALIASES.items():
            for n in names:
                if n in headers:
                    cols[key] = headers[n]
                    break
        missing = [k for k in REQUIRED if k not in cols]
        if missing:
            sys.exit(f"Error: {path} is missing column(s): {', '.join(missing)}. "
                     f"Found: {', '.join(headers)}")

        players = []
        for row in reader:
            get = lambda k: row.get(cols[k]) if k in cols else None
            zone, man, rpg = _num(get("yprr_zone")), _num(get("yprr_man")), _num(get("routes_per_game"))
            if zone is None or man is None or rpg is None:
                continue                         # player lacks a split; skip
            players.append(PlayerSplits(
                player=get("player").strip(), team=(get("team") or "").strip(),
                yprr_zone=zone, yprr_man=man, routes_per_game=rpg,
                sd_yards=_num(get("sd_yards")),
                routes_zone=_num(get("routes_zone")), routes_man=_num(get("routes_man")),
            ))
    return players


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_players(splits: List[PlayerSplits], name: str, team: str = "") -> List[PlayerSplits]:
    """Exact (normalized) match first, then substring, e.g. 'jefferson' -> 'J.Jefferson'."""
    pool = [p for p in splits if not team or p.team.upper() == team.upper()]
    q = _norm(name)
    exact = [p for p in pool if _norm(p.player) == q]
    return exact or [p for p in pool if q in _norm(p.player)]


def resolve_player(splits: List[PlayerSplits], name: str, team: str = "") -> PlayerSplits:
    matches = find_players(splits, name, team)
    if not matches:
        sys.exit(f"Error: no player matching '{name}'" + (f" on {team}" if team else "")
                 + ". Try `lookup` to search.")
    if len(matches) > 1:
        listed = ", ".join(f"{p.player} ({p.team})" if p.team else p.player for p in matches[:10])
        sys.exit(f"Error: '{name}' matches several players: {listed}. "
                 "Use the full name or add --team.")
    return matches[0]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def adjustments(prop: Prop) -> Tuple[float, List[Tuple[str, float]]]:
    factors: List[Tuple[str, float]] = []

    f = DEF_MAP.get(prop.def_strength, 1.0)
    if f != 1.0:
        factors.append((f"Pass defense: {prop.def_strength}", f))

    if prop.shadowed:
        factors.append((f"Shadowed by {prop.shadow_tier} CB", SHADOW_MAP.get(prop.shadow_tier, 1.0)))

    if prop.qb_recent is not None and prop.qb_season:
        f = math.sqrt(prop.qb_recent / prop.qb_season)      # dampened: indirect effect
        if abs(f - 1) > 0.005:
            factors.append(("QB recent form", f))

    factors = [(label, clamp(f, *FACTOR_BOUNDS)) for label, f in factors]
    net = 1.0
    for _, f in factors:
        net *= f
    return clamp(net, *NET_BOUNDS), factors


def evaluate(prop: Prop, s: PlayerSplits) -> Result:
    z = prop.zone_pct / 100
    blended = z * s.yprr_zone + (1 - z) * s.yprr_man
    routes = prop.routes if prop.routes is not None else s.routes_per_game
    base = blended * routes
    net, factors = adjustments(prop)
    exp_yards = base * net

    warnings = []
    sd = s.sd_yards
    if sd is None or sd <= 0:
        sd = FALLBACK_CV * exp_yards
        warnings.append(f"no sd_yards in CSV; using {FALLBACK_CV:.2f} x projection")
    for label, n in (("zone", s.routes_zone), ("man", s.routes_man)):
        if n is not None and n < LOW_SAMPLE_ROUTES:
            warnings.append(f"only {n:.0f} routes vs {label}; split may be noisy")

    p_over = clamp(1 - normal_cdf(prop.line, exp_yards, sd), 0.001, 0.999)
    p_under = 1 - p_over

    raw_o, raw_u = american_to_implied(prop.over_odds), american_to_implied(prop.under_odds)
    fair_o = raw_o / (raw_o + raw_u)

    best = None
    for side, p, fair, odds in (("OVER", p_over, fair_o, prop.over_odds),
                                ("UNDER", p_under, 1 - fair_o, prop.under_odds)):
        b = american_to_payout(odds)
        ev = p * b - (1 - p)
        kelly = max(0.0, ev / b)
        cand = (p - fair, side, p, fair, odds, ev, kelly)
        if best is None or cand[0] > best[0]:
            best = cand
    edge, side, p, fair, odds, ev, kelly = best

    return Result(prop.player, prop.line, side, odds, blended, routes, base, exp_yards, sd,
                  net, factors, p, fair, edge, ev, kelly, warnings)


# --------------------------------------------------------------------------- #
# Props file
# --------------------------------------------------------------------------- #

def load_props(path: str) -> List[Prop]:
    try:
        with open(path) as f:
            return [Prop(**item) for item in json.load(f)]
    except FileNotFoundError:
        return []


def save_props(props: List[Prop], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in props], f, indent=2)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def spct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x * 100:.1f}%"


def fmt_odds(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)


def money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):.2f}"


def print_report(results: List[Result], bankroll: float, kelly_fraction: float,
                 min_edge: float) -> None:
    if not results:
        print("No props in the file. Use `add` first (or `demo` to try sample data).")
        return

    header = (f"{'#':<3}{'PLAYER':<20}{'SIDE':<6}{'LINE':>6}{'ODDS':>7}{'PROJ':>7}"
              f"{'MODEL P':>9}{'EDGE':>8}{'EV/$100':>10}{'STAKE':>10}  VERDICT")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        stake = r.kelly_full * kelly_fraction * bankroll
        play = r.edge >= min_edge and r.ev > 0
        name = r.player if len(r.player) <= 19 else r.player[:18] + "…"
        print(f"{i:<3}{name:<20}{r.side:<6}{r.line:>6.1f}{fmt_odds(r.odds):>7}{r.exp_yards:>7.1f}"
              f"{pct(r.p_model):>9}{spct(r.edge):>8}{money(r.ev * 100):>10}"
              f"{'$' + format(stake, '.2f'):>10}  {'PLAY' if play else 'pass'}")
        print(f"     blended YPRR {r.blended_yprr:.2f} x {r.routes:.1f} routes = {r.base_yards:.1f} yds"
              f"   (sd {r.sd:.1f})")
        for label, f in r.factors:
            print(f"       {label:<28} {spct(f - 1)}")
        if r.factors:
            print(f"       {'Net adjustment':<28} {spct(r.net_factor - 1)}  -> {r.exp_yards:.1f} yds")
        for w in r.warnings:
            print(f"       ! {w}")
    print()
    print(f"Edge is vs the de-vigged market. EV and stake use the offered odds; "
          f"stake = {kelly_fraction:g} Kelly on a ${bankroll:,.0f} bankroll. "
          f"PLAY = edge >= {min_edge * 100:.1f}% and EV > 0.")


# --------------------------------------------------------------------------- #
# Demo data (made-up players)
# --------------------------------------------------------------------------- #

def write_demo(splits_path: str, props_path: str) -> None:
    rows = [
        ["player", "team", "yprr_zone", "yprr_man", "routes_per_game", "sd_yards", "routes_zone", "routes_man"],
        ["A.Receiver", "AAA", 2.45, 1.70, 34.2, 28.5, 410, 150],
        ["B.Wideout", "BBB", 1.60, 2.10, 31.0, 24.0, 360, 140],
        ["C.Slot", "CCC", 1.95, 1.30, 27.5, 19.5, 300, 85],
    ]
    with open(splits_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    props = [
        Prop("A.Receiver", 74.5, -115, -105, zone_pct=75, def_strength="Below Avg",
             qb_recent=7.9, qb_season=7.3),
        Prop("B.Wideout", 61.5, -110, -110, zone_pct=60, shadowed=True, shadow_tier="Elite"),
        Prop("C.Slot", 44.5, +105, -125, zone_pct=82),
    ]
    save_props(props, props_path)
    print(f"Wrote {splits_path} and {props_path}. Now run:\n"
          f"  python rec_yards_analyzer.py report --splits {splits_path} --file {props_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receiving yards prop analyzer: YPRR vs zone/man splits -> projection, EV and quarter-Kelly stake.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, splits=True):
        if splits:
            sp.add_argument("--splits", default="player_splits.csv", help="CSV exported from R")
        sp.add_argument("--file", default="props.json", help="Saved props")

    lk = sub.add_parser("lookup", help="Search the splits CSV for a player.")
    lk.add_argument("name")
    lk.add_argument("--splits", default="player_splits.csv")

    a = sub.add_parser("add", help="Add a prop.")
    common(a)
    a.add_argument("--player", required=True)
    a.add_argument("--team", default="", help="Team abbreviation if the name is ambiguous")
    a.add_argument("--line", type=float, required=True)
    a.add_argument("--over", type=int, required=True, dest="over_odds", help="American odds, e.g. -110")
    a.add_argument("--under", type=int, required=True, dest="under_odds")
    a.add_argument("--zone-pct", type=float, required=True, help="Opponent zone rate, 0-100")
    a.add_argument("--routes", type=float, help="Projected routes (default: his routes per game)")
    a.add_argument("--def-strength", default="Average", choices=list(DEF_MAP))
    a.add_argument("--shadowed", action="store_true")
    a.add_argument("--shadow-tier", default="Average", choices=list(SHADOW_MAP))
    a.add_argument("--qb-recent", type=float)
    a.add_argument("--qb-season", type=float)

    r = sub.add_parser("report", help="Ranked report of all saved props.")
    common(r)
    r.add_argument("--bankroll", type=float, default=1000.0)
    r.add_argument("--kelly-fraction", type=float, default=0.25)
    r.add_argument("--min-edge-pct", type=float, default=2.0)

    c = sub.add_parser("clear", help="Delete all saved props.")
    common(c, splits=False)

    d = sub.add_parser("demo", help="Write made-up splits and props to try the tool.")
    d.add_argument("--splits", default="demo_splits.csv")
    d.add_argument("--file", default="demo_props.json")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "lookup":
        matches = find_players(load_splits(args.splits), args.name)
        if not matches:
            print(f"No players matching '{args.name}'.")
        for s in matches:
            team = f" ({s.team})" if s.team else ""
            sd = f"{s.sd_yards:.1f}" if s.sd_yards is not None else "n/a"
            print(f"{s.player}{team}: YPRR zone {s.yprr_zone:.2f} | man {s.yprr_man:.2f} | "
                  f"{s.routes_per_game:.1f} routes/g | sd {sd}")

    elif args.command == "add":
        if not 0 <= args.zone_pct <= 100:
            sys.exit("Error: --zone-pct must be between 0 and 100.")
        s = resolve_player(load_splits(args.splits), args.player, args.team)
        prop = Prop(
            player=s.player, team=s.team, line=args.line,
            over_odds=args.over_odds, under_odds=args.under_odds,
            zone_pct=args.zone_pct, routes=args.routes, def_strength=args.def_strength,
            shadowed=args.shadowed, shadow_tier=args.shadow_tier,
            qb_recent=args.qb_recent, qb_season=args.qb_season,
        )
        props = load_props(args.file)
        props.append(prop)
        save_props(props, args.file)
        print(f"Added {s.player} {args.line} to {args.file} ({len(props)} props total).")

    elif args.command == "report":
        splits = load_splits(args.splits)
        results = [evaluate(p, resolve_player(splits, p.player, p.team)) for p in load_props(args.file)]
        results.sort(key=lambda r: r.edge, reverse=True)
        print_report(results, args.bankroll, args.kelly_fraction, args.min_edge_pct / 100)

    elif args.command == "clear":
        save_props([], args.file)
        print(f"Cleared {args.file}.")

    elif args.command == "demo":
        write_demo(args.splits, args.file)


if __name__ == "__main__":
    main()
