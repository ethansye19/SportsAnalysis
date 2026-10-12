#!/usr/bin/env python3
"""
rec_yards_analyzer.py — Receiving Yards Prop Analyzer

Projects a receiver's yards from his Yards Per Route Run (YPRR) splits,
weighted by the opponent's coverage tendencies, then prices the over/under
against the sportsbook and sizes the bet with quarter Kelly.

PROJECTION
----------
  coverage YPRR  = zone% * YPRR_vs_zone + man% * YPRR_vs_man
  expected yards = expected routes * coverage YPRR * adjustments

  Optional adjustments (bounded multipliers):
    - Blitz rate:   (blitz% * YPRR_vs_blitz + (1 - blitz%) * YPRR_vs_no_blitz) / overall YPRR
    - Shadow CB:    Elite 0.82 | Good 0.90 | Average 0.96
    - QB form:      sqrt(recent / season), dampened since it's an indirect effect

  Expected routes come from --routes directly, or from
  --dropbacks (team's projected dropbacks) * --route-rate (% of dropbacks he runs a route).

PRICING
-------
  Receiving yards are right-skewed (a few big games pull the mean above the
  median), so outcomes are modeled as lognormal by default, with spread set
  by a coefficient of variation (CV). The CV is taken from the player's game
  log if you supply one, otherwise a default of 0.55.

  Edge  = model probability - de-vigged (fair) market probability
  EV    = expected profit per $1 at the offered odds
  Stake = Kelly fraction (default 0.25, quarter Kelly) * full Kelly * bankroll

CLI QUICK START
---------------
  python rec_yards_analyzer.py sample --file props.json
  python rec_yards_analyzer.py report --file props.json --bankroll 1000

  python rec_yards_analyzer.py add --file props.json \\
      --player "J. Smith" --line 64.5 --over -115 --under -105 \\
      --yprr-zone 2.35 --yprr-man 1.60 --zone-pct 72 \\
      --dropbacks 38 --route-rate 88 \\
      --yprr 2.10 --blitz-pct 31 --yprr-blitz 2.45 --yprr-no-blitz 1.95 \\
      --shadowed --shadow-tier Good \\
      --qb-recent 7.9 --qb-season 7.3 \\
      --game-log 74,61,95,50,82,68,90
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

DEFAULT_CV = 0.55
SHADOW_MAP = {"Elite": 0.82, "Good": 0.90, "Average": 0.96}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Prop:
    player: str
    line: float
    over_odds: int                      # American odds, e.g. -110 / +150
    under_odds: int
    yprr_zone: float                    # player's YPRR vs zone coverage
    yprr_man: float                     # player's YPRR vs man coverage
    zone_pct: float                     # opponent's zone rate, 0-100
    routes: float                       # expected routes run this game
    man_pct: Optional[float] = None     # defaults to 100 - zone_pct
    yprr: Optional[float] = None        # overall YPRR (needed for blitz adjustment)
    blitz_pct: Optional[float] = None   # opponent's blitz rate, 0-100
    yprr_blitz: Optional[float] = None
    yprr_no_blitz: Optional[float] = None
    shadowed: bool = False
    shadow_tier: str = "Average"        # Elite | Good | Average
    qb_recent: Optional[float] = None   # QB efficiency metric, last 3-5 games
    qb_season: Optional[float] = None   # QB efficiency metric, season
    game_log: List[float] = field(default_factory=list)
    dist: str = "lognormal"             # "lognormal" | "normal"


@dataclass
class Result:
    player: str
    line: float
    side: str
    odds: int
    coverage_yprr: float
    routes: float
    base_yards: float
    exp_yards: float
    adj_factor: float
    factors: List[Tuple[str, float]]
    cv: float
    p_model: float
    fair_p: float
    edge: float
    ev: float
    kelly_full: float


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def american_to_payout(odds: int) -> float:
    """Profit per $1 staked (Kelly's b)."""
    return odds / 100 if odds > 0 else 100 / -odds


def prob_over(line: float, mean: float, cv: float, dist: str) -> float:
    if mean <= 0:
        return 0.0
    if dist == "normal":
        return 1 - phi((line - mean) / (cv * mean))
    # Lognormal matched to the projected mean and CV
    sigma2 = math.log(1 + cv * cv)
    mu = math.log(mean) - sigma2 / 2
    if line <= 0:
        return 1.0
    return 1 - phi((math.log(line) - mu) / math.sqrt(sigma2))


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def coverage_weights(prop: Prop) -> Tuple[float, float]:
    zone = prop.zone_pct
    man = prop.man_pct if prop.man_pct is not None else 100 - zone
    total = zone + man
    if total <= 0:
        raise ValueError(f"{prop.player}: zone% + man% must be positive")
    return zone / total, man / total    # normalized in case they don't sum to 100


def project(prop: Prop) -> Tuple[float, float, float, List[Tuple[str, float]]]:
    """Returns (coverage YPRR, base yards, net adjustment, factors applied)."""
    wz, wm = coverage_weights(prop)
    cov_yprr = wz * prop.yprr_zone + wm * prop.yprr_man
    base_yards = prop.routes * cov_yprr

    factors: List[Tuple[str, float]] = []

    if (prop.blitz_pct is not None and prop.yprr_blitz is not None
            and prop.yprr_no_blitz is not None and prop.yprr):
        b = prop.blitz_pct / 100
        blitz_yprr = b * prop.yprr_blitz + (1 - b) * prop.yprr_no_blitz
        f = clamp(blitz_yprr / prop.yprr, 0.80, 1.20)
        if abs(f - 1) > 0.005:
            factors.append((f"Opp blitz rate {prop.blitz_pct:.0f}%", f))

    if prop.shadowed:
        factors.append((f"Shadowed by {prop.shadow_tier} CB",
                        SHADOW_MAP.get(prop.shadow_tier, 1.0)))

    if prop.qb_recent is not None and prop.qb_season:
        f = math.sqrt(clamp(prop.qb_recent / prop.qb_season, 0.85, 1.15))
        if abs(f - 1) > 0.005:
            factors.append(("QB recent form", f))

    net = 1.0
    for _, f in factors:
        net *= f
    net = clamp(net, 0.6, 1.4)
    return cov_yprr, base_yards, net, factors


def estimate_cv(game_log: List[float]) -> float:
    if len(game_log) >= 3:
        m = statistics.mean(game_log)
        if m > 0:
            return clamp(statistics.stdev(game_log) / m, 0.30, 1.00)
    return DEFAULT_CV


def evaluate(prop: Prop) -> Result:
    cov_yprr, base_yards, net, factors = project(prop)
    exp_yards = base_yards * net
    cv = estimate_cv(prop.game_log)

    p_over = clamp(prob_over(prop.line, exp_yards, cv, prop.dist), 0.001, 0.999)
    p_under = 1 - p_over

    raw_over = american_to_implied(prop.over_odds)
    raw_under = american_to_implied(prop.under_odds)
    fair_over = raw_over / (raw_over + raw_under)
    fair_under = 1 - fair_over

    sides = []
    for side, p, fair, odds in (("OVER", p_over, fair_over, prop.over_odds),
                                ("UNDER", p_under, fair_under, prop.under_odds)):
        b = american_to_payout(odds)
        ev = p * b - (1 - p)
        kelly = max(0.0, ev / b)
        sides.append((p - fair, side, p, fair, odds, ev, kelly))
    edge, side, p, fair, odds, ev, kelly = max(sides)

    return Result(prop.player, prop.line, side, odds, cov_yprr, prop.routes, base_yards,
                  exp_yards, net, factors, cv, p, fair, edge, ev, kelly)


# --------------------------------------------------------------------------- #
# Storage
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
# Report
# --------------------------------------------------------------------------- #

def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def signed_pct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x * 100:.1f}%"


def fmt_odds(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)


def print_report(results: List[Result], bankroll: float, kelly_fraction: float,
                 min_edge: float) -> None:
    if not results:
        print("No props loaded. Use `add` to add one, or `sample` for an example file.")
        return

    for r in results:
        stake = r.kelly_full * kelly_fraction * bankroll
        play = r.edge >= min_edge and r.ev > 0
        print("=" * 72)
        print(f"{r.player} — {r.side} {r.line} ({fmt_odds(r.odds)})"
              f"   [{'PLAY' if play else 'pass'}]")
        print("-" * 72)
        print(f"  Coverage-weighted YPRR  {r.coverage_yprr:.2f}  x  {r.routes:.1f} routes"
              f"  =  {r.base_yards:.1f} yds")
        for label, f in r.factors:
            print(f"    {label:<30} {signed_pct(f - 1)}")
        if r.factors:
            print(f"  Net adjustment {signed_pct(r.adj_factor - 1)}"
                  f"  ->  projected {r.exp_yards:.1f} yds")
        print(f"  Model P({r.side.lower()}) {pct(r.p_model)}  vs fair market {pct(r.fair_p)}"
              f"   (CV {r.cv:.2f})")
        print(f"  Edge {signed_pct(r.edge)}  |  EV per $100 "
              f"{'+' if r.ev >= 0 else '-'}${abs(r.ev) * 100:.2f}"
              f"  |  {kelly_fraction:g} Kelly stake ${stake:.2f}")
    print("=" * 72)
    print("Edge is vs the de-vigged market; EV and Kelly use the offered odds. "
          "Bet only what you can afford to lose.")


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #

def write_sample(path: str) -> None:
    sample = [
        Prop(player="J. Smith", line=64.5, over_odds=-115, under_odds=-105,
             yprr_zone=2.35, yprr_man=1.60, zone_pct=72, routes=33.4,
             yprr=2.10, blitz_pct=31, yprr_blitz=2.45, yprr_no_blitz=1.95,
             qb_recent=7.9, qb_season=7.3, game_log=[74, 61, 95, 50, 82, 68, 90]),
        Prop(player="D. Owens", line=48.5, over_odds=-110, under_odds=-110,
             yprr_zone=1.55, yprr_man=1.90, zone_pct=58, routes=30.0,
             shadowed=True, shadow_tier="Elite", game_log=[41, 66, 22, 58, 39, 71]),
        Prop(player="T. Grant", line=39.5, over_odds=+105, under_odds=-125,
             yprr_zone=1.80, yprr_man=1.25, zone_pct=80, routes=26.5),
    ]
    save_props(sample, path)
    print(f"Wrote {len(sample)} sample props to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_floats(s: Optional[str]) -> List[float]:
    return [float(v) for v in s.split(",") if v.strip()] if s else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receiving yards prop analyzer: YPRR splits x coverage tendencies -> EV and Kelly sizing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="Add a prop.")
    a.add_argument("--file", default="props.json")
    a.add_argument("--player", required=True)
    a.add_argument("--line", type=float, required=True)
    a.add_argument("--over", type=int, required=True, dest="over_odds")
    a.add_argument("--under", type=int, required=True, dest="under_odds")
    a.add_argument("--yprr-zone", type=float, required=True)
    a.add_argument("--yprr-man", type=float, required=True)
    a.add_argument("--zone-pct", type=float, required=True, help="Opponent zone rate, 0-100")
    a.add_argument("--man-pct", type=float, help="Opponent man rate (default 100 - zone)")
    a.add_argument("--routes", type=float, help="Expected routes run")
    a.add_argument("--dropbacks", type=float, help="Team's projected dropbacks")
    a.add_argument("--route-rate", type=float, help="%% of dropbacks player runs a route")
    a.add_argument("--yprr", type=float, help="Overall YPRR (needed for blitz adjustment)")
    a.add_argument("--blitz-pct", type=float)
    a.add_argument("--yprr-blitz", type=float)
    a.add_argument("--yprr-no-blitz", type=float)
    a.add_argument("--shadowed", action="store_true")
    a.add_argument("--shadow-tier", default="Average", choices=list(SHADOW_MAP))
    a.add_argument("--qb-recent", type=float)
    a.add_argument("--qb-season", type=float)
    a.add_argument("--game-log", help="Comma-separated recent yardage totals (sets the CV)")
    a.add_argument("--dist", default="lognormal", choices=["lognormal", "normal"])

    r = sub.add_parser("report", help="Rank all props in the file.")
    r.add_argument("--file", default="props.json")
    r.add_argument("--bankroll", type=float, default=1000.0)
    r.add_argument("--kelly-fraction", type=float, default=0.25)
    r.add_argument("--min-edge-pct", type=float, default=2.0)

    c = sub.add_parser("clear", help="Delete all props.")
    c.add_argument("--file", default="props.json")

    s = sub.add_parser("sample", help="Write an example props file.")
    s.add_argument("--file", default="props_sample.json")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "add":
        if args.routes is not None:
            routes = args.routes
        elif args.dropbacks is not None and args.route_rate is not None:
            routes = args.dropbacks * args.route_rate / 100
        else:
            sys.exit("Error: give --routes, or both --dropbacks and --route-rate.")
        for name in ("zone_pct", "man_pct", "blitz_pct", "route_rate"):
            v = getattr(args, name)
            if v is not None and not 0 <= v <= 100:
                sys.exit(f"Error: --{name.replace('_', '-')} must be between 0 and 100.")

        prop = Prop(
            player=args.player, line=args.line,
            over_odds=args.over_odds, under_odds=args.under_odds,
            yprr_zone=args.yprr_zone, yprr_man=args.yprr_man,
            zone_pct=args.zone_pct, man_pct=args.man_pct, routes=routes,
            yprr=args.yprr, blitz_pct=args.blitz_pct,
            yprr_blitz=args.yprr_blitz, yprr_no_blitz=args.yprr_no_blitz,
            shadowed=args.shadowed, shadow_tier=args.shadow_tier,
            qb_recent=args.qb_recent, qb_season=args.qb_season,
            game_log=parse_floats(args.game_log), dist=args.dist,
        )
        props = load_props(args.file)
        props.append(prop)
        save_props(props, args.file)
        print(f"Added {args.player} ({routes:.1f} routes) to {args.file} — {len(props)} props total.")

    elif args.command == "report":
        results = sorted((evaluate(p) for p in load_props(args.file)),
                         key=lambda r: r.edge, reverse=True)
        print_report(results, args.bankroll, args.kelly_fraction, args.min_edge_pct / 100)

    elif args.command == "clear":
        save_props([], args.file)
        print(f"Cleared {args.file}.")

    elif args.command == "sample":
        write_sample(args.file)


if __name__ == "__main__":
    main()
