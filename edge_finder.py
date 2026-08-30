#!/usr/bin/env python3
"""
edge_finder.py — prop betting edge model with matchup-context adjustments.

Pure stdlib. Works as a CLI (JSON file as your prop "database") or as an
importable library.

CLI QUICK START
----------------
  # Write a sample props file to see the expected shape
  python edge_finder.py sample --file props.json

  # Add a prop (context flags are all optional)
  python edge_finder.py add --file props.json \\
      --player "J. Smith - Rec Yards" --values 74,61,95,50,82 \\
      --line 64.5 --over -110 --under -110 --model normal \\
      --def-strength Weak --scheme Zone --vs-zone 80 --vs-man 60 \\
      --shadowed --shadow-tier Elite \\
      --qb-zone 8.2 --qb-man 7.1 --qb-recent 7.8 --qb-season 7.3

  # Print the ranked report
  python edge_finder.py report --file props.json --bankroll 1000 \\
      --kelly-fraction 0.25 --min-edge-pct 2.0

  # Wipe the file
  python edge_finder.py clear --file props.json

LIBRARY QUICK START
--------------------
  from edge_finder import PropInput, Context, evaluate_prop

  prop = PropInput(
      player="J. Smith - Rec Yards", values=[74, 61, 95, 50, 82],
      line=64.5, over_odds=-110, under_odds=-110, model="normal",
      context=Context(def_strength="Weak", scheme="Zone", vs_zone=80, vs_man=60,
                       shadowed=True, shadow_tier="Elite",
                       qb_zone=8.2, qb_man=7.1, qb_recent=7.8, qb_season=7.3),
  )
  result = evaluate_prop(prop)
  print(result.side, result.edge, result.ev)

MODEL NOTES
-----------
"Edge" compares the model's estimated probability against the de-vigged
(no-juice) market probability implied by both sides' odds. EV$ and Kelly
stake use the *actual* offered odds, since that's what you're paid on.
Matchup-context adjustments (defense strength, scheme fit, shadow coverage,
QB splits/form) shift the projected mean by a bounded multiplier so no
single factor can dominate. Small historical samples, stale ratings, or a
guessed shadow assignment propagate straight into the output — this is a
probability estimate built on the inputs you give it, not a guarantee.
Bet only what you can afford to lose. If gambling stops feeling fun or in
your control, the National Problem Gambling Helpline (1-800-522-4700) is
free and confidential.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Context:
    def_strength: str = "Average"       # Elite | Above Avg | Average | Below Avg | Weak
    scheme: str = "Unknown"             # Zone | Man | Unknown
    vs_zone: Optional[float] = None     # player's avg stat vs zone coverage
    vs_man: Optional[float] = None      # player's avg stat vs man coverage
    shadowed: bool = False
    shadow_tier: str = "Average"        # Elite | Good | Average
    qb_zone: Optional[float] = None     # QB efficiency metric vs zone
    qb_man: Optional[float] = None      # QB efficiency metric vs man
    qb_recent: Optional[float] = None   # QB efficiency metric, last 3-5 games
    qb_season: Optional[float] = None   # QB efficiency metric, season avg


@dataclass
class PropInput:
    player: str
    values: List[float]                 # historical game-log values
    line: float
    over_odds: int                      # American odds, e.g. -110 / +150
    under_odds: int
    model: str = "normal"               # "normal" | "poisson"
    context: Context = field(default_factory=Context)


@dataclass
class EvaluatedProp:
    player: str
    line: float
    side: str                           # "OVER" | "UNDER"
    p_model: float
    edge: float                         # vs de-vigged fair market probability
    ev: float                           # expected value per $1 staked
    kelly_raw: float                    # un-fractioned Kelly stake, 0-1
    odds: int
    base_mean: float
    adj_mean: float
    ctx_factor: float
    factors: List[Tuple[str, float]]    # (label, multiplier) applied
    n: int


# --------------------------------------------------------------------------
# Core math
# --------------------------------------------------------------------------

def normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


def poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    total = 0.0
    term = math.exp(-lam)
    total += term
    for i in range(1, max(k, 0) + 1):
        term *= lam / i
        total += term
    return min(total, 1.0)


def sample_stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else (-odds) / ((-odds) + 100)


def american_to_payout(odds: int) -> float:
    """Profit per $1 staked, i.e. the 'b' in Kelly's formula."""
    return odds / 100 if odds > 0 else 100 / (-odds)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_odds(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)


def fmt_factor(f: float) -> str:
    sign = "+" if f >= 1 else ""
    return f"{sign}{(f - 1) * 100:.1f}%"


# --------------------------------------------------------------------------
# Context adjustment
# --------------------------------------------------------------------------

DEF_MAP = {"Elite": 0.86, "Above Avg": 0.94, "Average": 1.0, "Below Avg": 1.06, "Weak": 1.14}
SHADOW_MAP = {"Elite": 0.82, "Good": 0.90, "Average": 0.96}


def compute_context_factors(prop: PropInput) -> Tuple[float, List[Tuple[str, float]]]:
    """Returns (net multiplier, list of (label, multiplier) that were applied)."""
    ctx = prop.context
    factors: List[Tuple[str, float]] = []
    total = 1.0

    # Opponent defense strength
    def_f = DEF_MAP.get(ctx.def_strength, 1.0)
    if def_f != 1.0:
        factors.append((f"Opp defense: {ctx.def_strength}", def_f))
        total *= def_f

    # Scheme fit — player's own zone/man splits vs overall average
    overall = statistics.mean(prop.values) if prop.values else 0.0
    if ctx.scheme != "Unknown" and ctx.vs_zone is not None and ctx.vs_man is not None and overall > 0:
        relevant = ctx.vs_zone if ctx.scheme == "Zone" else ctx.vs_man
        f = clamp(relevant / overall, 0.75, 1.25)
        if abs(f - 1) > 0.005:
            factors.append((f"Scheme fit vs {ctx.scheme}", f))
            total *= f

    # Shadow coverage
    if ctx.shadowed:
        f = SHADOW_MAP.get(ctx.shadow_tier, 1.0)
        factors.append((f"Shadowed by {ctx.shadow_tier} CB", f))
        total *= f

    # QB scheme split — dampened (secondary/indirect effect)
    if ctx.scheme != "Unknown" and ctx.qb_zone is not None and ctx.qb_man is not None:
        qb_avg = (ctx.qb_zone + ctx.qb_man) / 2
        if qb_avg > 0:
            relevant = ctx.qb_zone if ctx.scheme == "Zone" else ctx.qb_man
            raw = clamp(relevant / qb_avg, 0.85, 1.15)
            f = math.sqrt(raw)
            if abs(f - 1) > 0.005:
                factors.append((f"QB vs {ctx.scheme} split", f))
                total *= f

    # QB recent form — dampened (secondary/indirect effect)
    if ctx.qb_recent is not None and ctx.qb_season is not None and ctx.qb_season > 0:
        raw = clamp(ctx.qb_recent / ctx.qb_season, 0.85, 1.15)
        f = math.sqrt(raw)
        if abs(f - 1) > 0.005:
            factors.append(("QB recent form", f))
            total *= f

    total = clamp(total, 0.6, 1.5)
    return total, factors


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _side_calc(p: float, odds: int) -> Tuple[float, float, float]:
    """Returns (implied_prob, ev_per_$1, kelly_raw) for one side at these odds."""
    b = american_to_payout(odds)
    ev = p * b - (1 - p)
    kelly_raw = max(0.0, (b * p - (1 - p)) / b)
    return american_to_implied(odds), ev, kelly_raw


def evaluate_prop(prop: PropInput) -> EvaluatedProp:
    base_mean = statistics.mean(prop.values)
    base_std = sample_stdev(prop.values)

    ctx_factor, factors = compute_context_factors(prop)
    adj_mean = base_mean * ctx_factor
    adj_std = base_std * ctx_factor

    if prop.model == "poisson":
        k = math.floor(prop.line)
        p_over = 1 - poisson_cdf(k, adj_mean)
    else:
        p_over = 1 - normal_cdf(prop.line, adj_mean, adj_std)
    p_over = clamp(p_over, 0.001, 0.999)
    p_under = 1 - p_over

    over_raw = american_to_implied(prop.over_odds)
    under_raw = american_to_implied(prop.under_odds)
    vig_sum = over_raw + under_raw
    fair_over = over_raw / vig_sum
    fair_under = under_raw / vig_sum

    _, ev_over, kelly_over = _side_calc(p_over, prop.over_odds)
    _, ev_under, kelly_under = _side_calc(p_under, prop.under_odds)

    edge_over = p_over - fair_over
    edge_under = p_under - fair_under

    use_over = edge_over >= edge_under
    side = "OVER" if use_over else "UNDER"
    p_model = p_over if use_over else p_under
    edge = edge_over if use_over else edge_under
    ev = ev_over if use_over else ev_under
    kelly_raw = kelly_over if use_over else kelly_under
    odds = prop.over_odds if use_over else prop.under_odds

    return EvaluatedProp(
        player=prop.player, line=prop.line, side=side, p_model=p_model, edge=edge,
        ev=ev, kelly_raw=kelly_raw, odds=odds, base_mean=base_mean, adj_mean=adj_mean,
        ctx_factor=ctx_factor, factors=factors, n=len(prop.values),
    )


def rank_props(props: List[PropInput]) -> List[EvaluatedProp]:
    return sorted((evaluate_prop(p) for p in props), key=lambda e: e.edge, reverse=True)


# --------------------------------------------------------------------------
# Persistence (JSON file acts as your prop "database")
# --------------------------------------------------------------------------

def load_props(path: str) -> List[PropInput]:
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return []
    props = []
    for item in raw:
        ctx_dict = item.get("context", {})
        props.append(PropInput(
            player=item["player"], values=item["values"], line=item["line"],
            over_odds=item["over_odds"], under_odds=item["under_odds"],
            model=item.get("model", "normal"), context=Context(**ctx_dict),
        ))
    return props


def save_props(props: List[PropInput], path: str) -> None:
    raw = [asdict(p) for p in props]
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


# --------------------------------------------------------------------------
# Report printing
# --------------------------------------------------------------------------

def print_report(evaluated: List[EvaluatedProp], kelly_fraction: float,
                  bankroll: float, min_edge: float) -> None:
    if not evaluated:
        print("No props loaded. Use `add` to add one, or `sample` for an example file.")
        return

    top = evaluated[0]
    if top.edge > 0:
        stake = max(0.0, top.kelly_raw * kelly_fraction * bankroll)
        print("=" * 72)
        print(f"TOP PLAY: {top.player} — {top.side} {top.line}")
        ctx_note = f" · context adj {fmt_factor(top.ctx_factor)}" if top.ctx_factor != 1 else ""
        print(f"  model p({top.side.lower()}) {fmt_pct(top.p_model)} · odds {fmt_odds(top.odds)} "
              f"· n={top.n}{ctx_note}")
        print(f"  edge vs fair mkt +{fmt_pct(top.edge)}  |  EV/$100 "
              f"{'+' if top.ev >= 0 else ''}${top.ev * 100:.2f}  |  suggested stake ${stake:.2f}")
        print("=" * 72)
        print()

    header = f"{'PLAYER':<28}{'SIDE':<7}{'ODDS':>7}{'MODEL P':>9}{'EDGE':>9}{'EV/$100':>10}{'STAKE':>10}  VERDICT"
    print(header)
    print("-" * len(header))
    for e in evaluated:
        stake = max(0.0, e.kelly_raw * kelly_fraction * bankroll)
        is_play = e.edge >= min_edge and e.ev > 0
        verdict = "PLAY" if is_play else "pass"
        name = (e.player[:26] + "…") if len(e.player) > 27 else e.player
        print(f"{name:<28}{e.side:<7}{fmt_odds(e.odds):>7}{fmt_pct(e.p_model):>9}"
              f"{('+' if e.edge >= 0 else '') + fmt_pct(e.edge):>9}"
              f"{('+' if e.ev >= 0 else '') + '$' + format(e.ev * 100, '.2f'):>10}"
              f"{'$' + format(stake, '.2f'):>10}  {verdict}")
        if e.factors:
            print(f"    ↳ baseline avg {e.base_mean:.1f} → adjusted avg {e.adj_mean:.1f} "
                  f"(net {fmt_factor(e.ctx_factor)})")
            for label, f in e.factors:
                print(f"       - {label:<32} {fmt_factor(f)}")
    print()
    print("Model notes: edge is vs the de-vigged (no-juice) market; EV$/Kelly stake use the "
          "actual offered odds. Context adjustments are bounded multipliers, not guarantees. "
          "Bet only what you can afford to lose.")


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------

def write_sample(path: str) -> None:
    sample = [
        PropInput(
            player="J. Smith - Rec Yards", values=[74, 61, 95, 50, 82, 68, 90],
            line=64.5, over_odds=-115, under_odds=-105, model="normal",
            context=Context(def_strength="Weak", scheme="Zone", vs_zone=85, vs_man=58,
                             shadowed=False, qb_zone=8.2, qb_man=7.1,
                             qb_recent=7.9, qb_season=7.3),
        ),
        PropInput(
            player="D. Owens - Receptions", values=[3, 5, 4, 2, 6, 4, 3], line=3.5,
            over_odds=-120, under_odds=+100, model="poisson",
            context=Context(shadowed=True, shadow_tier="Elite", def_strength="Above Avg"),
        ),
        PropInput(
            player="R. Carter - Rush Yards", values=[102, 88, 65, 120, 74, 91], line=79.5,
            over_odds=-108, under_odds=-112, model="normal",
            context=Context(def_strength="Elite"),
        ),
    ]
    save_props(sample, path)
    print(f"Wrote {len(sample)} sample props to {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prop betting edge model with matchup-context adjustments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a prop to the JSON file.")
    p_add.add_argument("--file", default="props.json")
    p_add.add_argument("--player", required=True)
    p_add.add_argument("--values", required=True, help="Comma-separated historical values, e.g. 74,61,95")
    p_add.add_argument("--line", required=True, type=float)
    p_add.add_argument("--over", required=True, type=int, dest="over_odds", help="American odds, e.g. -110")
    p_add.add_argument("--under", required=True, type=int, dest="under_odds")
    p_add.add_argument("--model", default="normal", choices=["normal", "poisson"])
    p_add.add_argument("--def-strength", default="Average",
                        choices=["Elite", "Above Avg", "Average", "Below Avg", "Weak"])
    p_add.add_argument("--scheme", default="Unknown", choices=["Zone", "Man", "Unknown"])
    p_add.add_argument("--vs-zone", type=float, default=None, help="Player's avg stat vs zone coverage")
    p_add.add_argument("--vs-man", type=float, default=None, help="Player's avg stat vs man coverage")
    p_add.add_argument("--shadowed", action="store_true")
    p_add.add_argument("--shadow-tier", default="Average", choices=["Elite", "Good", "Average"])
    p_add.add_argument("--qb-zone", type=float, default=None)
    p_add.add_argument("--qb-man", type=float, default=None)
    p_add.add_argument("--qb-recent", type=float, default=None)
    p_add.add_argument("--qb-season", type=float, default=None)

    p_report = sub.add_parser("report", help="Print the ranked report for all props in the file.")
    p_report.add_argument("--file", default="props.json")
    p_report.add_argument("--bankroll", type=float, default=1000.0)
    p_report.add_argument("--kelly-fraction", type=float, default=0.25)
    p_report.add_argument("--min-edge-pct", type=float, default=2.0,
                           help="Minimum edge %% to mark a prop PLAY (default 2.0)")

    p_clear = sub.add_parser("clear", help="Delete all props from the file.")
    p_clear.add_argument("--file", default="props.json")

    p_sample = sub.add_parser("sample", help="Write an example props file.")
    p_sample.add_argument("--file", default="props_sample.json")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "add":
        values = [float(v.strip()) for v in args.values.split(",") if v.strip()]
        if len(values) < 3:
            print("Error: enter at least 3 historical values.", file=sys.stderr)
            sys.exit(1)
        ctx = Context(
            def_strength=args.def_strength, scheme=args.scheme,
            vs_zone=args.vs_zone, vs_man=args.vs_man,
            shadowed=args.shadowed, shadow_tier=args.shadow_tier,
            qb_zone=args.qb_zone, qb_man=args.qb_man,
            qb_recent=args.qb_recent, qb_season=args.qb_season,
        )
        prop = PropInput(
            player=args.player, values=values, line=args.line,
            over_odds=args.over_odds, under_odds=args.under_odds,
            model=args.model, context=ctx,
        )
        props = load_props(args.file)
        props.append(prop)
        save_props(props, args.file)
        print(f"Added '{args.player}' to {args.file} ({len(props)} props total).")

    elif args.command == "report":
        props = load_props(args.file)
        evaluated = rank_props(props)
        print_report(evaluated, kelly_fraction=args.kelly_fraction,
                     bankroll=args.bankroll, min_edge=args.min_edge_pct / 100)

    elif args.command == "clear":
        save_props([], args.file)
        print(f"Cleared {args.file}.")

    elif args.command == "sample":
        write_sample(args.file)


if __name__ == "__main__":
    main()
