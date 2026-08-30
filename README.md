# Edge Finder

A prop-betting edge model: feed it a player's historical stat line, the sportsbook's
line and odds, and optional matchup context (opponent defense strength, zone/man
scheme fit, shadow coverage, QB splits and recent form) — it estimates the true
probability of over/under, de-vigs the market to find your edge, and sizes a
suggested stake with fractional Kelly.

Two versions of the same model:

- **`edge_finder.html`** — a self-contained interactive tool. Open it in a browser,
  add props through the form, see a live-ranked table and a "top play" card.
- **`edge_finder.py`** — a pure-stdlib Python CLI/library. No dependencies to install.

## Python usage

```bash
# see the expected data shape
python edge_finder.py sample --file props.json

# add a prop (context flags are all optional)
python edge_finder.py add --file props.json \
    --player "J. Smith - Rec Yards" --values 74,61,95,50,82 \
    --line 64.5 --over -110 --under -110 \
    --def-strength Weak --scheme Zone --vs-zone 80 --vs-man 60 \
    --shadowed --shadow-tier Elite \
    --qb-zone 8.2 --qb-man 7.1 --qb-recent 7.8 --qb-season 7.3

# print the ranked report
python edge_finder.py report --file props.json --bankroll 1000 --kelly-fraction 0.25
```

Or use it as a library:

```python
from edge_finder import PropInput, Context, evaluate_prop

prop = PropInput(
    player="J. Smith - Rec Yards", values=[74, 61, 95, 50, 82],
    line=64.5, over_odds=-110, under_odds=-110,
    context=Context(def_strength="Weak", scheme="Zone", vs_zone=80, vs_man=60),
)
result = evaluate_prop(prop)
```

## How the model works

- **Probability model** — Normal distribution for continuous stats (yards, points),
  Poisson for count stats (TDs, receptions), fit from the historical values you give it.
- **De-vigging** — strips the sportsbook's built-in juice from both sides' odds to get
  a fair, no-vig market probability. "Edge" = model probability − fair market probability.
- **EV & Kelly** — expected value and stake sizing use the *actual* offered odds (what
  you're really paid), scaled by a fractional-Kelly setting so sizing isn't overly aggressive.
- **Context adjustments** — defense strength, scheme fit, shadow coverage, and QB
  splits/form each contribute a bounded multiplier to the projected mean. Every
  adjustment is shown in the output — nothing is a black box.

## Disclaimer

This is a probability estimate built on whatever inputs you give it — small
samples, stale ratings, or a guessed shadow assignment all propagate straight
into the output. It is not a guarantee, and sports outcomes have real variance
even when the model is right. Bet only what you can afford to lose. If gambling
stops feeling fun or in your control, the National Problem Gambling Helpline
(1-800-522-4700) is free and confidential.
