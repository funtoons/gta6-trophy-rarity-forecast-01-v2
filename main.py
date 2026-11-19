#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads every ``*.in`` case file in the input directory (``$MAKEFUN_INPUT_DIR``
or ``/input/``) and writes one ``<id>.out`` forecast per case to the output
directory (``$MAKEFUN_OUTPUT_DIR`` or ``/output/``).

Model, in one paragraph: Exophase refreshes a game's counts on its own
schedule, hours apart, so the house's reads sit unchanged between refreshes.
We collapse consecutive identical reads into one plateau point each, then,
per trophy, fit three views of growth against log(1 + hours since release)
over the recent window with recency weights: log earners, log players
tracked, and the logit of the earn rate itself.  Story trophies (the chapter
chain) share one pooled earners slope, because they move together; grind and
collection trophies keep their own slope.  The median forecast blends the
earners/players decomposition, the direct logit trend and persistence on the
last rate in logit space, then structural priors are applied as a soft
adjustment (chapter rates non-increasing along the story, the platinum no
more common than any other trophy, 100% completion no more common than the
final mission).  The spread grows with the horizon, the fit's residual,
and the slope's own uncertainty extrapolated to the target, with a
slightly longer lower tail than upper tail, because a flood of new
players dilutes rates downward while growth upward is slower.

Python 3.13, standard library only, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tunable model settings (defaults chosen by sweep over synthetic
# final-like worlds; see the sweep notes in the repository README).
CFG = {
    "half_life_h": 12.0,      # recency weight half-life, hours
    "window_h": 36.0,         # fit window, hours before the checkpoint
    "w_decomp": 0.40,         # blend weight: earners/players decomposition
    "w_trend": 0.30,          # blend weight: direct logit trend
    "w_persist": 0.30,        # blend weight: persistence on the last rate
    "sigma_floor": 0.30,      # logit spread floor
    "sigma_horizon_k": 0.10,  # spread grows with horizon (days): k * days
    "sigma_disagree_k": 1.0,  # weight on the between-views variance
    "tail_lo": 1.10,          # lower-tail multiplier (longer tail down)
    "tail_hi": 1.00,          # upper-tail multiplier
    "median_clamp": 3.5,      # median logit may move this far from last rate
    "prior_blend": 0.50,      # how far structural priors pull the medians
    "pool_trend": True,       # pool the logit-trend slope across the story
    "slope_max_earn": 1.6,    # clamp on the earners growth slope
    "slope_max_play": 1.3,    # clamp on the players growth slope
    "slope_max_logit": 3.0,   # clamp on the logit-rate growth slope
    "slope_min_logit": -1.0,
    "ceil_platinum": 10.0,    # median ceilings, percent
    "ceil_completion": 12.0,
    "ceil_other": 25.0,
}

# Trophy ids of the story chain, in story order: Chapter 1..4, final mission.
STORY_IDS = (3, 4, 5, 6, 7)
PLATINUM_ID = 1
COMPLETION_ID = 14
FIRST_TROPHY_ID = 2  # "Complete Gone Fishing", ~100% from launch night


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return (t - RELEASE).total_seconds() / 3600.0


def x_of(t):
    """Log-time coordinate; reads just before release map near zero."""
    return math.log1p(max(-0.99, hours_since_release(t)))


def wls(points, cfg):
    """Weighted least squares y = a + b*x over (x, y, w) points.

    Returns (a, b, resid_var, wsum, sxx, x_mean) or None when the fit is
    degenerate.  resid_var is the weighted mean squared residual; the
    slope's standard error is sqrt(resid_var / sxx).
    """
    if len(points) < 2:
        return None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - (a + b * x)) ** 2 for x, y, w in points) / sw
    return (a, b, resid, sw, sxx, mx)


def collapse_plateaus(history):
    """Merge consecutive identical reads into one point each.

    Returns a list of (time, players, rates) with the time of the first
    read showing each distinct page state.
    """
    out = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(None if r is None else round(r, 4) for r in rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((parse_time(read["at"]), read.get("players_tracked"),
                    rates))
    return out


def forecast_case(case, cfg=None):
    cfg = dict(CFG) if cfg is None else cfg
    trophies = case["trophies"]
    n = len(trophies)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)
    x_target = x_of(target)

    points = collapse_plateaus(case.get("history") or [])
    last_time = points[-1][0] if points else checkpoint

    # --- per-trophy fits -------------------------------------------------
    # For each trophy: (median_logit, resid_var, last_logit, last_rate)
    medians = [None] * n
    resid_vars = [0.0] * n
    extrap_vars = [0.0] * n
    disagree_vars = [0.0] * n
    last_logits = [0.0] * n
    last_rates = [1.0] * n

    fits = []  # per trophy: dict of fit results, for pooling
    for i, tr in enumerate(trophies):
        series = []  # (x, log_earners, log_players, logit_rate, weight)
        last_rate = None
        for t, players, rates in points:
            if i >= len(rates) or rates[i] is None:
                continue
            if players is None or players <= 0:
                continue
            r = rates[i]
            last_rate = r
            age_h = max(0.0, (last_time - t).total_seconds() / 3600.0)
            if age_h > cfg["window_h"]:
                continue
            w = 0.5 ** (age_h / cfg["half_life_h"])
            earners = max(r / 100.0 * players, 0.5)
            series.append((x_of(t), math.log(earners), math.log(players),
                           logit(r), w))
        if last_rate is None:
            last_rate = 1.0
        last_rates[i] = last_rate
        last_logits[i] = logit(last_rate)

        if len(series) < 2:
            fits.append(None)
            medians[i] = last_logits[i]
            continue
        fe = wls([(x, y, w) for x, y, _, _, w in series], cfg)
        fp = wls([(x, y, w) for x, _, y, _, w in series], cfg)
        fl = wls([(x, y, w) for x, _, _, y, w in series], cfg)
        fits.append({"earn": fe, "play": fp, "logit": fl,
                     "series": series})

    # --- pooled slopes for the story chain -------------------------------
    # Story trophies move together: average their earners slope (and, when
    # enabled, their logit slope) weighted by each fit's weight mass, then
    # refit each trophy's intercept at the pooled slope.
    def pooled_slope(key, ids):
        num = 0.0
        den = 0.0
        for i in ids:
            f = fits[i] if i < len(fits) else None
            if f is None or f[key] is None:
                continue
            num += f[key][1] * f[key][3]
            den += f[key][3]
        if den <= 0:
            return None
        return num / den

    story_idx = [i for i, tr in enumerate(trophies)
                 if tr["id"] in STORY_IDS]
    pooled_earn = pooled_slope("earn", story_idx)
    pooled_logit = pooled_slope("logit", story_idx) if cfg["pool_trend"] \
        else None

    def refit_intercept(series, col, slope):
        sw = sum(w for *_, w in series)
        if sw <= 0:
            return 0.0
        return sum(w * (row[col] - slope * row[0])
                   for row, w in ((s, s[4]) for s in series)) / sw

    # --- median forecast per trophy --------------------------------------
    for i, tr in enumerate(trophies):
        f = fits[i]
        if f is None:
            medians[i] = last_logits[i]
            continue
        fe, fp, fl = f["earn"], f["play"], f["logit"]
        series = f["series"]
        if fe is None or fp is None or fl is None:
            medians[i] = last_logits[i]
            continue

        b_earn = fe[1]
        b_logit = fl[1]
        a_earn = fe[0]
        a_logit = fl[0]
        if i in story_idx:
            if pooled_earn is not None:
                b_earn = clamp(pooled_earn, 0.0, cfg["slope_max_earn"])
                a_earn = refit_intercept(series, 1, b_earn)
            if pooled_logit is not None:
                b_logit = clamp(pooled_logit, cfg["slope_min_logit"],
                                cfg["slope_max_logit"])
                a_logit = refit_intercept(series, 3, b_logit)
        else:
            b_earn = clamp(b_earn, 0.0, cfg["slope_max_earn"])
            b_logit = clamp(b_logit, cfg["slope_min_logit"],
                            cfg["slope_max_logit"])
        b_play = clamp(fp[1], 0.0, cfg["slope_max_play"])

        # Decomposition: earners / players at the target.
        log_earn_t = a_earn + b_earn * x_target
        log_play_t = fp[0] + b_play * x_target
        rate_decomp = 100.0 * math.exp(log_earn_t - log_play_t)
        # Direct logit trend at the target.
        rate_trend = inv_logit(a_logit + b_logit * x_target)

        med = (cfg["w_decomp"] * logit(rate_decomp)
               + cfg["w_trend"] * logit(rate_trend)
               + cfg["w_persist"] * last_logits[i])
        # Model disagreement: when the three views of the future disagree,
        # the truth is genuinely uncertain, so the spread widens by their
        # weighted variance around the blended median.
        disagree_vars[i] = (
            cfg["w_decomp"] * (logit(rate_decomp) - med) ** 2
            + cfg["w_trend"] * (logit(rate_trend) - med) ** 2
            + cfg["w_persist"] * (last_logits[i] - med) ** 2)
        med = clamp(med, last_logits[i] - cfg["median_clamp"],
                    last_logits[i] + cfg["median_clamp"])
        medians[i] = med
        resid_vars[i] = fl[2]
        # Slope uncertainty extrapolated to the target: a noisy fit, or a
        # long extrapolation from the fit's centre, widens the band.
        if fl[4] > 1e-12:
            slope_se = math.sqrt(max(fl[2], 0.0) / fl[4])
            dx = x_target - fl[5]
            extrap_vars[i] = min((slope_se * dx) ** 2, 1.0)

    # --- structural priors, as a soft adjustment -------------------------
    # Work in logit space on a copy; pull each median part-way toward the
    # constraint-satisfying value instead of clipping hard, so the
    # quantiles keep honest width around a plausible centre.
    by_id = {tr["id"]: i for i, tr in enumerate(trophies)}
    adj = list(medians)
    pb = cfg["prior_blend"]

    def pull_toward(idx, ceiling_logit):
        if adj[idx] > ceiling_logit:
            adj[idx] = (1.0 - pb) * adj[idx] + pb * ceiling_logit

    # Chapter chain non-increasing: Ch1 >= Ch2 >= Ch3 >= Ch4 >= final.
    chain = [by_id[t] for t in STORY_IDS if t in by_id]
    if len(chain) >= 2:
        # Isotonic (pool adjacent violators) on the chain, then blend.
        vals = [adj[i] for i in chain]
        pooled = pava_decreasing(vals)
        for k, i in enumerate(chain):
            adj[i] = (1.0 - pb) * adj[i] + pb * pooled[k]
    # Platinum no more common than any other trophy.
    if PLATINUM_ID in by_id:
        others = [adj[i] for j, i in enumerate(range(n))
                  if trophies[j]["id"] != PLATINUM_ID]
        if others:
            pull_toward(by_id[PLATINUM_ID], min(others))
    # 100% completion no more common than the final mission.
    if COMPLETION_ID in by_id and STORY_IDS[-1] in by_id:
        pull_toward(by_id[COMPLETION_ID], adj[by_id[STORY_IDS[-1]]])

    # --- ceilings on the median ------------------------------------------
    # Story trophies may climb much higher than grind trophies in week
    # one (every player passes Chapter 1; few finish a collection), so
    # the chapter chain gets its own, higher ceilings.
    story_caps = {3: 45.0, 4: 35.0, 5: 30.0, 6: 25.0, 7: 20.0}
    for i, tr in enumerate(trophies):
        if tr["id"] == PLATINUM_ID:
            cap = cfg["ceil_platinum"]
        elif tr["id"] == COMPLETION_ID:
            cap = cfg["ceil_completion"]
        elif tr["id"] in story_caps:
            cap = story_caps[tr["id"]]
        else:
            cap = cfg["ceil_other"]
        adj[i] = min(adj[i], logit(cap))

    # --- quantiles --------------------------------------------------------
    rows = []
    for i, tr in enumerate(trophies):
        if last_rates[i] >= 99.995:
            # Essentially every tracked player holds it; the refresh that
            # took players 638 -> 3868 left it at 100%.  Only the lower
            # tail admits dilution by brand-new players.  q95 is 100.0,
            # not 99.99, so a truth of exactly 100% still counts covered.
            rows.append({"id": tr["id"],
                         "q": [93.0, 99.5, 99.99, 99.99, 100.0]})
            continue
        sig = math.sqrt(cfg["sigma_floor"] ** 2
                        + cfg["sigma_horizon_k"] * horizon_days
                        + resid_vars[i] + extrap_vars[i]
                        + cfg["sigma_disagree_k"] * disagree_vars[i])
        qs = []
        for z in Z:
            tail = cfg["tail_lo"] if z < 0 else cfg["tail_hi"]
            qs.append(round(clamp(inv_logit(adj[i] + z * sig * tail),
                                  0.01, 99.99), 4))
        # Guarantee non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def pava_decreasing(vals):
    """Pool-adjacent-violators for a non-increasing sequence."""
    blocks = [[v, 1] for v in vals]
    out = []
    for v, c in blocks:
        out.append([v, c])
        while len(out) >= 2 and out[-2][0] < out[-1][0]:
            v2, c2 = out.pop()
            v1, c1 = out.pop()
            out.append([(v1 * c1 + v2 * c2) / (c1 + c2), c1 + c2])
    res = []
    for v, c in out:
        res.extend([v] * c)
    return res


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
