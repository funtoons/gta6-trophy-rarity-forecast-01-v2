# gta6-trophy-rarity-forecast-01-v2

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recorded reads of the game's trophy page up to the deadline.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file (`/input/<id>.in`) and writes `/output/<id>.out`. For each trophy it:

1. Collapses consecutive identical reads into plateaus (Exophase refreshes its counts on its own schedule, hours apart, so the page sits unchanged between refreshes).
2. Fits, on the last 36 hours of reads with recency weights, three views of growth against `log(1 + hours since release)`: log earners, log players tracked, and the logit of the earn rate itself. Story trophies (the chapter chain) share one pooled earners slope, since they move together; grind and collection trophies keep their own slope.
3. Blends those views in logit space (decomposition earners/players, direct logit trend, and persistence on the last rate) into a median forecast for the target, with structural priors applied as a soft adjustment: chapter rates non-increasing along the story, the platinum no more common than any other trophy, and 100% completion no more common than the final mission.
4. Widens the spread with the horizon (4.52 days to the target) and the fit's residual, with a longer lower tail than upper tail, because a flood of new players dilutes rates downward while growth upward is slower.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the case files (`p00.in`, `p01.in` in the kit).

## Entry

This repository is an entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) "Forecast how many GTA VI players earn each trophy in week one" ($250, twenty places).

Optional: earn referral rewards, a second way to make money — join make.fun with my referral link: https://make.fun/r/mqqhsens
