"""Configuration and tuning knobs for the trade suggestion engine.

This module centralizes constants and trade-balance presets so they can
be reused without importing the full engine.
"""

# From league rules / engine tuning
ROSTER_SIZE = 10  # From league rules
REPLACEMENT_PERCENTILE = 0.85  # Top 85% of rostered players
MIN_GAMES_REQUIRED = 25  # Weekly minimum games in current configuration
AVG_GAMES_PER_PLAYER = 3.5  # Approximate NBA games per player per fantasy week
MIN_TRADE_FP_G = 40.0  # Players below this FP/G are treated as drop-tier and excluded from trade search

# Hard caps on how many players per team enter the trade engine. These were
# originally 10/10; tightening them reduces combinatorial explosion while still
# considering the key pieces on each roster.
MAX_CANDIDATES_YOUR = 8
MAX_CANDIDATES_THEIR = 8

# Per-pattern cap on combinations evaluated for a given team pair.
MAX_COMBINATIONS_PER_PATTERN = 1000

ENABLE_VALUE_FAIRNESS_GUARD = False
MAX_OPP_WEEKLY_LOSS = 15.0  # Max weekly core FP we assume an opponent would accept losing (slightly loosened)
EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS = 10
MAX_OPP_CORE_AVG_DROP = 1.5  # Max FP/G drop in opponent's core average (league philosophy: FP/G > total FP)
MIN_GP_SHARE_OF_MAX = 0.25
MAX_COMPLEXITY_OPS = 200_000  # Rough cap on estimated combination evaluations
EQUAL_COUNT_MAX_AVG_FPTS_RATIO = 1.15
EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = 1.5
TRADE_BALANCE_LEVEL = 5  # 1-10 scale, 5 = standard
SHOW_COMPLEXITY_DEBUG = False  # Toggle to print per-team combination estimates
MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM = 12  # Early-exit once we have plenty of good trades for a pattern vs team


def _apply_trade_balance_level(level: int) -> None:
	"""Apply threshold settings for a 1-10 trade balance level.

	This function now acts as a fallback static ramp. League-aware caps are
	derived later from calculate_league_scarcity_context and the stored
	TRADE_BALANCE_LEVEL inside find_trade_suggestions.
	"""
	global EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS, EQUAL_COUNT_MAX_AVG_FPTS_RATIO, EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO
	level = max(1, min(10, int(level)))

	# Linear ramps between conservative (1) and aggressive (10)
	EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS = 4.0 + (level - 1) * 1.5  # 4  17.5
	EQUAL_COUNT_MAX_AVG_FPTS_RATIO = 1.05 + (level - 1) * 0.025  # 1.05  1.275
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = 1.02 + (level - 1) * 0.02   # 1.02  1.20


def set_trade_balance_preset(preset) -> None:
	"""Adjust equal-count realism thresholds.

	Accepts either legacy labels ("conservative", "standard", "aggressive")
	or a numeric balance level 1-10 (5 = standard).
	"""
	global TRADE_BALANCE_LEVEL
	if isinstance(preset, (int, float)):
		level = int(preset)
	else:
		key = (preset or 'standard').lower()
		legacy_map = {
			'very conservative': 2,
			'conservative': 3,
			'standard': 5,
			'aggressive': 7,
			'ultra aggressive': 9,
		}
		level = legacy_map.get(key, 5)

	TRADE_BALANCE_LEVEL = max(1, min(50, int(level)))
	_apply_trade_balance_level(TRADE_BALANCE_LEVEL)
