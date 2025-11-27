"""Realism and fairness checks for the trade suggestion engine.

This module holds the guards that determine whether a given package swap
is realistic enough that an opponent might plausibly accept it. It is
used by the core engine in modules.trade_suggestions and by any
auxiliary tools that need to reason about trade realism.
"""

from typing import Dict, Tuple, List

import modules.trade_suggestions_config as cfg


def _check_avg_fpts_ratio(your_avg_fpts: float, their_avg_fpts: float, is_consolidating: bool, is_expanding: bool) -> bool:
	"""Average FP/G sanity check for package quality similarity."""
	if min(your_avg_fpts, their_avg_fpts) <= 0:
		fpts_ratio = 999
	else:
		fpts_ratio = max(your_avg_fpts, their_avg_fpts) / min(your_avg_fpts, their_avg_fpts)
	if is_consolidating:
		if fpts_ratio > 1.35:
			return False
	elif is_expanding:
		if fpts_ratio > 1.25:
			return False
	else:
		if fpts_ratio > cfg.EQUAL_COUNT_MAX_AVG_FPTS_RATIO:
			return False
	return True


def _check_total_fpts_and_piece_quality(
	your_players,
	their_players,
	your_total_fpts: float,
	their_total_fpts: float,
your_count: int,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	if is_consolidating:
		max_ratio = 1.25
		their_max = max(p['Mean FPts'] for p in their_players)
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		if their_max >= elite_threshold:
			max_ratio = 1.35
		elif their_max >= star_threshold:
			max_ratio = 1.30
		if your_count == 2:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			if your_sorted[1] < their_max * 0.45:
				return False
			if your_sorted[0] < their_max * 0.60:
				return False
		elif your_count == 3:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			if any(p < their_max * 0.38 for p in your_sorted):
				return False
			if your_sorted[0] < their_max * 0.55:
				return False
		your_min = min(p['Mean FPts'] for p in your_players)
		your_avg = sum(p['Mean FPts'] for p in your_players) / len(your_players)
		min_piece_ratio = 0.52
		avg_piece_ratio = 0.64
		if your_min < their_max * min_piece_ratio:
			return False
		if your_avg < their_max * avg_piece_ratio:
			return False
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		package_loss_for_them = their_total_fpts - your_total_fpts
		if package_loss_for_them > 0:
			if their_max >= elite_threshold:
				max_loss = 30
			elif their_max >= star_threshold:
				max_loss = 25
			else:
				max_loss = 20
			if package_loss_for_them > max_loss:
				return False
	elif is_expanding:
		max_ratio = 1.25
		your_max = max(p['Mean FPts'] for p in your_players)
		their_min = min(p['Mean FPts'] for p in their_players)
		their_avg = sum(p['Mean FPts'] for p in their_players) / len(their_players)
		min_piece_ratio = 0.52
		avg_piece_ratio = 0.64
		if their_min < your_max * min_piece_ratio:
			return False
		if their_avg < your_max * avg_piece_ratio:
			return False
	else:
		max_ratio = cfg.EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO
	if min(your_total_fpts, their_total_fpts) <= 0:
		total_ratio = 999
	else:
		total_ratio = max(your_total_fpts, their_total_fpts) / min(your_total_fpts, their_total_fpts)
	if total_ratio > max_ratio:
		return False
	return True


def _check_value_fairness_guard(
	your_players,
	their_players,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	if not cfg.ENABLE_VALUE_FAIRNESS_GUARD:
		return True
	if 'Value' not in your_players[0] or 'Value' not in their_players[0]:
		return True
	your_total_value = sum(float(p['Value']) for p in your_players)
	their_total_value = sum(float(p['Value']) for p in their_players)
	if your_total_value <= 0 or their_total_value <= 0:
		return True
	value_ratio = max(your_total_value, their_total_value) / min(your_total_value, their_total_value)
	if is_consolidating:
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		incoming_max_fpts = max(p['Mean FPts'] for p in their_players)
		if incoming_max_fpts >= elite_threshold:
			max_value_ratio = 1.33
		elif incoming_max_fpts >= star_threshold:
			max_value_ratio = 1.30
		else:
			max_value_ratio = 1.25
	elif is_expanding:
		max_value_ratio = 1.18
	else:
		max_value_ratio = 1.12
	if value_ratio > max_value_ratio:
		return False
	return True


def _check_cv_tradeoff(
	your_players,
	their_players,
	your_avg_cv: float,
	their_avg_cv: float,
	your_avg_fpts: float,
	their_avg_fpts: float,
	is_consolidating: bool,
) -> bool:
	if your_avg_cv < their_avg_cv:
		cv_loss = their_avg_cv - your_avg_cv
		their_max_fpts = max(p['Mean FPts'] for p in their_players)
		your_max_fpts = max(p['Mean FPts'] for p in your_players)
		if cv_loss > 10:
			consistency_upgrade_needed = 1.05
		elif cv_loss > 5:
			consistency_upgrade_needed = 1.02
		else:
			consistency_upgrade_needed = 1.0
		if is_consolidating and their_max_fpts >= your_max_fpts + 5:
			consistency_upgrade_needed = 1.0
		if their_avg_fpts < your_avg_fpts * consistency_upgrade_needed:
			return False
	return True


def _derive_max_fpts(your_players, their_players) -> Tuple[float, float]:
	your_max_fpts = max(p['Mean FPts'] for p in your_players)
	their_max_fpts = max(p['Mean FPts'] for p in their_players)
	return your_max_fpts, their_max_fpts


def _check_tier_protection(
	your_players,
	their_players,
	your_max_fpts: float,
	their_max_fpts: float,
	your_count: int,
	their_count: int,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	starter_threshold = league_tiers['starter']
	if your_max_fpts >= elite_threshold:
		if is_expanding and their_count >= 3:
			if not all(p['Mean FPts'] >= quality_threshold for p in their_players):
				return False
		elif their_max_fpts < star_threshold:
			return False
		if is_consolidating and their_count == 1:
			if their_max_fpts < elite_threshold * 0.90:
				return False
	elif their_max_fpts >= elite_threshold:
		if is_consolidating and your_count >= 3:
			if not all(p['Mean FPts'] >= quality_threshold for p in your_players):
				return False
		elif your_max_fpts < star_threshold:
			return False
		if your_count == 1 and your_max_fpts < elite_threshold * 0.90:
			return False
	if your_max_fpts >= star_threshold:
		if is_expanding and their_count >= 3:
			if not all(p['Mean FPts'] >= starter_threshold for p in their_players):
				return False
		elif their_max_fpts < quality_threshold:
			return False
	if their_max_fpts >= star_threshold:
		if is_consolidating and your_count >= 3:
			if not all(p['Mean FPts'] >= starter_threshold for p in your_players):
				return False
		elif your_max_fpts < quality_threshold:
			return False
	return True


def _check_consolidation_upgrade(
	is_consolidating: bool,
	your_max_fpts: float,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	if not is_consolidating:
		return True
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	if their_max_fpts >= elite_threshold:
		if your_max_fpts < star_threshold:
			return False
		min_upgrade = elite_threshold * 0.15
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	elif their_max_fpts >= star_threshold:
		min_upgrade = star_threshold * 0.12
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	elif their_max_fpts >= quality_threshold:
		min_upgrade = quality_threshold * 0.10
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	else:
		min_upgrade = their_max_fpts * 0.08
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	return True


def _check_best_player_ratio(
	is_consolidating: bool,
	is_expanding: bool,
	your_max_fpts: float,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	if min(your_max_fpts, their_max_fpts) <= 0:
		best_player_ratio = 999
	else:
		best_player_ratio = max(your_max_fpts, their_max_fpts) / min(your_max_fpts, their_max_fpts)
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	if is_consolidating:
		if their_max_fpts >= elite_threshold:
			best_ratio_limit = 1.20
		elif their_max_fpts >= star_threshold:
			best_ratio_limit = 1.22
		elif their_max_fpts >= quality_threshold:
			best_ratio_limit = 1.25
		else:
			best_ratio_limit = 1.18
	elif is_expanding:
		best_ratio_limit = 1.12
	else:
		best_ratio_limit = 1.08
	if best_player_ratio > best_ratio_limit:
		return False
	return True


def _check_individual_matchups(
	your_players,
	their_players,
	is_consolidating: bool,
	is_expanding: bool,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	your_fpts_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
	their_fpts_sorted = sorted([p['Mean FPts'] for p in their_players], reverse=True)
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	for i, your_fpts in enumerate(your_fpts_sorted):
		if i < len(their_fpts_sorted):
			their_fpts = their_fpts_sorted[i]
			if is_consolidating:
				if their_max_fpts >= elite_threshold:
					player_ratio_limit = 1.28
				elif their_max_fpts >= star_threshold:
					player_ratio_limit = 1.32
				elif their_max_fpts >= quality_threshold:
					player_ratio_limit = 1.35
				else:
					player_ratio_limit = 1.20
			elif is_expanding:
				player_ratio_limit = 1.15
			else:
				player_ratio_limit = 1.10
			if min(your_fpts, their_fpts) <= 0:
				player_ratio = 999
			else:
				player_ratio = max(your_fpts, their_fpts) / min(your_fpts, their_fpts)
			if player_ratio > player_ratio_limit:
				return False
	return True


def _check_1_for_n_package_ratio(
	your_player,
	their_players,
	league_tiers: Dict[str, float],
) -> bool:
	"""Tier- and slider-aware total FP/G ratio guard specifically for 1-for-2/1-for-3.

	At normal/strict settings, keep ratios fairly modest; at very loose settings,
	allow bigger upgrades, especially when you're giving up an elite.
	"""
	your_fpts = float(your_player['Mean FPts'])
	their_total_fpts = float(sum(p['Mean FPts'] for p in their_players))
	if your_fpts <= 0 or their_total_fpts <= 0:
		return True
	# You are expanding: giving 1, getting >1
	total_ratio = max(your_fpts, their_total_fpts) / min(your_fpts, their_total_fpts)
	# Base caps by tier of your outgoing player
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	if your_fpts >= elite_threshold:
		# Elite outgoing: allow strong but not absurd depth upgrades
		base_cap = 1.85
	elif your_fpts >= star_threshold:
		# Star outgoing: consolidating into multiple solid pieces can be more dramatic
		base_cap = 2.25
	else:
		base_cap = 1.8
	# Trade-balance slider adjustment: higher looseness => higher cap
	# TRADE_BALANCE_LEVEL ~ 1..50; map to multiplier ~ [0.9, 1.2]
	looseness = cfg.TRADE_BALANCE_LEVEL
	looseness_factor = 0.9 + (min(max(looseness, 1), 50) - 1) * (0.3 / 49.0)
	max_ratio = base_cap * looseness_factor
	return total_ratio <= max_ratio


def _check_3_for_2_package_ratio(
	your_players: List[Dict],
	their_players: List[Dict],
) -> bool:
	"""Slider-aware total FP/G ratio guard for 3-for-2 consolidations.

	At strict settings, you should not give up much more total FP/G than you
	receive when consolidating 3 pieces into 2. At looser settings, allow
	slightly larger FP/G sacrifices for meaningful consolidation upgrades.
	"""
	your_total_fpts = float(sum(p['Mean FPts'] for p in your_players))
	their_total_fpts = float(sum(p['Mean FPts'] for p in their_players))
	if your_total_fpts <= 0 or their_total_fpts <= 0:
		return True
	# Only guard cases where you are giving more total FP/G than you get.
	if your_total_fpts <= their_total_fpts:
		return True
	total_ratio = your_total_fpts / their_total_fpts
	# Base cap: at strictest setting, allow a moderate FP/G sacrifice for consolidation.
	base_cap = 1.15
	# TRADE_BALANCE_LEVEL ~ 1..50; map to multiplier ~ [1.0, 1.3]
	looseness = cfg.TRADE_BALANCE_LEVEL
	looseness_factor = 1.0 + (min(max(looseness, 1), 50) - 1) * (0.3 / 49.0)
	max_ratio = base_cap * looseness_factor
	return total_ratio <= max_ratio


def _is_realistic_trade(your_players, their_players, league_tiers: Dict[str, float]):
	"""Check if a trade is realistic (not too lopsided).
	Prevents suggesting trades that no one would accept.
	Considers both FPts and consistency (CV%).

	Now more lenient to allow:
	- Consistency upgrades (lower CV%) even if FP/G is sideways
	- Consolidation trades that improve core FP/G

	Uses percentile-based tiers from actual league data instead of hardcoded thresholds.
	"""
	your_count = len(your_players)
	their_count = len(their_players)
	is_consolidating = your_count > their_count
	is_expanding = their_count > your_count
	if cfg.TRADE_BALANCE_LEVEL >= 40 and (is_consolidating or is_expanding):
		return True
	your_avg_fpts = sum(p['Mean FPts'] for p in your_players) / len(your_players)
	their_avg_fpts = sum(p['Mean FPts'] for p in their_players) / len(their_players)
	your_avg_cv = sum(p['CV %'] for p in your_players) / len(your_players)
	their_avg_cv = sum(p['CV %'] for p in their_players) / len(their_players)
	# NOTE: Consistency (CV) is mostly your edge. Opponent realism is driven by FP/G.
	# We do NOT let "consistency upgrades" justify huge FP/G gaps an opponent
	# would never consciously accept.
	# 1) Average FP/G sanity check: don't allow trades where the average quality
	# of pieces is wildly different.
	if not _check_avg_fpts_ratio(your_avg_fpts, their_avg_fpts, is_consolidating, is_expanding):
		return False
	# 2) Total FP/G sanity check (package totals). This is lenient for
	# consolidations into true elite players, but independent of CV.
	your_total_fpts = sum(p['Mean FPts'] for p in your_players)
	their_total_fpts = sum(p['Mean FPts'] for p in their_players)
	if not _check_total_fpts_and_piece_quality(
		your_players,
		their_players,
		your_total_fpts,
		their_total_fpts,
		your_count,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	if not _check_value_fairness_guard(
		your_players,
		their_players,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	# CV% check: Only block if trading consistency for volatility WITHOUT FP/G upgrade
	# Now more lenient - consistency for FP/G is a valid tradeoff
	if not _check_cv_tradeoff(
		your_players,
		their_players,
		your_avg_cv,
		their_avg_cv,
		your_avg_fpts,
		their_avg_fpts,
		is_consolidating,
	):
		return False
	# Additional check: prevent trading elite players for scrubs
	your_max_fpts, their_max_fpts = _derive_max_fpts(your_players, their_players)
	# TIER-BASED PROTECTION: Elite players require elite return
	# BUT: Allow 1-for-3 expansions as a valid depth strategy
	# Now uses percentile-based tiers instead of hardcoded thresholds
	if not _check_tier_protection(
		your_players,
		their_players,
		your_max_fpts,
		their_max_fpts,
		your_count,
		their_count,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	# When consolidating, ensure the incoming player meaningfully upgrades your top end
	# Now uses percentile-based upgrade requirements
	if not _check_consolidation_upgrade(
		is_consolidating,
		your_max_fpts,
		their_max_fpts,
		league_tiers,
	):
		return False
	# Additional check: best player comparison
	# The best player in the trade shouldn't be too much better than the other side's best
	if not _check_best_player_ratio(
		is_consolidating,
		is_expanding,
		your_max_fpts,
		their_max_fpts,
		league_tiers,
	):
		return False
	# Check individual player matchups - each player you give should have a comparable player you get
	if not _check_individual_matchups(
		your_players,
		their_players,
		is_consolidating,
		is_expanding,
		their_max_fpts,
		league_tiers,
	):
		return False
	return True
