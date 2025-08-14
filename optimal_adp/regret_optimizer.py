"""Regret calculation and ADP optimization for fantasy football drafts."""

import logging
from typing import Dict

from .config import Player
from .draft_simulator import DraftState, simulate_from_pick

logger = logging.getLogger(__name__)


def _check_position_hierarchy_violation(
    draft_state: DraftState, pick_number: int, drafted_player: Player
) -> float | None:
    """Check if a pick violated position hierarchy constraints.

    A violation occurs when a player was drafted while a same-position player
    with higher AVG was still available on the draft board.

    Args:
        draft_state: Complete draft state
        pick_number: Pick number to check (0-based)
        drafted_player: Player that was drafted at this pick

    Returns:
        High regret score if hierarchy was violated, None otherwise
    """
    # Get the state of the draft board just before this pick
    cloned_draft = draft_state.clone()
    pre_pick_state = cloned_draft.rewind_to_pick(pick_number)

    # Find all available players of the same position with higher AVG
    available_better_players = []
    for player in pre_pick_state.draft_board.available_players:
        if (
            player.name not in pre_pick_state.draft_board.drafted_players
            and player.position == drafted_player.position
            and player.avg > drafted_player.avg
        ):
            available_better_players.append(player)

    if available_better_players:
        # Calculate high regret based on the difference in average scores
        best_available_avg = max(p.avg for p in available_better_players)
        avg_difference = best_available_avg - drafted_player.avg

        # Return a high regret score proportional to the missed opportunity
        # Use a multiplier to ensure this significantly penalizes hierarchy violations
        hierarchy_violation_regret = avg_difference * 10.0  # Multiplier for emphasis

        logger.debug(
            f"Hierarchy violation: {drafted_player.name} (AVG: {drafted_player.avg}) "
            f"drafted while {len(available_better_players)} better players available "
            f"(best AVG: {best_available_avg})"
        )

        return hierarchy_violation_regret

    return None


def calculate_pick_regret(original_draft: DraftState, pick_number: int) -> float:
    """Calculate regret score for a specific pick in the draft.

    Regret is calculated by comparing the team's total score in the original
    draft vs. a counterfactual draft where that pick is skipped and the draft
    continues with the greedy algorithm.

    Args:
        original_draft: Complete draft state from original simulation
        pick_number: Pick number to calculate regret for (0-based)

    Returns:
        Regret score (counterfactual_score - original_score)
        Higher values indicate more regret (worse pick)

    Raises:
        ValueError: If pick_number is invalid or pick wasn't made
    """
    if pick_number < 0 or pick_number >= len(original_draft.draft_history):
        raise ValueError(f"Invalid pick number: {pick_number}")

    # Get the originally drafted player and team
    original_pick_player = original_draft.draft_history[pick_number][1]
    team_idx = original_draft.pick_order[pick_number]
    original_team = original_draft.teams[team_idx]

    logger.debug(
        f"Calculating regret for pick {pick_number}: "
        f"{original_pick_player.name} to team {team_idx}"
    )

    # Get original team score
    original_score = original_team.calculate_total_score()

    # Check position hierarchy constraint first
    # If a better same-position player was available, assign high regret
    hierarchy_violation_regret = _check_position_hierarchy_violation(
        original_draft, pick_number, original_pick_player
    )

    if hierarchy_violation_regret is not None:
        logger.debug(
            f"Position hierarchy violation for pick {pick_number}: "
            f"{original_pick_player.name} - assigning high regret "
            f"{hierarchy_violation_regret:.2f}"
        )
        return hierarchy_violation_regret

    # Create counterfactual scenario:
    # 1. Clone original draft to avoid modifying it
    cloned_draft = original_draft.clone()

    # 2. Rewind to before this pick
    counterfactual_state = cloned_draft.rewind_to_pick(pick_number)

    # 3. Remove the originally drafted player from available pool
    # (This forces a different pick since the original player is unavailable)
    counterfactual_state.draft_board.drafted_players.add(original_pick_player.name)

    # 4. Simulate draft forward from this pick
    counterfactual_state = simulate_from_pick(counterfactual_state, pick_number)

    # 5. Get counterfactual team score
    counterfactual_team = counterfactual_state.teams[team_idx]
    counterfactual_score = counterfactual_team.calculate_total_score()

    regret = counterfactual_score - original_score

    logger.debug(
        f"Pick {pick_number} regret: {regret:.2f} "
        f"(original: {original_score:.2f}, counterfactual: {counterfactual_score:.2f})"
    )

    return regret


def calculate_all_regrets(draft_state: DraftState) -> Dict[str, float]:
    """Calculate regret scores for all picks in the completed draft.

    Args:
        draft_state: Complete draft state with all picks made

    Returns:
        Dictionary mapping player names to their regret scores
    """
    logger.info("Calculating regret scores for all picks...")

    player_regrets = {}

    for pick_number in range(len(draft_state.draft_history)):
        player_name = draft_state.draft_history[pick_number][1].name
        regret_score = calculate_pick_regret(draft_state, pick_number)
        player_regrets[player_name] = regret_score

        if pick_number % 20 == 0:  # Progress logging every 20 picks
            logger.debug(f"Calculated regret for {pick_number + 1} picks...")

    logger.info(f"Calculated regret scores for {len(player_regrets)} players")
    return player_regrets


def update_adp_from_regret(
    current_adp: Dict[str, float],
    player_regrets: Dict[str, float],
    learning_rate: float,
) -> Dict[str, float]:
    """Apply ADP update using raw regret scores directly.

    New_ADP = Old_ADP + η × regret_score
    Higher regret → later pick (higher ADP number)

    Args:
        current_adp: Current ADP values for all players
        player_regrets: Raw regret scores (in fantasy points)
        learning_rate: Learning rate η (e.g., 0.5)

    Returns:
        Updated ADP values before rescaling
    """
    updated_adp = current_adp.copy()

    for player_name in player_regrets:
        if player_name in current_adp:
            old_adp = current_adp[player_name]
            regret_adjustment = learning_rate * player_regrets[player_name]
            new_adp = old_adp + regret_adjustment  # Positive regret = later pick
            updated_adp[player_name] = new_adp

            logger.debug(
                f"{player_name}: ADP {old_adp:.2f} → {new_adp:.2f} "
                f"(regret: {player_regrets[player_name]:.3f})"
            )

    return updated_adp


def rescale_adp_to_picks(updated_adp: Dict[str, float]) -> Dict[str, float]:
    """Rescale ADP values to valid pick positions (1, 2, 3, ...).

    Sort all players by updated ADP values and assign sequential pick numbers.
    This maintains relative ordering while ensuring all ADPs are valid.

    Args:
        updated_adp: ADP values after raw update (may be outside valid range)

    Returns:
        Rescaled ADP values as sequential pick numbers
    """
    if not updated_adp:
        return {}

    # Sort players by ADP (lower ADP = earlier pick)
    sorted_players = sorted(updated_adp.items(), key=lambda x: x[1])

    # Assign sequential pick numbers
    rescaled_adp = {}
    for pick_number, (player_name, _) in enumerate(sorted_players, start=1):
        rescaled_adp[player_name] = float(pick_number)

    logger.debug(
        f"Rescaled ADP: {len(rescaled_adp)} players assigned picks 1-{len(rescaled_adp)}"
    )

    return rescaled_adp


def validate_position_hierarchy(
    updated_adp: Dict[str, float], players: list[Player]
) -> bool:
    """Validate that position hierarchy is maintained after ADP update.

    Within each position, players with higher AVG should have lower (earlier) ADP.

    Args:
        updated_adp: Updated ADP values to validate
        players: List of all players with their stats

    Returns:
        True if hierarchy is maintained, False if violated
    """
    # Group players by position
    by_position: dict[str, list[Player]] = {}
    for player in players:
        if player.name in updated_adp:
            if player.position not in by_position:
                by_position[player.position] = []
            by_position[player.position].append(player)

    # Check hierarchy within each position
    violations = 0
    for position, position_players in by_position.items():
        # Sort by ADP (earlier picks first)
        sorted_by_adp = sorted(position_players, key=lambda p: updated_adp[p.name])

        # Check that AVG scores are in descending order
        prev_avg = float("inf")
        for player in sorted_by_adp:
            if player.avg > prev_avg:
                logger.warning(
                    f"Position hierarchy violation: {player.name} "
                    f"(AVG: {player.avg}) has earlier ADP than higher-scoring player"
                )
                violations += 1
            prev_avg = player.avg

    is_valid = violations == 0
    if is_valid:
        logger.debug("Position hierarchy validation passed")
    else:
        logger.warning(f"Position hierarchy violations: {violations}")

    return is_valid


def check_convergence(
    initial_adp: Dict[str, float], final_adp: Dict[str, float]
) -> int:
    """Check convergence by counting total position moves between initial and final ADP.

    Convergence occurs when no players change positions after regret updates + rescaling.

    Args:
        initial_adp: ADP values before regret updates (pre-iteration)
        final_adp: ADP values after regret updates + rescaling (post-iteration)

    Returns:
        Total magnitude of position changes across all players (0 means converged)
        Example: Player moving from rank 21→23 contributes 2, player 5→5 contributes 0
    """
    if not initial_adp or not final_adp:
        return 0

    # Get rankings for both ADPs (lower ADP = better rank)
    initial_ranking = {
        player: rank
        for rank, (player, _) in enumerate(
            sorted(initial_adp.items(), key=lambda x: x[1]), start=1
        )
    }

    final_ranking = {
        player: rank
        for rank, (player, _) in enumerate(
            sorted(final_adp.items(), key=lambda x: x[1]), start=1
        )
    }

    # Count total position changes (magnitude of moves)
    position_changes = 0
    for player in initial_ranking:
        if player in final_ranking:
            initial_position = initial_ranking[player]
            final_position = final_ranking[player]
            position_changes += abs(final_position - initial_position)

    logger.info(f"Total position changes this iteration: {position_changes}")

    if position_changes == 0:
        logger.info("Convergence achieved: no position changes")

    return position_changes
