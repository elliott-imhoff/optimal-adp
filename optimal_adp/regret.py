"""Regret calculation and ADP optimization for fantasy football drafts."""

import logging

from .models import DraftState, Player

logger = logging.getLogger(__name__)


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

    # Create counterfactual scenario:
    # 1. Clone original draft to avoid modifying it
    cloned_draft = original_draft.clone()

    # 2. Rewind to before this pick
    counterfactual_state = cloned_draft.rewind_to_pick(pick_number)

    # 3. Remove the originally drafted player from available pool
    # (This forces a different pick since the original player is unavailable)
    counterfactual_state.draft_board.drafted_players.add(original_pick_player.name)

    # 4. Simulate draft forward from this pick
    counterfactual_state = counterfactual_state.simulate_from_pick(pick_number)

    # 5. Get counterfactual team score
    counterfactual_team = counterfactual_state.teams[team_idx]
    counterfactual_score = counterfactual_team.calculate_total_score()

    regret = counterfactual_score - original_score

    logger.debug(
        f"Pick {pick_number} regret: {regret:.2f} "
        f"(original: {original_score:.2f}, counterfactual: {counterfactual_score:.2f})"
    )

    return regret


def calculate_all_regrets(draft_state: DraftState) -> dict[str, float]:
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


def update_adp_from_regret_constrained(
    current_adp: dict[str, float],
    player_regrets: dict[str, float],
    learning_rate: float,
    all_players: list[Player],
) -> dict[str, float]:
    """Apply ADP update with position hierarchy constraints.

    Updates ADP while ensuring players with higher AVG within the same position
    maintain earlier (lower) ADP values by swapping violators.

    Args:
        current_adp: Current ADP values for all players
        player_regrets: Raw regret scores (in fantasy points)
        learning_rate: Learning rate η (e.g., 0.5)
        all_players: All players with their stats for hierarchy constraints

    Returns:
        Updated ADP values that respect position hierarchy
    """
    # First, apply unconstrained updates
    updated_adp = current_adp.copy()

    for player_name in player_regrets:
        if player_name in current_adp:
            old_adp = current_adp[player_name]
            regret_adjustment = learning_rate * player_regrets[player_name]
            new_adp = old_adp + regret_adjustment
            updated_adp[player_name] = new_adp

    # Create player lookup for constraints
    player_lookup = {p.name: p for p in all_players}

    # Find and fix hierarchy violations by swapping ADP values
    swaps_made = 0
    for player1_name in updated_adp:
        if player1_name not in player_lookup:
            continue

        player1 = player_lookup[player1_name]
        player1_adp = updated_adp[player1_name]

        for player2_name in updated_adp:
            if player2_name not in player_lookup or player1_name == player2_name:
                continue

            player2 = player_lookup[player2_name]
            player2_adp = updated_adp[player2_name]

            # Check if same position and hierarchy violation exists
            should_swap = False
            if player1.position == player2.position and player1_adp > player2_adp:
                if player1.avg > player2.avg:
                    # Higher AVG should have earlier ADP
                    should_swap = True
                elif player1.avg == player2.avg and player1_name < player2_name:
                    # Tie-breaker: alphabetically earlier name should have earlier ADP
                    should_swap = True

            if should_swap:
                # Swap their ADP values
                updated_adp[player1_name] = player2_adp
                updated_adp[player2_name] = player1_adp
                swaps_made += 1

                tie_breaker = " (tie-breaker)" if player1.avg == player2.avg else ""
                logger.debug(
                    f"Hierarchy fix: Swapped {player1_name} (AVG: {player1.avg:.1f}) "
                    f"and {player2_name} (AVG: {player2.avg:.1f}){tie_breaker} - "
                    f"ADPs: {player1_adp:.2f} ↔ {player2_adp:.2f}"
                )

    if swaps_made > 0:
        logger.debug(f"Made {swaps_made} ADP swaps to maintain position hierarchy")

    return updated_adp


def rescale_adp_to_picks(updated_adp: dict[str, float]) -> dict[str, float]:
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


def check_convergence(
    initial_adp: dict[str, float], final_adp: dict[str, float]
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
