"""Main ADP optimization loop combining all phases."""

import csv
import logging
from typing import Dict, List, Tuple

from optimal_adp.config import NUM_TEAMS
from optimal_adp.data_io import load_player_data, compute_initial_adp
from optimal_adp.draft_simulator import simulate_full_draft
from optimal_adp.regret_optimizer import (
    calculate_all_regrets,
    update_adp_from_regret,
    rescale_adp_to_picks,
    check_convergence,
    validate_position_hierarchy,
)

logger = logging.getLogger(__name__)


def optimize_adp(
    data_file_path: str,
    num_teams: int = NUM_TEAMS,
    max_iterations: int = 50,
    learning_rate: float = 0.1,
    output_file_path: str = "artifacts/final_adp.csv",
) -> Tuple[Dict[str, float], List[int], int]:
    """Optimize ADP values using regret minimization.

    Iteratively optimizes ADP values using regret minimization until convergence
    or maximum iterations reached.

    Args:
        data_file_path: Path to CSV file containing player statistics
        num_teams: Number of teams in the draft (defaults to NUM_TEAMS)
        max_iterations: Maximum number of optimization iterations
        learning_rate: Learning rate for ADP updates (η in update formula)
        output_file_path: Path to write final ADP results

    Returns:
        Tuple of (final_adp, convergence_history, iterations_completed)
        - final_adp: Dict mapping player names to final ADP values
        - convergence_history: List of position changes per iteration
        - iterations_completed: Number of iterations actually completed
    """
    logger.info("Starting ADP optimization")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Learning rate: {learning_rate}")

    # Step 1: Load and filter player data
    logger.info(f"Loading player data from {data_file_path}")
    all_players = load_player_data(data_file_path)
    logger.info(f"Loaded {len(all_players)} players")

    # Step 2: Compute initial VBR-based ADP
    logger.info("Computing initial VBR-based ADP")
    initial_adp_data = compute_initial_adp(all_players)
    current_adp = {player.name: float(adp) for player, vbr, adp in initial_adp_data}
    logger.info(f"Initial ADP computed for {len(current_adp)} players")

    # Step 2.1: Save initial VBR-based ADP as artifact
    from pathlib import Path

    output_path = Path(output_file_path)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_adp_path = output_path.parent / "initial_vbr_adp.csv"
    logger.info(f"Saving initial VBR-based ADP to {initial_adp_path}")

    # Create initial ADP list for CSV output
    initial_adp_players = []
    for player, vbr, adp in initial_adp_data:
        player_dict = {
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "avg": player.avg,
            "total": player.total,
            "vbr": float(vbr),
            "adp": float(adp),
        }
        initial_adp_players.append(player_dict)

    # Write initial ADP CSV
    with open(initial_adp_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "position", "team", "avg", "total", "vbr", "adp"]
        )
        writer.writeheader()
        writer.writerows(initial_adp_players)

    # Track convergence history
    convergence_history: List[int] = []
    iterations_completed = 0

    # Step 3: Main optimization loop
    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
        iteration_start_adp = current_adp.copy()

        # 3a: Simulate draft with current ADP
        logger.debug("Simulating draft with current ADP")
        draft_state = simulate_full_draft(all_players, current_adp, num_teams)
        logger.debug(f"Draft completed with {len(draft_state.draft_history)} picks")

        # 3b: Calculate regret scores for all picks
        logger.debug("Calculating regret scores for all picks")
        player_regrets = calculate_all_regrets(draft_state)
        logger.debug(f"Regret calculated for {len(player_regrets)} players")

        # 3c: Update ADP with regret scores
        logger.debug("Updating ADP from regret scores")
        updated_adp = update_adp_from_regret(current_adp, player_regrets, learning_rate)

        # 3d: Rescale to valid pick numbers
        logger.debug("Rescaling ADP to valid pick numbers")
        current_adp = rescale_adp_to_picks(updated_adp)

        # 3e: Validate position hierarchy
        hierarchy_valid = validate_position_hierarchy(current_adp, all_players)
        if not hierarchy_valid:
            logger.warning(
                f"Position hierarchy validation failed at iteration {iteration + 1}"
            )

        iterations_completed = iteration + 1

        # 3f: Check convergence every iteration
        position_changes = check_convergence(iteration_start_adp, current_adp)
        convergence_history.append(position_changes)

        logger.info(f"Iteration {iteration + 1}: {position_changes} position changes")

        if position_changes == 0:
            logger.info(f"Convergence achieved at iteration {iteration + 1}")
            break

    # Step 4: Save final results
    logger.info(f"Optimization completed after {iterations_completed} iterations")
    logger.info(f"Writing final ADP to {output_file_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create simple list for CSV output
    adp_players = []
    for player in all_players:
        if player.name in current_adp:
            # Create a simple player dict for CSV writing
            player_dict = {
                "name": player.name,
                "position": player.position,
                "team": player.team,
                "avg": player.avg,
                "total": player.total,
                "adp": current_adp[player.name],
            }
            adp_players.append(player_dict)

    # Write CSV manually since we have a dict format
    with open(output_file_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "position", "team", "avg", "total", "adp"]
        )
        writer.writeheader()
        writer.writerows(adp_players)

    logger.info("ADP optimization completed successfully")
    return current_adp, convergence_history, iterations_completed
