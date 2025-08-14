"""Main ADP optimization loop combining all phases."""

import logging

from optimal_adp.config import NUM_TEAMS
from optimal_adp.data_io import (
    compute_initial_adp,
    create_run_directory,
    load_player_data,
    save_convergence_history_csv,
    save_final_adp_csv,
    save_initial_vbr_adp_csv,
    save_regrets_csv,
    save_run_parameters_txt,
    save_team_scores_csv,
)
from optimal_adp.models import simulate_full_draft, DraftState, Player
from optimal_adp.regret import (
    calculate_all_regrets,
    rescale_adp_to_picks,
    check_convergence,
    validate_position_hierarchy_detailed,
    update_adp_from_regret_constrained,
)
from optimal_adp.validation import (
    perturb_initial_adp,
    validate_optimization_results,
)

logger = logging.getLogger(__name__)


def get_position_changes_detailed(
    old_adp: dict[str, float], new_adp: dict[str, float], players: list[Player]
) -> list[str]:
    """Get detailed list of all position changes between iterations.

    Args:
        old_adp: Previous ADP values
        new_adp: Current ADP values
        players: List of all players for context

    Returns:
        List of strings describing position changes
    """
    changes = []
    player_lookup = {player.name: player for player in players}

    for player_name in old_adp:
        if player_name in new_adp:
            old_pos = int(round(old_adp[player_name]))
            new_pos = int(round(new_adp[player_name]))

            if old_pos != new_pos:
                player = player_lookup.get(player_name)
                position = player.position if player else "UNK"
                avg = f"{player.avg:.1f}" if player else "N/A"

                direction = "↑" if new_pos < old_pos else "↓"
                move_size = abs(new_pos - old_pos)

                changes.append(
                    f"  {direction} {player_name} ({position}, AVG:{avg}): "
                    f"ADP {old_pos} → {new_pos} ({move_size} spots)"
                )

    # Sort by magnitude of change (largest moves first)
    changes.sort(key=lambda x: int(x.split("(")[-1].split(" spots")[0]), reverse=True)
    return changes


def optimize_adp(
    players: list[Player],
    initial_adp: dict[str, float],
    num_teams: int = 12,
    max_iterations: int = 1000,
    learning_rate: float = 0.1,
) -> tuple[dict[str, float], list[int], int, dict[str, float], DraftState | None]:
    """Run pure ADP optimization algorithm without I/O operations.

    Args:
        players: List of all player data
        initial_adp: Initial ADP values for all players
        num_teams: Number of teams in draft
        max_iterations: Maximum optimization iterations
        learning_rate: Learning rate for ADP updates

    Returns:
        Tuple of:
        - Final ADP values
        - Convergence history (position changes per iteration)
        - Number of iterations completed
        - Final regret values for all players
        - Final draft state
    """
    logger.info("Starting ADP optimization")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Learning rate: {learning_rate}")

    # Use provided data
    all_players = players
    current_adp = dict(initial_adp)  # Make a copy
    logger.info(
        f"Using {len(all_players)} players with initial ADP for {len(current_adp)} players"
    )

    # Track convergence history
    convergence_history: list[int] = []
    iterations_completed = 0
    final_regrets: dict[str, float] = {}
    final_draft_state = None

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

        # Store final regret values and draft state for the last iteration
        final_regrets = player_regrets
        final_draft_state = draft_state

        # 3c: Update ADP with regret scores (with hierarchy constraints)
        logger.debug("Updating ADP from regret scores")

        updated_adp = update_adp_from_regret_constrained(
            current_adp, player_regrets, learning_rate, all_players
        )

        # 3d: Rescale to valid pick numbers
        logger.debug("Rescaling ADP to valid pick numbers")
        current_adp = rescale_adp_to_picks(updated_adp)

        # 3e: Validate position hierarchy after each iteration
        hierarchy_valid, violations = validate_position_hierarchy_detailed(
            current_adp, all_players
        )
        if not hierarchy_valid:
            logger.warning(
                f"Position hierarchy validation failed at iteration {iteration + 1} "
                f"({len(violations)} violations)"
            )
            # Log first few violations for debugging
            for i, violation in enumerate(violations[:3]):
                logger.warning(f"  Violation {i+1}: {violation}")
            if len(violations) > 3:
                logger.warning(f"  ... and {len(violations) - 3} more violations")
        else:
            logger.info(
                f"Position hierarchy validation passed at iteration {iteration + 1}"
            )

        iterations_completed = iteration + 1

        # 3f: Check convergence and show detailed position changes
        position_changes = check_convergence(iteration_start_adp, current_adp)
        convergence_history.append(position_changes)

        logger.info(f"Iteration {iteration + 1}: {position_changes} position changes")

        # Show detailed position changes if any occurred
        if position_changes > 0:
            detailed_changes = get_position_changes_detailed(
                iteration_start_adp, current_adp, all_players
            )
            if detailed_changes:
                logger.info(f"Position changes in iteration {iteration + 1}:")
                for change in detailed_changes[:10]:  # Show up to 10 largest changes
                    logger.info(change)
                if len(detailed_changes) > 10:
                    logger.info(f"  ... and {len(detailed_changes) - 10} more changes")

        if position_changes == 0:
            logger.info(f"Convergence achieved at iteration {iteration + 1}")
            break

    # Complete optimization
    logger.info(f"Optimization completed after {iterations_completed} iterations")

    logger.info("ADP optimization completed successfully")
    return (
        current_adp,
        convergence_history,
        iterations_completed,
        final_regrets,
        final_draft_state,
    )


def run_optimization_with_validation_and_io(
    data_file_path: str,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    num_teams: int = NUM_TEAMS,
    perturbation_factor: float = 0.1,
    artifacts_outputs: bool = True,
) -> bool:
    """Run complete optimization process with I/O, validation, and artifacts.

    This function centralizes all I/O operations (CSV reading, artifact writing)
    and provides the same interface as the old validate_optimization function.

    Args:
        data_file_path: Path to player data CSV
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations to allow
        num_teams: Number of teams in draft
        perturbation_factor: Amount of perturbation to apply
        artifacts_outputs: Whether to save optimization artifacts

    Returns:
        True if all validations pass, False otherwise
    """
    try:
        # Step 1: Load and prepare data (centralized I/O)
        players = load_player_data(data_file_path)
        initial_adp_data = compute_initial_adp(players)
        initial_adp = {player.name: float(adp) for player, _, adp in initial_adp_data}

        if perturbation_factor:
            initial_adp_list = [
                (player, vbr, adp) for player, vbr, adp in initial_adp_data
            ]
            perturbed_list = perturb_initial_adp(initial_adp_list, perturbation_factor)
            initial_adp = {player.name: float(adp) for player, _, adp in perturbed_list}
            initial_adp_data = perturbed_list

        # Step 2: Run optimization
        (
            final_adp,
            convergence_history,
            iterations,
            regret_values,
            draft_state,
        ) = optimize_adp(
            players=players,
            initial_adp=initial_adp,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            num_teams=num_teams,
        )

        # Step 3: Run validation using simplified function
        result = validate_optimization_results(
            players=players,
            final_adp=final_adp,
            iterations=iterations,
            max_iterations=max_iterations,
            num_teams=num_teams,
        )

        # Step 4: Save artifacts if requested (centralized I/O)
        if artifacts_outputs:
            run_id, run_dir = create_run_directory(learning_rate, max_iterations)

            # Save initial VBR-based ADP
            save_initial_vbr_adp_csv(run_dir / "initial_vbr_adp.csv", initial_adp_data)

            # Save final optimized ADP
            save_final_adp_csv(
                str(run_dir / "final_adp.csv"),
                players,
                final_adp,
                draft_state,
                num_teams,
            )

            # Save other artifacts
            save_convergence_history_csv(
                run_dir / "convergence_history.csv", convergence_history
            )
            save_team_scores_csv(run_dir / "team_scores.csv", draft_state)
            save_regrets_csv(run_dir / "regrets.csv", regret_values, final_adp)
            save_run_parameters_txt(
                run_dir / "run_parameters.txt",
                run_id,
                data_file_path,
                learning_rate,
                max_iterations,
                num_teams,
                perturbation_factor,
                iterations,
                convergence_history,
            )

        # Step 5: Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION VALIDATION RESULTS")
        print("=" * 60)

        for message in result.messages:
            print(message)

        if iterations is not None:
            print(f"\nConvergence: {iterations} iterations")

        print("=" * 60)

        return result.all_passed()

    except Exception as e:
        logging.error(f"Optimization with validation failed: {e}")
        return False
