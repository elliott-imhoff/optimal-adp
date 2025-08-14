"""Main optimization module with validation and result archiving."""

import csv
import logging
import random
import os
from datetime import datetime
from pathlib import Path

from optimal_adp.config import NUM_TEAMS, Player
from optimal_adp.data_io import load_player_data, compute_initial_adp
from optimal_adp.optimizer import optimize_adp

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation test results."""

    def __init__(self) -> None:
        """Initialize validation result container."""
        self.passed: bool = True
        self.messages: list[str] = []
        self.convergence_iterations: int | None = None
        self.final_position_changes: int | None = None
        self.convergence_history: list[int] = []
        self.run_id: str | None = None
        self.artifacts_dir: Path | None = None

    def add_failure(self, message: str) -> None:
        """Add a validation failure message."""
        self.passed = False
        self.messages.append(f"❌ {message}")
        logger.warning(message)

    def add_success(self, message: str) -> None:
        """Add a validation success message."""
        self.messages.append(f"✅ {message}")
        logger.info(message)

    def add_info(self, message: str) -> None:
        """Add informational message."""
        self.messages.append(f"ℹ️  {message}")
        logger.info(message)


def create_run_directory(learning_rate: float, max_iterations: int) -> tuple[str, Path]:
    """Create a timestamped directory for this optimization run.

    Args:
        learning_rate: Learning rate used for optimization
        max_iterations: Maximum iterations for optimization

    Returns:
        Tuple of (run_id, artifacts_directory_path)
    """
    # Generate run ID based on current timestamp and parameters
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_lr{learning_rate}_iter{max_iterations}"

    # Create artifacts directory structure
    artifacts_base = Path("artifacts")
    run_dir = artifacts_base / f"run_{run_id}"

    # Create directories if they don't exist
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")
    return run_id, run_dir


def perturb_initial_adp(
    initial_adp_data: list[tuple[Player, float, int]],
    perturbation_factor: float = 0.1,
) -> list[tuple[Player, float, int]]:
    """Randomly perturb initial ADP values slightly.

    Args:
        initial_adp_data: List of (player, vbr, adp) tuples
        perturbation_factor: Maximum relative change to apply (0.1 = 10%)

    Returns:
        Perturbed initial ADP data with same structure
    """
    if perturbation_factor == 0.0:
        # No perturbation - return copy of original
        return list(initial_adp_data)

    perturbed = []
    for player, vbr, adp in initial_adp_data:
        # Apply random perturbation to ADP value
        perturbation = random.uniform(-perturbation_factor, perturbation_factor)
        new_adp = adp * (1 + perturbation)
        # Ensure ADP stays positive and convert back to int
        new_adp = max(1, round(new_adp))
        perturbed.append((player, vbr, new_adp))

    # Re-sort by new ADP values to maintain relative order
    perturbed.sort(key=lambda x: x[2])

    return perturbed


def validate_position_hierarchy(
    final_adp: dict[str, float], players: list[Player]
) -> tuple[bool, list[str]]:
    """Validate that same-position players are ranked by AVG score.

    Args:
        final_adp: Final ADP values mapping player names to pick numbers
        players: List of all players with stats

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    # Group players by position
    players_by_position: dict[str, list[Player]] = {}
    for player in players:
        if player.name in final_adp:
            if player.position not in players_by_position:
                players_by_position[player.position] = []
            players_by_position[player.position].append(player)

    # Check hierarchy within each position
    for position, position_players in players_by_position.items():
        # Sort by ADP (lower = better)
        position_players.sort(key=lambda p: final_adp[p.name])

        # Check that AVG scores are in descending order
        for i in range(len(position_players) - 1):
            current_player = position_players[i]
            next_player = position_players[i + 1]

            if current_player.avg < next_player.avg:
                violations.append(
                    f"{position}: {current_player.name} (AVG: {current_player.avg:.1f}, "
                    f"ADP: {final_adp[current_player.name]:.1f}) ranked before "
                    f"{next_player.name} (AVG: {next_player.avg:.1f}, "
                    f"ADP: {final_adp[next_player.name]:.1f})"
                )

    return len(violations) == 0, violations


def validate_elite_players_first_round(
    final_adp: dict[str, float], players: list[Player], num_teams: int = NUM_TEAMS
) -> tuple[bool, list[str]]:
    """Validate that top QB, RB, and WR are drafted in first round.

    Args:
        final_adp: Final ADP values mapping player names to pick numbers
        players: List of all players with stats
        num_teams: Number of teams (first round = picks 1 to num_teams)

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []
    first_round_cutoff = num_teams  # First round is picks 1 through num_teams

    # Find top player by AVG in each major position
    positions_to_check = ["QB", "RB", "WR"]

    for position in positions_to_check:
        position_players = [
            p for p in players if p.position == position and p.name in final_adp
        ]

        if not position_players:
            violations.append(f"No {position} players found in final ADP")
            continue

        # Find top player by AVG score
        top_player = max(position_players, key=lambda p: p.avg)
        player_adp = final_adp[top_player.name]

        if player_adp > first_round_cutoff:
            violations.append(
                f"Top {position} {top_player.name} (AVG: {top_player.avg:.1f}) "
                f"has ADP {player_adp:.1f}, should be ≤{first_round_cutoff}"
            )

    return len(violations) == 0, violations


def validate_convergence(
    data_file_path: str,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    num_teams: int = NUM_TEAMS,
    enable_perturbation: bool = False,
    perturbation_factor: float = 0.1,
    artifacts_outputs: bool = True,
) -> ValidationResult:
    """Run optimization with validation and optional result archiving.

    Args:
        data_file_path: Path to player data CSV
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations to allow
        num_teams: Number of teams in draft
        enable_perturbation: Whether to perturb initial ADP values
        perturbation_factor: Amount of perturbation to apply
        artifacts_outputs: Whether to artifacts optimization outputs

    Returns:
        ValidationResult with test outcomes
    """
    result = ValidationResult()

    try:
        # Create run directory if archiving
        if artifacts_outputs:
            run_id, artifacts_dir = create_run_directory(learning_rate, max_iterations)
            result.run_id = run_id
            result.artifacts_dir = artifacts_dir
            result.add_info(f"Run ID: {run_id}")
            result.add_info(f"Artifacts directory: {artifacts_dir}")

        # Load player data
        result.add_info(f"Loading player data from {data_file_path}")
        players = load_player_data(data_file_path)
        result.add_info(f"Loaded {len(players)} players")

        # Handle perturbation if requested
        if enable_perturbation:
            result.add_info(f"Applying perturbation (factor: {perturbation_factor})")
            # Get initial ADP data and perturb it
            initial_adp_data = compute_initial_adp(players)
            initial_adp_data = perturb_initial_adp(
                initial_adp_data, perturbation_factor
            )

        result.add_info(
            f"Running optimization (learning_rate={learning_rate}, max_iterations={max_iterations})"
        )

        # Set up output file path
        if artifacts_outputs and result.artifacts_dir:
            output_file = result.artifacts_dir / "final_adp.csv"
        else:
            # Use temporary file if not archiving
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                output_file = Path(tmp_file.name)

        try:
            (
                final_adp,
                convergence_history,
                iterations,
                final_regrets,
                team_scores,
            ) = optimize_adp(
                data_file_path=data_file_path,
                num_teams=num_teams,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                output_file_path=str(output_file),
            )

            # Save additional run metadata if archiving
            if artifacts_outputs and result.artifacts_dir:
                # Save run parameters
                params_file = result.artifacts_dir / "run_parameters.txt"
                with open(params_file, "w") as f:
                    f.write(f"Run ID: {result.run_id}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Data file: {data_file_path}\n")
                    f.write(f"Learning rate: {learning_rate}\n")
                    f.write(f"Max iterations: {max_iterations}\n")
                    f.write(f"Number of teams: {num_teams}\n")
                    f.write(f"Perturbation enabled: {enable_perturbation}\n")
                    f.write(f"Perturbation factor: {perturbation_factor}\n")
                    f.write(f"Final iterations: {iterations}\n")
                    f.write(
                        f"Final position changes: {convergence_history[-1] if convergence_history else 'N/A'}\n"
                    )

                # Save convergence history
                history_file = result.artifacts_dir / "convergence_history.csv"
                with open(history_file, "w") as f:
                    f.write("iteration,position_changes\n")
                    for i, changes in enumerate(convergence_history, 1):
                        f.write(f"{i},{changes}\n")

                # Save final regret values (sorted by ADP)
                regrets_file = result.artifacts_dir / "final_regrets.csv"
                with open(regrets_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["player_name", "regret_score", "final_adp"])
                    # Create list of (player_name, regret, adp) and sort by ADP
                    regret_data = []
                    for player_name, regret in final_regrets.items():
                        adp_value = final_adp.get(player_name, float("inf"))
                        regret_data.append((player_name, regret, adp_value))

                    # Sort by ADP
                    regret_data.sort(key=lambda x: x[2])

                    for player_name, regret, adp_value in regret_data:
                        writer.writerow([player_name, regret, adp_value])

                # Save team scores
                team_scores_file = result.artifacts_dir / "team_scores.csv"
                with open(team_scores_file, "w", newline="") as f:
                    team_writer = csv.DictWriter(
                        f, fieldnames=["team_id", "total_score", "avg_per_week"]
                    )
                    team_writer.writeheader()
                    team_writer.writerows(team_scores)

                result.add_info(f"Results artifactsd to: {result.artifacts_dir}")
            else:
                # Clean up temp file if not archiving
                if not artifacts_outputs:
                    os.unlink(str(output_file))

        except Exception as e:
            result.add_failure(f"Optimization failed: {str(e)}")
            return result

        # Store convergence metrics
        result.convergence_iterations = iterations
        result.final_position_changes = (
            convergence_history[-1] if convergence_history else None
        )
        result.convergence_history = convergence_history

        # Validate convergence
        if result.final_position_changes == 0:
            result.add_success(f"Converged after {iterations} iterations")
        elif iterations >= max_iterations:
            result.add_failure(
                f"Did not converge within {max_iterations} iterations (final position changes: {result.final_position_changes})"
            )
        else:
            result.add_success(f"Converged after {iterations} iterations")

        # Validate position hierarchy
        hierarchy_valid, hierarchy_violations = validate_position_hierarchy(
            final_adp, players
        )
        if hierarchy_valid:
            result.add_success(
                "Position hierarchy maintained (same-position players ranked by AVG)"
            )
        else:
            result.add_failure(
                f"Position hierarchy violations: {len(hierarchy_violations)}"
            )
            for violation in hierarchy_violations[:5]:  # Show first 5 violations
                result.add_failure(f"  {violation}")
            if len(hierarchy_violations) > 5:
                result.add_failure(
                    f"  ... and {len(hierarchy_violations) - 5} more violations"
                )

        # Validate elite players in first round
        elite_valid, elite_violations = validate_elite_players_first_round(
            final_adp, players, num_teams
        )
        if elite_valid:
            result.add_success("Top QB, RB, and WR all drafted in first round")
        else:
            result.add_failure("Elite players not in first round:")
            for violation in elite_violations:
                result.add_failure(f"  {violation}")

        # Summary
        if result.passed:
            result.add_success("🎉 All validation checks passed!")
        else:
            result.add_failure("❌ Validation failed - see issues above")

    except Exception as e:
        result.add_failure(f"Validation error: {str(e)}")
        logger.exception("Validation failed with exception")

    return result


def validate_optimization(
    data_file_path: str,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    num_teams: int = NUM_TEAMS,
    enable_perturbation: bool = False,
    perturbation_factor: float = 0.1,
    artifacts_outputs: bool = True,
) -> bool:
    """Run optimization with validation and result archiving.

    Args:
        data_file_path: Path to player data CSV
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations to allow
        num_teams: Number of teams in draft
        enable_perturbation: Whether to perturb initial ADP values
        perturbation_factor: Amount of perturbation to apply
        artifacts_outputs: Whether to artifacts optimization outputs

    Returns:
        True if all validations pass, False otherwise
    """
    result = validate_convergence(
        data_file_path=data_file_path,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        num_teams=num_teams,
        enable_perturbation=enable_perturbation,
        perturbation_factor=perturbation_factor,
        artifacts_outputs=artifacts_outputs,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION VALIDATION RESULTS")
    print("=" * 60)

    for message in result.messages:
        print(message)

    if result.convergence_iterations is not None:
        print(f"\nConvergence: {result.convergence_iterations} iterations")
    if result.final_position_changes is not None:
        print(f"Final position changes: {result.final_position_changes}")

    print("=" * 60)

    return result.passed
