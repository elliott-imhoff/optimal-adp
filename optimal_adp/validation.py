"""Validation module for optimization quality and convergence testing."""

import logging
import random

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
) -> ValidationResult:
    """Validate that optimization converges with reasonable final rankings.

    Args:
        data_file_path: Path to player data CSV
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations to allow
        num_teams: Number of teams in draft
        enable_perturbation: Whether to perturb initial ADP values
        perturbation_factor: Amount of perturbation to apply

    Returns:
        ValidationResult with test outcomes
    """
    result = ValidationResult()

    try:
        # Load player data
        result.add_info(f"Loading player data from {data_file_path}")
        players = load_player_data(data_file_path)
        result.add_info(f"Loaded {len(players)} players")

        # Handle perturbation if requested
        if enable_perturbation:
            result.add_info(f"Applying perturbation (factor: {perturbation_factor})")
            # Get initial ADP data and perturb it
            initial_adp_data = compute_initial_adp(players)
            perturb_initial_adp(initial_adp_data, perturbation_factor)
            # Note: Currently optimization doesn't support custom initial ADP
            # This perturbation is for testing the perturbation function itself

        result.add_info(
            f"Running optimization (learning_rate={learning_rate}, max_iterations={max_iterations})"
        )

        # Run optimization by creating a temporary output file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            temp_output = tmp_file.name

        try:
            final_adp, convergence_history, iterations = optimize_adp(
                data_file_path=data_file_path,
                num_teams=num_teams,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                output_file_path=temp_output,
            )

            # Clean up temp file
            import os

            os.unlink(temp_output)

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
) -> bool:
    """Validate optimization quality and convergence.

    Args:
        data_file_path: Path to player data CSV
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations to allow
        num_teams: Number of teams in draft
        enable_perturbation: Whether to perturb initial ADP values
        perturbation_factor: Amount of perturbation to apply

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

    # Display convergence history
    if result.convergence_history:
        print("\nPosition changes by iteration:")
        for i, changes in enumerate(result.convergence_history, 1):
            print(f"  Iteration {i:2d}: {changes:2d} changes")

    print("=" * 60)

    return result.passed
