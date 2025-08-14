"""Validation functions to verify optimization results without I/O operations."""

import logging
from pathlib import Path

from optimal_adp.config import NUM_TEAMS
from optimal_adp.models import Player

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

    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return self.passed

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        if not other.passed:
            self.passed = False
        self.messages.extend(other.messages)


def validate_position_hierarchy(
    final_adp: dict[str, float], players: list[Player], detailed: bool = True
) -> tuple[bool, list[str]]:
    """Validate that same-position players are ranked by AVG score.

    Within each position, players with higher AVG should have lower (earlier) ADP.
    This is the consolidated function that replaces overlapping implementations
    from regret and validation modules.

    Args:
        final_adp: Final ADP values mapping player names to pick numbers
        players: List of all players with stats
        detailed: If True, return detailed violation messages; if False, just log count

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
        # Sort by ADP (lower = better/earlier)
        position_players.sort(key=lambda p: final_adp[p.name])

        # Check that AVG scores are in descending order
        for i in range(len(position_players) - 1):
            current_player = position_players[i]
            next_player = position_players[i + 1]

            if current_player.avg < next_player.avg:
                violation_msg = (
                    f"{position}: {current_player.name} (AVG: {current_player.avg:.1f}, "
                    f"ADP: {final_adp[current_player.name]:.1f}) ranked before "
                    f"{next_player.name} (AVG: {next_player.avg:.1f}, "
                    f"ADP: {final_adp[next_player.name]:.1f})"
                )
                violations.append(violation_msg)
                if detailed:
                    logger.warning(f"Position hierarchy violation: {violation_msg}")

    is_valid = len(violations) == 0
    if is_valid:
        logger.debug("Position hierarchy validation passed")
    elif not detailed:
        logger.warning(f"Position hierarchy violations: {len(violations)}")

    return is_valid, violations


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


def validate_convergence_criteria(
    iterations: int, max_iterations: int = 1000
) -> ValidationResult:
    """
    Validate that optimization converged within expected iteration count.

    Args:
        iterations: Number of iterations completed
        max_iterations: Maximum allowed iterations

    Returns:
        ValidationResult with success/failure status
    """
    result = ValidationResult()

    if iterations == 0:
        result.add_failure("Optimization failed: no iterations completed")
        return result

    if iterations < max_iterations:
        result.add_success(f"Converged after {iterations} iterations")
    else:
        result.add_failure(f"Failed to converge within {max_iterations} iterations")

    return result


def validate_optimization_results(
    players: list[Player],
    final_adp: dict[str, float],
    iterations: int,
    max_iterations: int,
    num_teams: int = NUM_TEAMS,
) -> ValidationResult:
    """
    Run all validation checks on optimization results.

    Args:
        players: List of all players with stats
        final_adp: Final ADP values after optimization
        iterations: Number of iterations completed
        max_iterations: Maximum iterations allowed
        num_teams: Number of teams in draft

    Returns:
        ValidationResult with combined validation status
    """
    result = ValidationResult()

    try:
        # Check convergence criteria
        convergence_result = validate_convergence_criteria(iterations, max_iterations)
        result.merge(convergence_result)

        # Validate position hierarchy
        is_hierarchy_valid, hierarchy_violations = validate_position_hierarchy(
            final_adp, players
        )
        if is_hierarchy_valid:
            result.add_success("Position hierarchy maintained")
        else:
            result.add_failure("Position hierarchy violations found")
            for violation in hierarchy_violations:
                result.add_failure(violation)

        # Validate elite players placement
        is_elite_valid, elite_violations = validate_elite_players_first_round(
            final_adp, players, num_teams
        )
        if is_elite_valid:
            result.add_success("Top QB, RB, and WR all drafted in first round")
        else:
            result.add_failure("Elite players not in first round:")
            for violation in elite_violations:
                result.add_failure(f"  {violation}")

        # Final summary
        if result.all_passed():
            result.add_success("🎉 All validation checks passed!")
        else:
            result.add_failure("❌ Validation failed - see issues above")

    except Exception as e:
        result.add_failure(f"Validation error: {str(e)}")

    return result
