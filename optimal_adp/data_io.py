"""Data input/output module for player statistics and ADP values."""

import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random

from optimal_adp.config import BASELINE_POSITIONS
from optimal_adp.models import DraftState, Player

logger = logging.getLogger(__name__)


def load_player_data(
    csv_path: str, min_weeks: int = 10, top_n_by_total: int = 150
) -> list[Player]:
    """Load and filter player data from CSV file.

    Args:
        csv_path: Path to CSV file with player statistics
        min_weeks: Minimum weeks played to include player (calculated from total/avg)
        top_n_by_total: Keep only top N players by total points

    Returns:
        List of filtered Player objects

    Filters applied:
        - Include only positions defined in BASELINE_POSITIONS (QB, RB, WR, TE)
        - Exclude players outside top N by total points
        - Exclude players with < min_weeks played (calculated from total/avg)
    """
    players = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip empty rows or rows with missing essential data
            if not row or not row.get("Player") or not row.get("Pos"):
                continue

            # Only include positions that have baseline values defined
            if row["Pos"] not in BASELINE_POSITIONS.keys():
                continue

            # Convert numeric fields, handling None values and empty strings
            try:
                avg_val = row.get("AVG")
                total_val = row.get("TTL")

                # Skip if values are None or empty strings
                if (
                    avg_val is None
                    or total_val is None
                    or avg_val == ""
                    or total_val == ""
                ):
                    continue

                avg = float(avg_val)
                total = float(total_val)
            except (ValueError, TypeError, KeyError):
                continue  # Skip rows with invalid data

            # Calculate weeks played from total/average (avoid division by zero)
            if avg == 0:
                weeks_played = 0
            else:
                weeks_played = round(total / avg)

            # Skip players with insufficient weeks played
            if weeks_played < min_weeks:
                continue

            players.append(
                Player(
                    name=row["Player"],
                    position=row["Pos"],
                    team=row["Team"],
                    avg=avg,
                    total=total,
                )
            )

    # Keep only top N by total points
    players.sort(key=lambda p: p.total, reverse=True)
    players = players[:top_n_by_total]

    return players


def compute_initial_adp(
    players: list[Player], baseline_positions: dict[str, int] | None = None
) -> list[tuple[Player, float, int]]:
    """Calculate initial ADP using Value-Based Ranking (VBR).

    Args:
        players: List of Player objects
        baseline_positions: Optional dict mapping position to baseline rank.
                          If None, uses BASELINE_POSITIONS constant.

    Returns:
        List of tuples: (player, vbr, adp) sorted by VBR descending

    VBR Calculation:
        - Group players by position
        - Find baseline player at specified position rank
        - VBR = player_avg - baseline_avg for that position
        - Sort by VBR descending to get ADP order
    """
    if baseline_positions is None:
        baseline_positions = BASELINE_POSITIONS

    # Group players by position
    by_position: dict[str, list[Player]] = defaultdict(list)
    for player in players:
        by_position[player.position].append(player)

    # Sort each position by average points descending
    for pos_players in by_position.values():
        pos_players.sort(key=lambda p: p.avg, reverse=True)

    # Find baseline points for each position
    baseline_points = {}
    for pos, baseline_rank in baseline_positions.items():
        pos_players = by_position[pos]
        if len(pos_players) >= baseline_rank:
            # Use the baseline rank player (1-indexed)
            baseline_points[pos] = pos_players[baseline_rank - 1].avg
        else:
            # If not enough players, use lowest available
            baseline_points[pos] = pos_players[-1].avg if pos_players else 0.0

    # Calculate VBR for each player
    players_with_vbr = []
    for player in players:
        baseline = baseline_points.get(player.position, 0.0)
        vbr = player.avg - baseline
        players_with_vbr.append((player, vbr))

    # Sort by VBR descending (higher VBR = earlier pick)
    players_with_vbr.sort(key=lambda x: x[1], reverse=True)

    # Return list of (player, vbr, adp) tuples
    result = []
    for i, (player, vbr) in enumerate(players_with_vbr):
        result.append((player, vbr, i + 1))

    return result


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

    # logger.info("EXTREME PERTURBATION: Reversing ADP values")

    # # Sort by current ADP to get original order
    # sorted_data = sorted(initial_adp_data, key=lambda x: x[2])

    # # Reverse the ADP assignments: best player gets worst ADP, worst gets best
    # perturbed = []
    # total_players = len(sorted_data)

    # for i, (player, vbr, original_adp) in enumerate(sorted_data):
    #     # Flip the ADP: position 0 gets ADP total_players, position 1 gets total_players-1, etc.
    #     reversed_adp = total_players - i
    #     perturbed.append((player, vbr, reversed_adp))

    # # Sort by new ADP values to maintain proper structure
    # perturbed.sort(key=lambda x: x[2])

    return perturbed


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


def save_final_adp_csv(
    output_file_path: str,
    players: list[Player],
    final_adp: dict[str, float],
    final_draft_state: DraftState | None,
) -> None:
    """Save final ADP results to CSV file with draft details.

    Args:
        output_file_path: Path where to save the CSV file
        players: List of all players
        final_adp: Final ADP values for all players
        final_draft_state: Final draft state (can be None)
        num_teams: Number of teams in the draft
    """
    # Create enhanced ADP list with draft details
    adp_players = []
    for player in players:
        if player.name in final_adp:
            team_id, round_num, pick_num = (
                final_draft_state.get_pick_details(player.name)
                if final_draft_state
                else (0, 0, 0)
            )

            # Create a simple player dict for CSV writing
            player_dict = {
                "name": player.name,
                "position": player.position,
                "team": player.team,
                "avg": player.avg,
                "total": player.total,
                "adp": final_adp[player.name],
                "Team": team_id,
                "Round": round_num,
                "draft_pick": pick_num,
            }
            adp_players.append(player_dict)

    # Sort by ADP (ascending order)
    adp_players.sort(key=lambda x: float(str(x["adp"])))
    if final_draft_state:
        adp_players = adp_players[: final_draft_state.total_picks]

    # Ensure output directory exists
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Write enhanced CSV with draft details
    with open(output_file_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "position",
                "team",
                "avg",
                "total",
                "adp",
                "Team",
                "Round",
                "draft_pick",
            ],
        )
        writer.writeheader()
        writer.writerows(adp_players)


def save_team_scores_csv(
    file_path: Path, final_draft_state: DraftState | None
) -> list[dict[str, float]]:
    """Save team scores to CSV file and return the data.

    Args:
        file_path: Path where to save the team scores CSV
        final_draft_state: Final draft state (can be None)

    Returns:
        List of team score dictionaries
    """
    team_scores = []
    if final_draft_state:
        for i, team in enumerate(final_draft_state.teams):
            team_score = team.calculate_total_score()
            team_dict = {
                "team_id": i + 1,  # 1-indexed
                "total_score": team_score,
                "avg_per_week": team_score,  # Same as total_score since we use avg
            }
            team_scores.append(team_dict)

        # Save to CSV
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["team_id", "total_score", "avg_per_week"]
            )
            writer.writeheader()
            writer.writerows(team_scores)

    return team_scores


def save_regrets_csv(
    file_path: Path, final_regrets: dict[str, float], final_adp: dict[str, float]
) -> None:
    """Save final regret values to CSV file, sorted by ADP.

    Args:
        file_path: Path where to save the regrets CSV
        final_regrets: Dictionary of player regret scores
        final_adp: Dictionary of final ADP values for sorting
    """
    with open(file_path, "w", newline="") as f:
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


def save_convergence_history_csv(
    file_path: Path, convergence_history: list[int]
) -> None:
    """Save convergence history to CSV file.

    Args:
        file_path: Path where to save the convergence history CSV
        convergence_history: List of position changes per iteration
    """
    with open(file_path, "w") as f:
        f.write("iteration,position_changes\n")
        for i, changes in enumerate(convergence_history, 1):
            f.write(f"{i},{changes}\n")


def save_run_parameters_txt(
    file_path: Path,
    run_id: str,
    data_file_path: str,
    learning_rate: float,
    max_iterations: int,
    num_teams: int,
    perturbation_factor: float,
    iterations: int,
    convergence_history: list[int],
) -> None:
    """Save run parameters to text file.

    Args:
        file_path: Path where to save the parameters file
        run_id: Run identifier
        data_file_path: Path to input data file
        learning_rate: Learning rate used
        max_iterations: Maximum iterations allowed
        num_teams: Number of teams in draft
        perturbation_factor: Perturbation factor used
        iterations: Actual iterations completed
        convergence_history: History of position changes per iteration
    """
    with open(file_path, "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Data file: {data_file_path}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Number of teams: {num_teams}\n")
        f.write(f"Perturbation factor: {perturbation_factor}\n")
        f.write(f"Final iterations: {iterations}\n")
        f.write(
            f"Final position changes: {convergence_history[-1] if convergence_history else 'N/A'}\n"
        )


def save_initial_vbr_adp_csv(
    file_path: Path, initial_adp_data: list[tuple[Player, float, int]]
) -> None:
    """Save initial VBR-based ADP values to CSV file.

    Args:
        file_path: Path where to save the initial ADP CSV
        initial_adp_data: List of (player, vbr, adp) tuples
    """
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
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "position", "team", "avg", "total", "vbr", "adp"]
        )
        writer.writeheader()
        writer.writerows(initial_adp_players)
