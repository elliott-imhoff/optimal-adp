"""Data loading and saving functions for optimal ADP calculation."""

import csv
from collections import defaultdict
from typing import Any

from .config import Player

# Baseline positions for VBR calculation (position: rank)
# These represent the "replacement level" player at each position
BASELINE_POSITIONS = {
    "QB": 12,  # QB12 baseline
    "RB": 24,  # RB24 baseline
    "WR": 36,  # WR36 baseline
    "TE": 12,  # TE12 baseline
}


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
        - Exclude K and DEF positions
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

            # Skip K and DEF positions
            if row["Pos"] in ["K", "DEF"]:
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


def save_adp_results(
    adp_data: list[tuple[Player, float, int]], output_path: str
) -> None:
    """Save ADP results to CSV file.

    Args:
        adp_data: List of (player, vbr, adp) tuples
        output_path: Path to output CSV file
    """
    if not adp_data:
        return

    fieldnames = ["player", "position", "team", "avg", "total", "vbr", "adp"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for player, vbr, adp in adp_data:
            writer.writerow(
                {
                    "player": player.name,
                    "position": player.position,
                    "team": player.team,
                    "avg": player.avg,
                    "total": player.total,
                    "vbr": vbr,
                    "adp": adp,
                }
            )


def save_regret_results(regret_data: list[dict[str, Any]], output_path: str) -> None:
    """Save regret results to CSV file.

    Args:
        regret_data: List of dictionaries with regret information
        output_path: Path to output CSV file
    """
    if not regret_data:
        return

    fieldnames = regret_data[0].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(regret_data)
