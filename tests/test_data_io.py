"""Tests for data I/O functions."""

import csv
import tempfile
from pathlib import Path

import pytest

from optimal_adp.config import Player
from optimal_adp.data_io import (
    compute_initial_adp,
    load_player_data,
    save_adp_results,
    save_regret_results,
)


def test_load_player_data_from_fixture() -> None:
    """Test loading player data from our test fixture CSV."""
    # Use our existing fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "sample_stats.csv"
    assert fixture_path.exists(), f"Fixture file not found: {fixture_path}"

    # Load using our actual function
    players = load_player_data(
        str(fixture_path), min_weeks=5
    )  # Lower threshold for test data

    # Should have loaded our fixture players
    assert len(players) == 10  # Our fixture has 10 players

    # Check that we got Player objects
    assert all(isinstance(p, Player) for p in players)

    # Check first player by total points (should be Lamar Jackson with 434.4)
    top_player = max(players, key=lambda p: p.total)
    assert top_player.name == "Lamar Jackson"
    assert top_player.position == "QB"
    assert top_player.avg == 25.6
    assert top_player.total == 434.4


def test_load_real_data_file() -> None:
    """Test loading the actual 2024 stats CSV file."""
    from pathlib import Path

    data_file = Path("data/2024_stats.csv")
    if not data_file.exists():
        pytest.skip("Real data file not available")

    players = load_player_data(str(data_file))

    # Should load players successfully
    assert len(players) > 0
    assert len(players) <= 150  # Respects top_n_by_total limit

    # Check that we got Player objects
    assert all(isinstance(p, Player) for p in players)

    # Check that no K or DEF positions were loaded
    positions = {p.position for p in players}
    assert "K" not in positions
    assert "DEF" not in positions

    # Check that all players have valid numeric stats
    for player in players:
        assert isinstance(player.avg, float)
        assert isinstance(player.total, float)
        assert player.avg >= 0
        assert player.total >= 0


def test_compute_initial_adp() -> None:
    """Test VBR calculation for initial ADP."""
    # Create test players using our Player dataclass
    players = [
        Player("Josh Allen", "QB", "BUF", 22.6, 385.0),
        Player("Lamar Jackson", "QB", "MIA", 25.6, 434.4),
        Player("Saquon Barkley", "RB", "PHI", 21.2, 338.8),
        Player("Jahmyr Gibbs", "RB", "DET", 19.8, 336.9),
        Player("Ja'Marr Chase", "WR", "CIN", 20.0, 339.5),
        Player("CeeDee Lamb", "WR", "DAL", 16.2, 275.5),
        Player("Travis Kelce", "TE", "KC", 12.2, 207.6),
        Player("George Kittle", "TE", "SF", 12.2, 195.4),
    ]

    # Use custom baseline positions suitable for our small test dataset
    # This will use the 2nd player at each position as baseline
    test_baselines = {
        "QB": 2,  # Josh Allen (22.6) as QB baseline
        "RB": 2,  # Jahmyr Gibbs (19.8) as RB baseline
        "WR": 2,  # CeeDee Lamb (16.2) as WR baseline
        "TE": 2,  # George Kittle (12.2) as TE baseline
    }

    # Use our actual compute_initial_adp function with custom baselines
    adp_data = compute_initial_adp(players, test_baselines)

    # Should return list of (player, vbr, adp) tuples
    assert isinstance(adp_data, list)
    assert len(adp_data) == len(players)

    # Check structure of results
    for entry in adp_data:
        assert len(entry) == 3
        player, vbr, adp = entry
        assert isinstance(player, Player)
        assert isinstance(vbr, (int, float))
        assert isinstance(adp, int)

    # Results should be sorted by VBR descending (best players first)
    vbr_values = [entry[1] for entry in adp_data]
    assert vbr_values == sorted(vbr_values, reverse=True)

    # ADP should be sequential starting from 1
    adp_values = [entry[2] for entry in adp_data]
    assert adp_values == list(range(1, len(players) + 1))

    # Test specific VBR calculations with our custom baselines:
    # QB baseline: Josh Allen (22.6), RB: Jahmyr Gibbs (19.8),
    # WR: CeeDee Lamb (16.2), TE: George Kittle (12.2)

    # Expected VBR values:
    # Ja'Marr Chase: 20.0 - 16.2 = 3.8 (should be #1)
    # Lamar Jackson: 25.6 - 22.6 = 3.0 (should be #2)
    # Saquon Barkley: 21.2 - 19.8 = 1.4 (should be #3)
    # Josh Allen: 22.6 - 22.6 = 0.0 (baseline QB)
    # Jahmyr Gibbs: 19.8 - 19.8 = 0.0 (baseline RB)
    # CeeDee Lamb: 16.2 - 16.2 = 0.0 (baseline WR)
    # Travis Kelce: 12.2 - 12.2 = 0.0 (both TEs same avg)
    # George Kittle: 12.2 - 12.2 = 0.0 (baseline TE)

    # Find specific players in results
    player_vbr_map = {entry[0].name: entry[1] for entry in adp_data}

    assert player_vbr_map["Ja'Marr Chase"] == pytest.approx(3.8)
    assert player_vbr_map["Lamar Jackson"] == pytest.approx(3.0)
    assert player_vbr_map["Saquon Barkley"] == pytest.approx(1.4)
    assert player_vbr_map["Josh Allen"] == pytest.approx(0.0)
    assert player_vbr_map["Jahmyr Gibbs"] == pytest.approx(0.0)
    assert player_vbr_map["CeeDee Lamb"] == pytest.approx(0.0)
    assert player_vbr_map["Travis Kelce"] == pytest.approx(0.0)
    assert player_vbr_map["George Kittle"] == pytest.approx(0.0)

    # Verify ADP order matches VBR ranking
    assert adp_data[0][0].name == "Ja'Marr Chase"  # Highest VBR
    assert adp_data[1][0].name == "Lamar Jackson"  # Second highest VBR
    assert adp_data[2][0].name == "Saquon Barkley"  # Third highest VBR


def test_save_adp_results() -> None:
    """Test saving ADP results to CSV."""
    # Create sample ADP data using our data structures
    adp_data = [
        (Player("Lamar Jackson", "QB", "BAL", 25.6, 434.4), 3.6, 1),
        (Player("Ja'Marr Chase", "WR", "CIN", 20.0, 339.5), 5.0, 2),
        (Player("Saquon Barkley", "RB", "PHI", 21.2, 338.8), 2.2, 3),
    ]

    # Test saving to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name

    # Use our actual save function
    save_adp_results(adp_data, temp_path)

    # Verify file was written correctly
    with open(temp_path, "r") as f:
        reader = csv.DictReader(f)
        saved_data = list(reader)

    assert len(saved_data) == 3
    assert saved_data[0]["player"] == "Lamar Jackson"
    assert saved_data[0]["position"] == "QB"
    assert float(saved_data[0]["adp"]) == 1
    assert float(saved_data[0]["vbr"]) == pytest.approx(3.6)

    # Clean up
    Path(temp_path).unlink()


def test_save_regret_results() -> None:
    """Test saving regret results to CSV."""
    regret_data = [
        {"pick_number": 1, "player": "Lamar Jackson", "regret": 0.0},
        {"pick_number": 2, "player": "Ja'Marr Chase", "regret": -1.5},
        {"pick_number": 3, "player": "Saquon Barkley", "regret": 2.1},
    ]

    # Test saving to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name

    # Use our actual save function
    save_regret_results(regret_data, temp_path)

    # Verify file contents
    with open(temp_path, "r") as f:
        reader = csv.DictReader(f)
        saved_data = list(reader)

    assert len(saved_data) == 3
    assert int(saved_data[0]["pick_number"]) == 1
    assert saved_data[0]["player"] == "Lamar Jackson"
    assert float(saved_data[0]["regret"]) == pytest.approx(0.0)
    assert float(saved_data[2]["regret"]) == pytest.approx(2.1)

    # Clean up
    Path(temp_path).unlink()


def test_player_filtering() -> None:
    """Test filtering logic for players."""
    # Mock data with players that should be filtered out
    all_players = [
        {"name": "Josh Allen", "position": "QB", "total": 385.0, "games": 16},
        {"name": "Some Kicker", "position": "K", "total": 120.0, "games": 16},
        {"name": "Some Defense", "position": "DEF", "total": 150.0, "games": 16},
        {"name": "Low Total Player", "position": "RB", "total": 25.0, "games": 16},
        {"name": "Injured Player", "position": "WR", "total": 200.0, "games": 8},
        {"name": "Valid Player", "position": "RB", "total": 300.0, "games": 15},
    ]

    # Apply filtering logic
    filtered = []
    for player in all_players:
        # Filter out K and DEF
        if player["position"] in ["K", "DEF"]:
            continue
        # Filter out low total (< 50 for this test)
        if float(str(player["total"] or 0)) < 50.0:  # Ensure type compatibility
            continue
        # Filter out < 10 games
        if int(str(player["games"] or 0)) < 10:  # Ensure type compatibility
            continue

        filtered.append(player)

    # Should only have Josh Allen and Valid Player
    assert len(filtered) == 2
    names = [p["name"] for p in filtered]
    assert "Josh Allen" in names
    assert "Valid Player" in names
    assert "Some Kicker" not in names
    assert "Injured Player" not in names
