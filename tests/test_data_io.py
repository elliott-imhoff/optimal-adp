"""Tests for data I/O functions."""

import csv
import tempfile
from pathlib import Path
import random

import pytest

from optimal_adp.data_io import (
    compute_initial_adp,
    create_run_directory,
    get_pick_details,
    load_player_data,
    save_adp_results,
    save_convergence_history_csv,
    save_final_adp_csv,
    save_initial_vbr_adp_csv,
    save_regret_results,
    save_regrets_csv,
    save_run_parameters_txt,
    save_team_scores_csv,
    perturb_initial_adp,
)
from optimal_adp.models import DraftState, Player


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


class TestPerturbInitialAdp:
    """Tests for perturb_initial_adp function."""

    def test_perturb_maintains_structure(self) -> None:
        """Test perturbation maintains the data structure."""
        # Create test data
        players = [
            Player("Player1", "QB", "TEAM1", 25.0, 400.0),
            Player("Player2", "RB", "TEAM2", 20.0, 300.0),
            Player("Player3", "WR", "TEAM3", 15.0, 200.0),
        ]
        initial_data = [(players[i], float(i * 5), i + 1) for i in range(3)]

        # Perturb
        perturbed = perturb_initial_adp(initial_data, perturbation_factor=0.1)

        # Check structure maintained
        assert len(perturbed) == len(initial_data)
        for player, vbr, adp in perturbed:
            assert isinstance(player, Player)
            assert isinstance(vbr, float)
            assert isinstance(adp, int)
            assert adp >= 1.0  # ADP should stay positive

    def test_perturb_changes_values(self) -> None:
        """Test perturbation actually changes ADP values."""
        # Create test data with fixed random seed for reproducibility

        random.seed(42)

        players = [Player("Player1", "QB", "TEAM1", 25.0, 400.0)]
        # Use larger ADP value so perturbation is more likely to change the rounded result
        initial_data = [(players[0], 10.0, 10)]

        # Perturb
        perturbed = perturb_initial_adp(initial_data, perturbation_factor=0.2)

        # Should be different (with high probability for larger values)
        original_adp = initial_data[0][2]
        perturbed_adp = perturbed[0][2]
        # Note: This might occasionally fail due to random chance, but very unlikely
        assert perturbed_adp != original_adp

    def test_perturb_zero_factor_no_change(self) -> None:
        """Test zero perturbation factor produces no change."""
        players = [Player("Player1", "QB", "TEAM1", 25.0, 400.0)]
        initial_data = [(players[0], 10.0, 5)]

        perturbed = perturb_initial_adp(initial_data, perturbation_factor=0.0)

        # Should be identical
        assert perturbed[0][2] == initial_data[0][2]


class TestArtifactsFunctions:
    """Test functions moved from artifacts.py module."""

    def test_create_run_directory(self) -> None:
        """Test creating a run directory with timestamp and parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory for this test
                import os

                os.chdir(temp_dir)

                learning_rate = 0.1
                max_iterations = 1000

                run_id, run_path = create_run_directory(learning_rate, max_iterations)

                # Check run_id format
                assert f"lr{learning_rate}" in run_id
                assert f"iter{max_iterations}" in run_id
                assert len(run_id.split("_")) >= 3  # timestamp_lr{lr}_iter{iter}

                # Check directory was created
                assert run_path.exists()
                assert run_path.is_dir()
                assert "artifacts" in str(run_path)
                assert f"run_{run_id}" in str(run_path)

            finally:
                os.chdir(original_cwd)

    def test_get_pick_details_drafted_player(self) -> None:
        """Test getting pick details for a drafted player."""
        # Create test players
        player1 = Player("Player1", "QB", "TEAM1", 25.0, 400.0)
        player2 = Player("Player2", "RB", "TEAM2", 20.0, 320.0)
        players = [player1, player2]

        # Create a simple draft state with 2 teams
        num_teams = 2
        draft_state = DraftState(players, {"Player1": 1.0, "Player2": 2.0})

        # Simulate the draft to populate draft_history
        draft_state.draft_history = [
            (0, player1),  # Team 0 picks Player1 first
            (1, player2),  # Team 1 picks Player2 second
        ]
        draft_state.pick_order = [0, 1]  # Team order for picks

        # Test Player1 (first pick)
        team_id, round_num, pick_num = get_pick_details(
            "Player1", draft_state, num_teams
        )
        assert team_id == 1  # Team 0 becomes team 1 (1-indexed)
        assert round_num == 1  # First round
        assert pick_num == 1  # First pick

        # Test Player2 (second pick)
        team_id, round_num, pick_num = get_pick_details(
            "Player2", draft_state, num_teams
        )
        assert team_id == 2  # Team 1 becomes team 2 (1-indexed)
        assert round_num == 1  # First round
        assert pick_num == 2  # Second pick

    def test_get_pick_details_undrafted_player(self) -> None:
        """Test getting pick details for an undrafted player."""
        player1 = Player("Player1", "QB", "TEAM1", 25.0, 400.0)
        players = [player1]

        draft_state = DraftState(players, {"Player1": 1.0})
        draft_state.draft_history = []  # Empty draft history

        # Test undrafted player
        team_id, round_num, pick_num = get_pick_details("Player1", draft_state, 2)
        assert team_id == 0
        assert round_num == 0
        assert pick_num == 0

    def test_save_final_adp_csv(self) -> None:
        """Test saving final ADP results to CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            # Create test data
            players = [
                Player("Player1", "QB", "TEAM1", 25.0, 400.0),
                Player("Player2", "RB", "TEAM2", 20.0, 320.0),
            ]
            final_adp = {"Player1": 1.5, "Player2": 2.3}

            # Save without draft state
            save_final_adp_csv(temp_path, players, final_adp, None, 2)

            # Read and verify
            with open(temp_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

            # Check first row (should be sorted by ADP)
            assert rows[0]["name"] == "Player1"
            assert rows[0]["position"] == "QB"
            assert float(rows[0]["adp"]) == 1.5
            assert int(rows[0]["Team"]) == 0  # No draft state
            assert int(rows[0]["Round"]) == 0
            assert int(rows[0]["draft_pick"]) == 0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_team_scores_csv_with_draft_state(self) -> None:
        """Test saving team scores to CSV with draft state."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Create test data with draft state
            players = [
                Player("Player1", "QB", "TEAM1", 25.0, 400.0),
                Player("Player2", "RB", "TEAM2", 20.0, 320.0),
            ]

            draft_state = DraftState(players, {"Player1": 1.0, "Player2": 2.0})

            # Save team scores
            team_scores = save_team_scores_csv(temp_path, draft_state)

            # Verify return value
            assert isinstance(team_scores, list)
            assert len(team_scores) > 0

            # Verify file was created
            assert temp_path.exists()

            # Read and verify CSV content
            with open(temp_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == len(team_scores)
            for i, row in enumerate(rows):
                assert int(row["team_id"]) == team_scores[i]["team_id"]
                assert float(row["total_score"]) == team_scores[i]["total_score"]

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_team_scores_csv_no_draft_state(self) -> None:
        """Test saving team scores to CSV without draft state."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Save with None draft state
            team_scores = save_team_scores_csv(temp_path, None)

            # Should return empty list
            assert team_scores == []

            # File should not be created or should be empty
            if temp_path.exists():
                assert temp_path.stat().st_size == 0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_regrets_csv(self) -> None:
        """Test saving regrets to CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            final_regrets = {"Player1": -0.5, "Player2": 1.2}
            final_adp = {"Player1": 1.0, "Player2": 2.0}

            save_regrets_csv(temp_path, final_regrets, final_adp)

            # Read and verify
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 3  # Header + 2 data rows
            assert rows[0] == ["player_name", "regret_score", "final_adp"]

            # Should be sorted by ADP
            assert rows[1][0] == "Player1"  # Lower ADP first
            assert float(rows[1][1]) == -0.5
            assert float(rows[1][2]) == 1.0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_convergence_history_csv(self) -> None:
        """Test saving convergence history to CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            convergence_history = [10, 5, 2, 1, 0]

            save_convergence_history_csv(temp_path, convergence_history)

            # Read and verify
            with open(temp_path, "r") as f:
                content = f.read().strip()

            lines = content.split("\n")
            assert len(lines) == 6  # Header + 5 data rows
            assert lines[0] == "iteration,position_changes"
            assert lines[1] == "1,10"
            assert lines[-1] == "5,0"

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_run_parameters_txt(self) -> None:
        """Test saving run parameters to text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_run_parameters_txt(
                temp_path,
                run_id="test_run_123",
                data_file_path="data/test.csv",
                learning_rate=0.1,
                max_iterations=1000,
                num_teams=12,
                perturbation_factor=0.05,
                iterations=500,
                convergence_history=[10, 5, 2, 1, 0],
            )

            # Read and verify
            with open(temp_path, "r") as f:
                content = f.read()

            assert "Run ID: test_run_123" in content
            assert "Data file: data/test.csv" in content
            assert "Learning rate: 0.1" in content
            assert "Max iterations: 1000" in content
            assert "Number of teams: 12" in content
            assert "Perturbation factor: 0.05" in content
            assert "Final iterations: 500" in content
            assert "Final position changes: 0" in content

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_initial_vbr_adp_csv(self) -> None:
        """Test saving initial VBR ADP to CSV."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Create test data
            player1 = Player("Player1", "QB", "TEAM1", 25.0, 400.0)
            player2 = Player("Player2", "RB", "TEAM2", 20.0, 320.0)

            initial_adp_data = [
                (player1, 5.2, 1),
                (player2, 3.8, 2),
            ]

            save_initial_vbr_adp_csv(temp_path, initial_adp_data)

            # Read and verify
            with open(temp_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

            # Check first row
            assert rows[0]["name"] == "Player1"
            assert rows[0]["position"] == "QB"
            assert rows[0]["team"] == "TEAM1"
            assert float(rows[0]["avg"]) == 25.0
            assert float(rows[0]["total"]) == 400.0
            assert float(rows[0]["vbr"]) == 5.2
            assert float(rows[0]["adp"]) == 1.0

        finally:
            temp_path.unlink(missing_ok=True)
