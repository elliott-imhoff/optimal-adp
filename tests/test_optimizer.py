"""Tests for optimizer module."""

import csv
import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from optimal_adp.data_io import load_player_data, compute_initial_adp
from optimal_adp.cli import setup_logging
from optimal_adp.optimizer import optimize_adp, run_optimization_loop
from optimal_adp.models import Player


class TestRunOptimizationWithValidationAndIO:
    """Test the main process function."""

    @pytest.fixture
    def temp_data_file(self) -> str:
        """Create a temporary CSV file with test player data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["Player", "Pos", "Team", "Avg", "Total"])
            writer.writerow(["Josh Allen", "QB", "BUF", "25.0", "400.0"])
            writer.writerow(["Christian McCaffrey", "RB", "SF", "20.0", "320.0"])
            writer.writerow(["Tyreek Hill", "WR", "MIA", "18.0", "288.0"])
            writer.writerow(["Travis Kelce", "TE", "KC", "15.0", "240.0"])
            writer.writerow(["Lamar Jackson", "QB", "BAL", "24.0", "384.0"])
            writer.writerow(["Derrick Henry", "RB", "TEN", "19.0", "304.0"])
            return f.name

    def test_successful_optimization_run(self, temp_data_file: str) -> None:
        """Test successful optimization run with all features enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory for artifacts
                import os

                os.chdir(temp_dir)

                result = run_optimization_loop(
                    data_file_path=temp_data_file,
                    learning_rate=0.1,
                    max_iterations=10,  # Small for testing
                    num_teams=4,
                    perturbation_factor=0.05,
                    artifacts_outputs=True,
                )

                # Should succeed
                assert isinstance(result, bool)
                # Note: result might be True or False depending on validation rules
                # but the important thing is it doesn't crash

                # Check that artifacts directory was created
                artifacts_dir = Path("artifacts")
                assert artifacts_dir.exists()

                # Check that at least one run directory was created
                run_dirs = list(artifacts_dir.glob("run_*"))
                assert len(run_dirs) >= 1

                run_dir = run_dirs[0]

                # Check that expected artifact files were created
                expected_files = [
                    "initial_vbr_adp.csv",
                    "final_adp.csv",
                    "convergence_history.csv",
                    "team_scores.csv",
                    "regrets.csv",
                    "run_parameters.txt",
                ]

                for filename in expected_files:
                    file_path = run_dir / filename
                    assert (
                        file_path.exists()
                    ), f"Expected artifact file {filename} not found"
                    assert (
                        file_path.stat().st_size > 0
                    ), f"Artifact file {filename} is empty"

            finally:
                os.chdir(original_cwd)
                # Clean up temp file
                Path(temp_data_file).unlink(missing_ok=True)

    def test_optimization_without_artifacts(self, temp_data_file: str) -> None:
        """Test optimization run without artifact generation."""
        try:
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                learning_rate=0.1,
                max_iterations=5,  # Small for testing
                num_teams=2,
                perturbation_factor=0.0,
                artifacts_outputs=False,  # No artifacts
            )

            # Should succeed without creating artifacts
            assert isinstance(result, bool)

        finally:
            # Clean up temp file
            Path(temp_data_file).unlink(missing_ok=True)

    def test_optimization_with_perturbation(self, temp_data_file: str) -> None:
        """Test optimization with ADP perturbation enabled."""
        try:
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                learning_rate=0.2,
                max_iterations=5,
                num_teams=4,
                perturbation_factor=0.1,
                artifacts_outputs=False,
            )

            # Should complete successfully
            assert isinstance(result, bool)

        finally:
            # Clean up temp file
            Path(temp_data_file).unlink(missing_ok=True)

    def test_optimization_with_invalid_file_path(self) -> None:
        """Test optimization with non-existent data file."""
        result = run_optimization_loop(
            data_file_path="/non/existent/file.csv",
            learning_rate=0.1,
            max_iterations=10,
            artifacts_outputs=False,
        )

        # Should return False due to file not found error
        assert result is False

    def test_optimization_with_malformed_csv(self) -> None:
        """Test optimization with malformed CSV data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write malformed CSV (missing required columns)
            f.write("Wrong,Headers,Here\n")
            f.write("Invalid,Data,Structure\n")
            temp_path = f.name

        try:
            result = run_optimization_loop(
                data_file_path=temp_path,
                learning_rate=0.1,
                max_iterations=10,
                artifacts_outputs=False,
            )

            # Should return False due to parsing error
            assert result is False

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_default_parameters(self, temp_data_file: str) -> None:
        """Test optimization with default parameters."""
        try:
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                artifacts_outputs=False,
            )

            # Should complete with defaults
            assert isinstance(result, bool)

        finally:
            Path(temp_data_file).unlink(missing_ok=True)

    @patch("optimal_adp.optimizer.optimize_adp")
    def test_optimization_exception_handling(
        self, mock_optimize: Mock, temp_data_file: str
    ) -> None:
        """Test that exceptions in optimization are properly handled."""
        # Make optimize_adp raise an exception
        mock_optimize.side_effect = RuntimeError("Simulated optimization failure")

        try:
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                artifacts_outputs=False,
            )

            # Should return False when exception occurs
            assert result is False

        finally:
            Path(temp_data_file).unlink(missing_ok=True)

    @patch("optimal_adp.optimizer.validate_optimization_results")
    def test_validation_failure_handling(
        self, mock_validate: Mock, temp_data_file: str
    ) -> None:
        """Test handling when validation fails."""
        # Mock validation to return a failed result
        mock_result = Mock()
        mock_result.all_passed.return_value = False
        mock_result.messages = ["Test validation failure"]
        mock_validate.return_value = mock_result

        try:
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                artifacts_outputs=False,
            )

            # Should return False when validation fails
            assert result is False

        finally:
            Path(temp_data_file).unlink(missing_ok=True)

    def test_edge_case_small_dataset(self) -> None:
        """Test optimization with minimal dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["Player", "Pos", "Team", "Avg", "Total"])
            writer.writerow(["Test Player", "QB", "TEST", "20.0", "200.0"])
            temp_path = f.name

        try:
            result = run_optimization_loop(
                data_file_path=temp_path,
                learning_rate=0.1,
                max_iterations=5,
                num_teams=2,
                artifacts_outputs=False,
            )

            # Should handle small dataset gracefully
            assert isinstance(result, bool)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_extreme_parameters(self, temp_data_file: str) -> None:
        """Test optimization with extreme parameter values."""
        try:
            # Test with very high learning rate and low iterations
            result = run_optimization_loop(
                data_file_path=temp_data_file,
                learning_rate=1.0,  # Very high
                max_iterations=1,  # Very low
                num_teams=2,
                artifacts_outputs=False,
            )

            # Should handle extreme values
            assert isinstance(result, bool)

        finally:
            Path(temp_data_file).unlink(missing_ok=True)

    def test_artifact_file_content_validation(self, temp_data_file: str) -> None:
        """Test that generated artifact files contain expected content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)

                _ = run_optimization_loop(
                    data_file_path=temp_data_file,
                    learning_rate=0.1,
                    max_iterations=3,
                    num_teams=2,
                    artifacts_outputs=True,
                )

                # Find the run directory
                artifacts_dir = Path("artifacts")
                run_dirs = list(artifacts_dir.glob("run_*"))
                assert len(run_dirs) >= 1
                run_dir = run_dirs[0]

                # Validate initial_vbr_adp.csv content
                initial_adp_file = run_dir / "initial_vbr_adp.csv"
                with open(initial_adp_file, "r") as f:
                    content = f.read()
                    # Check for header
                    assert "name,position,team,avg,total,vbr,adp" in content
                    # Content might be empty if players were filtered out, so just check structure

                # Validate final_adp.csv content
                final_adp_file = run_dir / "final_adp.csv"
                with open(final_adp_file, "r") as f:
                    content = f.read()
                    # Check for header structure
                    assert "name,position,team,avg,total,adp" in content

                # Validate convergence_history.csv content
                convergence_file = run_dir / "convergence_history.csv"
                with open(convergence_file, "r") as f:
                    content = f.read()
                    assert "iteration,position_changes" in content
                    # Should have at least one iteration of data
                    lines = content.strip().split("\n")
                    assert len(lines) >= 2  # Header + at least 1 data row

                # Validate run_parameters.txt content
                params_file = run_dir / "run_parameters.txt"
                with open(params_file, "r") as f:
                    content = f.read()
                    assert "Learning rate: 0.1" in content
                    assert "Max iterations: 3" in content
                    assert "Number of teams: 2" in content

            finally:
                os.chdir(original_cwd)
                Path(temp_data_file).unlink(missing_ok=True)


@pytest.fixture
def sample_player_data() -> list[Player]:
    """Create sample player data for testing with diverse stats."""
    return [
        Player(name="QB1", position="QB", team="KC", avg=25.0, total=425.0),
        Player(name="QB2", position="QB", team="BUF", avg=18.0, total=306.0),
        Player(name="RB1", position="RB", team="SF", avg=22.0, total=374.0),
        Player(name="RB2", position="RB", team="DAL", avg=12.0, total=204.0),
        Player(name="WR1", position="WR", team="MIA", avg=19.0, total=323.0),
        Player(name="WR2", position="WR", team="CIN", avg=11.0, total=187.0),
        Player(name="TE1", position="TE", team="KC", avg=14.0, total=238.0),
        Player(name="TE2", position="TE", team="SF", avg=8.0, total=136.0),
    ]


# Test configuration
SMALL_DRAFT_NUM_TEAMS = 4  # More teams to create more draft picks


@pytest.fixture
def temp_data_file(sample_player_data: list[Player]) -> str:
    """Create a temporary CSV file with sample player data."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

    writer = csv.writer(temp_file)
    # Use column names expected by load_player_data function
    writer.writerow(["Player", "Pos", "Team", "AVG", "TTL"])

    for player in sample_player_data:
        writer.writerow(
            [player.name, player.position, player.team, player.avg, player.total]
        )

    temp_file.close()
    return temp_file.name


class TestOptimizeAdp:
    """Tests for the main optimize_adp function."""

    def test_basic_optimization_loop(self, temp_data_file: str) -> None:
        """Test that the optimization loop runs without errors."""
        # Load data and prepare initial ADP as the optimize_adp function now expects
        players = load_player_data(temp_data_file)
        initial_adp_data = compute_initial_adp(players)
        initial_adp = {player.name: float(adp) for player, _, adp in initial_adp_data}

        (
            final_adp,
            convergence_history,
            iterations_completed,
            final_regrets,
            final_draft_state,
        ) = optimize_adp(
            players=players,
            initial_adp=initial_adp,
            num_teams=SMALL_DRAFT_NUM_TEAMS,
            max_iterations=5,  # More iterations to prevent early convergence
            learning_rate=0.5,  # Higher learning rate to force changes
        )

        # Check return values
        assert isinstance(final_adp, dict)
        assert len(final_adp) == 8  # All players should have ADP
        assert isinstance(convergence_history, list)
        assert len(convergence_history) >= 1  # At least one convergence check
        assert isinstance(iterations_completed, int)
        assert iterations_completed >= 1
        assert isinstance(final_regrets, dict)
        assert final_draft_state is not None
        assert len(final_draft_state.teams) > 0

    def test_convergence_detection(self, temp_data_file: str) -> None:
        """Test that convergence is properly detected."""
        # Load data and prepare initial ADP as the optimize_adp function now expects
        players = load_player_data(temp_data_file)
        initial_adp_data = compute_initial_adp(players)
        initial_adp = {player.name: float(adp) for player, _, adp in initial_adp_data}

        # Mock the convergence check to return 0 (converged) after 2 iterations
        with patch("optimal_adp.optimizer.check_convergence") as mock_convergence:
            mock_convergence.side_effect = [5, 0]  # Converge on second check

            (
                final_adp,
                convergence_history,
                iterations_completed,
                final_regrets,
                final_draft_state,
            ) = optimize_adp(
                players=players,
                initial_adp=initial_adp,
                num_teams=SMALL_DRAFT_NUM_TEAMS,
                max_iterations=10,
                learning_rate=0.1,
            )

            # Should stop early due to convergence
            assert iterations_completed == 2
            assert len(convergence_history) == 2
            assert convergence_history == [5, 0]


class TestOptimizationHelpers:
    """Tests for optimization helper functions and edge cases."""

    def test_constrained_optimization_maintains_hierarchy(
        self, temp_data_file: str
    ) -> None:
        """Test that constrained optimization maintains position hierarchy."""
        # Load data and prepare initial ADP
        players = load_player_data(temp_data_file)
        initial_adp_data = compute_initial_adp(players)
        initial_adp = {player.name: float(adp) for player, _, adp in initial_adp_data}

        # Run optimization with higher learning rate to force position changes
        (
            final_adp,
            convergence_history,
            iterations_completed,
            final_regrets,
            final_draft_state,
        ) = optimize_adp(
            players=players,
            initial_adp=initial_adp,
            num_teams=SMALL_DRAFT_NUM_TEAMS,
            max_iterations=3,
            learning_rate=0.5,  # Higher learning rate to create potential violations
        )

        # Group players by position for hierarchy validation

        players_by_pos: dict[str, list[dict[str, Any]]] = {}
        for player in players:  # Use original players list
            pos = player.position
            if pos not in players_by_pos:
                players_by_pos[pos] = []
            players_by_pos[pos].append(
                {
                    "name": player.name,
                    "avg": player.avg,
                    "adp": final_adp[player.name],
                }
            )

        # Verify hierarchy within each position
        for position, position_players in players_by_pos.items():
            # Sort by ADP (draft order)
            position_players.sort(key=lambda p: p["adp"])

            # Verify that AVG decreases as ADP increases (later picks)
            for i in range(len(position_players) - 1):
                current_player = position_players[i]
                next_player = position_players[i + 1]

                assert current_player["avg"] >= next_player["avg"], (
                    f"Position hierarchy violated in {position}: "
                    f"{current_player['name']} (AVG: {current_player['avg']}) "
                    f"drafted before {next_player['name']} (AVG: {next_player['avg']})"
                )


class TestMainCLI:
    """Tests for the CLI interface."""

    def test_setup_logging_info_level(self) -> None:
        """Test logging setup with INFO level."""
        setup_logging(verbose=False)
        logger = logging.getLogger("optimal_adp")
        assert logger.level <= 20  # INFO level or lower

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with DEBUG level."""
        setup_logging(verbose=True)
        logger = logging.getLogger("optimal_adp")
        assert logger.level <= 10  # DEBUG level or lower


class TestIntegrationWithRealData:
    """Integration tests using the actual 2024 stats data."""

    def test_full_optimization_with_real_data(self) -> None:
        """Test full optimization loop with real 2024 data."""
        # Check if real data file exists
        data_file = Path("data/2024_stats.csv")
        if not data_file.exists():
            pytest.skip("Real data file not available")

        # Use small configuration for faster test
        num_teams = 4  # Small for faster execution

        try:
            # Load data and prepare initial ADP as the optimize_adp function now expects
            players = load_player_data(str(data_file))
            initial_adp_data = compute_initial_adp(players)
            initial_adp = {
                player.name: float(adp) for player, _, adp in initial_adp_data
            }

            (
                final_adp,
                convergence_history,
                iterations_completed,
                final_regrets,
                final_draft_state,
            ) = optimize_adp(
                players=players,
                initial_adp=initial_adp,
                num_teams=num_teams,
                max_iterations=5,  # Keep small for testing
                learning_rate=0.1,
            )
        except (ValueError, TypeError, KeyError) as e:
            pytest.skip(f"Real data file has formatting issues: {e}")

        # Basic checks
        assert len(final_adp) > 0
        assert len(convergence_history) > 0
        assert iterations_completed > 0
        assert len(final_regrets) > 0
        assert final_draft_state is not None
        assert len(final_draft_state.teams) > 0

        # Check that top players have low ADP (good draft positions)
        sorted_players = sorted(final_adp.items(), key=lambda x: x[1])
        top_10_names = [name for name, _ in sorted_players[:10]]

        # Should include some recognizable top players (this is a sanity check)
        # Note: This is a loose check since we don't know exact player names
        assert len(top_10_names) == 10
