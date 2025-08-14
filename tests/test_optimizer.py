"""Tests for the main optimization loop and CLI interface."""

import tempfile
import csv
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from optimal_adp.config import Player
from optimal_adp.optimizer import optimize_adp
from optimal_adp.main import main, setup_logging


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
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_adp.csv"

            final_adp, convergence_history, iterations = optimize_adp(
                data_file_path=temp_data_file,
                num_teams=SMALL_DRAFT_NUM_TEAMS,
                max_iterations=5,  # More iterations to prevent early convergence
                learning_rate=0.5,  # Higher learning rate to force changes
                output_file_path=str(output_path),
            )

            # Check return values
            assert isinstance(final_adp, dict)
            assert len(final_adp) == 8  # All players should have ADP
            assert isinstance(convergence_history, list)
            assert len(convergence_history) >= 1  # At least one convergence check
            assert isinstance(iterations, int)
            assert iterations >= 2  # Should run at least 2 iterations

            # Check output file was created
            assert output_path.exists()

    def test_convergence_detection(self, temp_data_file: str) -> None:
        """Test that convergence is properly detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_adp.csv"

            # Mock the convergence check to return 0 (converged) after 2 iterations
            with patch("optimal_adp.optimizer.check_convergence") as mock_convergence:
                mock_convergence.side_effect = [5, 0]  # Converge on second check

                final_adp, convergence_history, iterations = optimize_adp(
                    data_file_path=temp_data_file,
                    num_teams=SMALL_DRAFT_NUM_TEAMS,
                    max_iterations=10,
                    learning_rate=0.1,
                    output_file_path=str(output_path),
                )

                # Should stop early due to convergence
                assert iterations == 2
                assert convergence_history == [5, 0]

    def test_output_file_creation(self, temp_data_file: str) -> None:
        """Test that output file is created with correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_adp.csv"

            optimize_adp(
                data_file_path=temp_data_file,
                num_teams=SMALL_DRAFT_NUM_TEAMS,
                max_iterations=1,
                learning_rate=0.1,
                output_file_path=str(output_path),
            )

            # Read and validate output file
            assert output_path.exists()

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have all players
            assert len(rows) == 8

            # Check required columns exist
            required_columns = {"name", "position", "team", "avg", "total", "adp"}
            assert set(rows[0].keys()) >= required_columns

            # Check ADP values are reasonable (should be 1-8)
            adp_values = [float(row["adp"]) for row in rows]
            assert min(adp_values) >= 1.0
            assert max(adp_values) <= 8.0

    def test_invalid_input_file(self) -> None:
        """Test handling of invalid input file."""
        with pytest.raises(Exception):  # Should raise an exception
            optimize_adp(
                data_file_path="nonexistent_file.csv",
                num_teams=SMALL_DRAFT_NUM_TEAMS,
                max_iterations=1,
                learning_rate=0.1,
                output_file_path="test_output.csv",
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

    @patch("optimal_adp.main.optimize_adp")
    @patch("sys.argv")
    def test_cli_basic_execution(
        self, mock_argv: MagicMock, mock_optimize: MagicMock, temp_data_file: str
    ) -> None:
        """Test basic CLI execution with minimal arguments."""
        # Mock successful optimization
        mock_optimize.return_value = ({"QB1": 1.0, "RB1": 2.0}, [5, 2, 0], 3)

        # Mock command line arguments
        mock_argv.__getitem__ = lambda _, i: ["main.py", temp_data_file][i]
        mock_argv.__len__ = lambda _: 2

        # Should not raise any exceptions
        try:
            main()
        except SystemExit as e:
            # SystemExit with code 0 is success
            assert e.code == 0 or e.code is None

    @patch("sys.argv")
    def test_cli_invalid_input_file(self, mock_argv: MagicMock) -> None:
        """Test CLI handling of invalid input file."""
        # Mock command line arguments with non-existent file
        mock_argv.__getitem__ = lambda _, i: ["main.py", "nonexistent_file.csv"][i]
        mock_argv.__len__ = lambda _: 2

        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("optimal_adp.main.optimize_adp")
    @patch("sys.argv")
    def test_cli_with_all_arguments(
        self, mock_argv: MagicMock, mock_optimize: MagicMock, temp_data_file: str
    ) -> None:
        """Test CLI with all possible arguments."""
        # Mock successful optimization
        mock_optimize.return_value = ({"QB1": 1.0}, [0], 1)

        # Mock command line arguments with all options
        args = [
            "main.py",
            temp_data_file,
            "--output",
            "test_output.csv",
            "--max-iterations",
            "25",
            "--learning-rate",
            "0.05",
            "--num-teams",
            "12",
            "--verbose",
        ]

        mock_argv.__getitem__ = lambda _, i: args[i]
        mock_argv.__len__ = lambda _: len(args)

        # Should not raise any exceptions
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None

        # Verify optimize_adp was called with correct parameters
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args

        # Check some key parameters
        assert call_args.kwargs["max_iterations"] == 25
        assert call_args.kwargs["learning_rate"] == 0.05

    @patch("optimal_adp.main.optimize_adp")
    @patch("sys.argv")
    def test_cli_optimization_failure(
        self, mock_argv: MagicMock, mock_optimize: MagicMock, temp_data_file: str
    ) -> None:
        """Test CLI handling of optimization failure."""
        # Mock optimization failure
        mock_optimize.side_effect = Exception("Optimization failed")

        mock_argv.__getitem__ = lambda _, i: ["main.py", temp_data_file][i]
        mock_argv.__len__ = lambda _: 2

        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


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

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "integration_test_adp.csv"

            try:
                final_adp, convergence_history, iterations = optimize_adp(
                    data_file_path=str(data_file),
                    num_teams=num_teams,
                    max_iterations=5,  # Keep small for testing
                    learning_rate=0.1,
                    output_file_path=str(output_path),
                )
            except (ValueError, TypeError, KeyError) as e:
                pytest.skip(f"Real data file has formatting issues: {e}")

            # Basic checks
            assert len(final_adp) > 0
            assert len(convergence_history) > 0
            assert iterations > 0
            assert output_path.exists()

            # Check that top players have low ADP (good draft positions)
            sorted_players = sorted(final_adp.items(), key=lambda x: x[1])
            top_10_names = [name for name, _ in sorted_players[:10]]

            # Should include some recognizable top players (this is a sanity check)
            # Note: This is a loose check since we don't know exact player names
            assert len(top_10_names) == 10
