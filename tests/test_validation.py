"""Tests for the validation module."""

import random
import tempfile
import pytest

from optimal_adp.config import NUM_TEAMS
from optimal_adp.models import Player
from optimal_adp.validation import (
    ValidationResult,
    perturb_initial_adp,
    validate_position_hierarchy,
    validate_elite_players_first_round,
    validate_optimization_results,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_initialization(self) -> None:
        """Test ValidationResult initializes correctly."""
        result = ValidationResult()
        assert result.passed is True
        assert result.messages == []
        assert result.convergence_iterations is None
        assert result.final_position_changes is None

    def test_add_failure(self) -> None:
        """Test adding failure changes passed status."""
        result = ValidationResult()
        result.add_failure("Test failure")
        assert result.passed is False
        assert len(result.messages) == 1
        assert "❌ Test failure" in result.messages[0]

    def test_add_success(self) -> None:
        """Test adding success doesn't change passed status."""
        result = ValidationResult()
        result.add_success("Test success")
        assert result.passed is True
        assert len(result.messages) == 1
        assert "✅ Test success" in result.messages[0]

    def test_add_info(self) -> None:
        """Test adding info doesn't change passed status."""
        result = ValidationResult()
        result.add_info("Test info")
        assert result.passed is True
        assert len(result.messages) == 1
        assert "ℹ️  Test info" in result.messages[0]


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


class TestValidatePositionHierarchy:
    """Tests for validate_position_hierarchy function."""

    def test_valid_hierarchy(self) -> None:
        """Test validation passes with correct hierarchy."""
        players = [
            Player("QB1", "QB", "TEAM1", 25.0, 400.0),
            Player("QB2", "QB", "TEAM2", 20.0, 300.0),
            Player("RB1", "RB", "TEAM3", 22.0, 350.0),
            Player("RB2", "RB", "TEAM4", 18.0, 250.0),
        ]

        # ADP should match AVG hierarchy (lower ADP = better)
        final_adp = {
            "QB1": 5.0,  # Best QB
            "QB2": 15.0,  # Worse QB
            "RB1": 8.0,  # Best RB
            "RB2": 20.0,  # Worse RB
        }

        is_valid, violations = validate_position_hierarchy(final_adp, players)
        assert is_valid is True
        assert len(violations) == 0

    def test_invalid_hierarchy(self) -> None:
        """Test validation fails with incorrect hierarchy."""
        players = [
            Player("QB1", "QB", "TEAM1", 25.0, 400.0),  # Better QB
            Player("QB2", "QB", "TEAM2", 20.0, 300.0),  # Worse QB
        ]

        # Incorrect ADP ordering (worse player has better ADP)
        final_adp = {
            "QB1": 15.0,  # Better player with worse ADP
            "QB2": 5.0,  # Worse player with better ADP
        }

        is_valid, violations = validate_position_hierarchy(final_adp, players)
        assert is_valid is False
        assert len(violations) == 1
        assert "QB2" in violations[0] and "QB1" in violations[0]

    def test_empty_players(self) -> None:
        """Test validation with no players."""
        is_valid, violations = validate_position_hierarchy({}, [])
        assert is_valid is True
        assert len(violations) == 0


class TestValidateElitePlayersFirstRound:
    """Tests for validate_elite_players_first_round function."""

    def test_elite_players_in_first_round(self) -> None:
        """Test validation passes when elite players in first round."""
        players = [
            Player("TopQB", "QB", "TEAM1", 30.0, 500.0),  # Best QB
            Player("QB2", "QB", "TEAM2", 25.0, 400.0),  # Second QB
            Player("TopRB", "RB", "TEAM3", 28.0, 450.0),  # Best RB
            Player("RB2", "RB", "TEAM4", 22.0, 350.0),  # Second RB
            Player("TopWR", "WR", "TEAM5", 26.0, 420.0),  # Best WR
            Player("WR2", "WR", "TEAM6", 20.0, 300.0),  # Second WR
        ]

        # Elite players all in first round (ADP 1-10 for 10-team league)
        final_adp = {
            "TopQB": 3.0,
            "TopRB": 1.0,
            "TopWR": 2.0,
            "QB2": 15.0,
            "RB2": 12.0,
            "WR2": 18.0,
        }

        is_valid, violations = validate_elite_players_first_round(
            final_adp, players, num_teams=10
        )
        assert is_valid is True
        assert len(violations) == 0

    def test_elite_player_outside_first_round(self) -> None:
        """Test validation fails when elite player outside first round."""
        players = [
            Player("TopQB", "QB", "TEAM1", 30.0, 500.0),  # Best QB
            Player("TopRB", "RB", "TEAM3", 28.0, 450.0),  # Best RB
            Player("TopWR", "WR", "TEAM5", 26.0, 420.0),  # Best WR
        ]

        # Top QB outside first round
        final_adp = {
            "TopQB": 15.0,  # Outside first round (>10)
            "TopRB": 1.0,
            "TopWR": 2.0,
        }

        is_valid, violations = validate_elite_players_first_round(
            final_adp, players, num_teams=10
        )
        assert is_valid is False
        assert len(violations) == 1
        assert "TopQB" in violations[0]

    def test_missing_position(self) -> None:
        """Test validation handles missing position gracefully."""
        players: list[Player] = []  # No players
        final_adp: dict[str, float] = {}

        is_valid, violations = validate_elite_players_first_round(final_adp, players)
        assert is_valid is False
        assert len(violations) == 3  # Should report all 3 positions missing


class TestValidationHelpers:
    """Tests for focused validation helper functions."""

    def test_validate_optimization_results_with_mock(self) -> None:
        """Test validation helper function integration."""
        # Test the pure validation logic without I/O

        # Setup test data
        mock_players = [
            Player("TopQB", "QB", "TEAM1", 30.0, 500.0),
            Player("TopRB", "RB", "TEAM3", 28.0, 450.0),
            Player("TopWR", "WR", "TEAM5", 26.0, 420.0),
        ]

        final_adp = {p.name: float(i + 1) for i, p in enumerate(mock_players)}

        # Test successful validation (converged before max iterations)
        result = validate_optimization_results(
            players=mock_players,
            final_adp=final_adp,
            iterations=5,
            max_iterations=50,
            num_teams=NUM_TEAMS,
        )

        assert result.all_passed() is True
        assert any("Converged after 5 iterations" in msg for msg in result.messages)
        assert any("All validation checks passed" in msg for msg in result.messages)


class TestValidateOptimization:
    """Tests for validate_optimization_results function."""

    def test_validate_optimization_success(self) -> None:
        """Test main validation function with successful case."""
        # Test the unified validation function

        # Setup mock data that will pass validation
        mock_players = [
            Player("TopQB", "QB", "TEAM1", 30.0, 500.0),
            Player("TopRB", "RB", "TEAM3", 28.0, 450.0),
            Player("TopWR", "WR", "TEAM5", 26.0, 420.0),
        ]

        final_adp = {p.name: float(i + 1) for i, p in enumerate(mock_players)}

        result = validate_optimization_results(
            players=mock_players,
            final_adp=final_adp,
            iterations=10,  # less than max
            max_iterations=1000,
            num_teams=10,
        )

        assert result.all_passed() is True

    def test_validate_optimization_failure(self) -> None:
        """Test main validation function with failure case."""
        # Test case that will fail validation (hit max iterations)

        mock_players = [Player("Player1", "QB", "TEAM1", 30.0, 500.0)]
        final_adp = {"Player1": 1.0}

        result = validate_optimization_results(
            players=mock_players,
            final_adp=final_adp,
            iterations=1000,  # hit max iterations
            max_iterations=1000,
            num_teams=10,
        )

        assert result.all_passed() is False


# Fixtures for testing
@pytest.fixture
def temp_data_file() -> str:
    """Create temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Write minimal CSV data
        f.write("name,position,team,avg,total\n")
        f.write("Player1,QB,TEAM1,25.0,400.0\n")
        f.write("Player2,RB,TEAM2,20.0,300.0\n")
        return f.name
