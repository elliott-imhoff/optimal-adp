"""Tests for regret calculation and ADP optimization logic."""

import pytest
from unittest.mock import patch, MagicMock

from optimal_adp.config import Player, DraftConfig
from optimal_adp.draft_simulator import DraftState
from optimal_adp.regret_optimizer import (
    calculate_pick_regret,
    calculate_all_regrets,
    update_adp_from_regret,
    rescale_adp_to_picks,
    validate_position_hierarchy,
    check_convergence,
)


@pytest.fixture
def sample_players() -> list[Player]:
    """Create sample players for testing."""
    return [
        # QBs
        Player(name="QB1", position="QB", team="KC", avg=25.0, total=400.0),
        Player(name="QB2", position="QB", team="BUF", avg=20.0, total=320.0),
        Player(name="QB3", position="QB", team="LAR", avg=18.0, total=290.0),
        Player(name="QB4", position="QB", team="DAL", avg=16.0, total=260.0),
        # RBs
        Player(name="RB1", position="RB", team="SF", avg=18.0, total=290.0),
        Player(name="RB2", position="RB", team="DAL", avg=15.0, total=240.0),
        Player(name="RB3", position="RB", team="CLE", avg=14.0, total=225.0),
        Player(name="RB4", position="RB", team="NYG", avg=13.0, total=210.0),
        # WRs
        Player(name="WR1", position="WR", team="MIA", avg=16.0, total=250.0),
        Player(name="WR2", position="WR", team="CIN", avg=14.0, total=220.0),
        Player(name="WR3", position="WR", team="GB", avg=13.0, total=210.0),
        Player(name="WR4", position="WR", team="TB", avg=12.0, total=195.0),
        Player(name="WR5", position="WR", team="SEA", avg=11.0, total=180.0),
        Player(name="WR6", position="WR", team="LV", avg=10.0, total=165.0),
        # TEs
        Player(name="TE1", position="TE", team="KC", avg=12.0, total=190.0),
        Player(name="TE2", position="TE", team="SF", avg=10.0, total=160.0),
    ]


@pytest.fixture
def sample_adp() -> dict[str, float]:
    """Create sample ADP mapping."""
    return {
        "QB1": 1.0,
        "RB1": 2.0,
        "WR1": 3.0,
        "QB2": 4.0,
        "RB2": 5.0,
        "WR2": 6.0,
        "TE1": 7.0,
        "WR3": 8.0,
        "RB3": 9.0,
        "QB3": 10.0,
        "WR4": 11.0,
        "RB4": 12.0,
        "WR5": 13.0,
        "TE2": 14.0,
        "QB4": 15.0,
        "WR6": 16.0,
    }


@pytest.fixture
def simple_draft_config() -> DraftConfig:
    """Create simple 2-team draft config for testing."""
    return DraftConfig(num_teams=2)


@pytest.fixture
def completed_draft_state(
    sample_players: list[Player],
    sample_adp: dict[str, float],
    simple_draft_config: DraftConfig,
) -> DraftState:
    """Create a simple completed draft state for testing."""
    # Create minimal draft state for testing regret calculation
    draft_state = DraftState(sample_players, sample_adp, simple_draft_config)

    # Manually set up a simple draft with just 4 picks for easier testing
    # Pick 0: Team 0 drafts QB1
    # Pick 1: Team 1 drafts RB1
    # Pick 2: Team 1 drafts QB2
    # Pick 3: Team 0 drafts RB2

    draft_state.draft_history = [
        (0, sample_players[0]),  # QB1 to team 0
        (1, sample_players[4]),  # RB1 to team 1
        (2, sample_players[1]),  # QB2 to team 1
        (3, sample_players[5]),  # RB2 to team 0
    ]

    # Set up minimal team rosters for the players that were drafted
    draft_state.teams[0].add_player(sample_players[0])  # QB1
    draft_state.teams[0].add_player(sample_players[5])  # RB2
    draft_state.teams[1].add_player(sample_players[4])  # RB1
    draft_state.teams[1].add_player(sample_players[1])  # QB2

    # Mark players as drafted
    for _, player in draft_state.draft_history:
        draft_state.draft_board.drafted_players.add(player.name)

    draft_state.current_pick = 4  # Draft complete for testing

    return draft_state


class TestCalculatePickRegret:
    """Tests for calculate_pick_regret function."""

    def test_invalid_pick_number_negative(
        self, completed_draft_state: DraftState
    ) -> None:
        """Test error handling for negative pick numbers."""
        with pytest.raises(ValueError, match="Invalid pick number"):
            calculate_pick_regret(completed_draft_state, -1)

    def test_invalid_pick_number_too_high(
        self, completed_draft_state: DraftState
    ) -> None:
        """Test error handling for pick numbers beyond draft history."""
        with pytest.raises(ValueError, match="Invalid pick number"):
            calculate_pick_regret(completed_draft_state, 10)

    @patch("optimal_adp.regret_optimizer.simulate_from_pick")
    def test_regret_calculation_logic(
        self, mock_simulate: MagicMock, completed_draft_state: DraftState
    ) -> None:
        """Test the core regret calculation logic."""
        # Mock counterfactual simulation to return known result
        mock_counterfactual_state = MagicMock()
        mock_counterfactual_team = MagicMock()
        mock_counterfactual_team.calculate_total_score.return_value = 45.0
        # Set up mock teams list properly indexed
        mock_counterfactual_state.teams = [mock_counterfactual_team, MagicMock()]
        mock_simulate.return_value = mock_counterfactual_state

        # Team 0 should have QB1 (25.0) + RB2 (15.0) = 40.0 total
        regret = calculate_pick_regret(completed_draft_state, 0)

        # Regret = counterfactual (45.0) - original (40.0) = 5.0
        assert regret == 5.0

        # Verify that the originally drafted player was removed from available pool
        mock_simulate.assert_called_once()
        args, _ = mock_simulate.call_args
        counterfactual_state = args[0]
        assert "QB1" in counterfactual_state.draft_board.drafted_players

    def test_original_draft_not_modified(
        self, completed_draft_state: DraftState
    ) -> None:
        """Test that original draft state is not modified during regret calculation."""
        # Capture original state
        original_history_length = len(completed_draft_state.draft_history)
        original_current_pick = completed_draft_state.current_pick
        original_drafted_players = (
            completed_draft_state.draft_board.drafted_players.copy()
        )

        # Calculate regret for pick 0
        with patch("optimal_adp.regret_optimizer.simulate_from_pick") as mock_simulate:
            mock_counterfactual_state = MagicMock()
            mock_counterfactual_team = MagicMock()
            mock_counterfactual_team.calculate_total_score.return_value = 45.0
            mock_counterfactual_state.teams = [mock_counterfactual_team, MagicMock()]
            mock_simulate.return_value = mock_counterfactual_state

            calculate_pick_regret(completed_draft_state, 0)

        # Verify original draft state is unchanged
        assert len(completed_draft_state.draft_history) == original_history_length
        assert completed_draft_state.current_pick == original_current_pick
        assert (
            completed_draft_state.draft_board.drafted_players
            == original_drafted_players
        )

    def test_position_hierarchy_violation_high_regret(
        self, sample_players: list[Player], sample_adp: dict[str, float]
    ) -> None:
        """Test that hierarchy violations result in high regret scores."""
        # Create a draft where a worse QB is picked before a better one
        draft_config = DraftConfig(num_teams=2)
        draft_state = DraftState(sample_players, sample_adp, draft_config)

        # Set up a scenario where QB2 (20.0 avg) is drafted when QB1 (25.0 avg) was available
        # Draft history: Pick 0 - Team 0 drafts QB2 (worse choice)
        draft_state.draft_history = [(0, sample_players[1])]  # QB2
        draft_state.teams[0].add_player(sample_players[1])  # QB2 to team 0
        draft_state.draft_board.drafted_players.add("QB2")
        draft_state.current_pick = 1

        # Calculate regret - should be high due to hierarchy violation
        regret = calculate_pick_regret(draft_state, 0)

        # Should be high regret (QB1 has 5.0 higher avg * 10 multiplier = 50.0)
        expected_regret = (25.0 - 20.0) * 10.0  # 50.0
        assert regret == expected_regret

    def test_position_hierarchy_no_violation(
        self, sample_players: list[Player], sample_adp: dict[str, float]
    ) -> None:
        """Test that picking the best available player doesn't trigger hierarchy violation."""
        # Create draft where best QB is picked first
        draft_config = DraftConfig(num_teams=2)
        draft_state = DraftState(sample_players, sample_adp, draft_config)

        # Draft history: Pick 0 - Team 0 drafts QB1 (best choice)
        draft_state.draft_history = [(0, sample_players[0])]  # QB1 (best QB)
        draft_state.teams[0].add_player(sample_players[0])
        draft_state.draft_board.drafted_players.add("QB1")
        draft_state.current_pick = 1

        # Mock counterfactual simulation since no hierarchy violation
        with patch("optimal_adp.regret_optimizer.simulate_from_pick") as mock_simulate:
            mock_counterfactual_state = MagicMock()
            mock_counterfactual_team = MagicMock()
            mock_counterfactual_team.calculate_total_score.return_value = 30.0
            mock_counterfactual_state.teams = [mock_counterfactual_team, MagicMock()]
            mock_simulate.return_value = mock_counterfactual_state

            regret = calculate_pick_regret(draft_state, 0)

        # Should use counterfactual calculation, not hierarchy violation
        # regret = counterfactual (30.0) - original (25.0) = 5.0
        assert regret == 5.0

    def test_position_hierarchy_different_position_no_violation(
        self, sample_players: list[Player], sample_adp: dict[str, float]
    ) -> None:
        """Test that better players of different positions don't trigger violations."""
        # Create draft where RB is picked when better QB is available (different position)
        draft_config = DraftConfig(num_teams=2)
        draft_state = DraftState(sample_players, sample_adp, draft_config)

        # Draft history: Pick 0 - Team 0 drafts RB1 when QB1 is available (different positions)
        draft_state.draft_history = [(0, sample_players[4])]  # RB1
        draft_state.teams[0].add_player(sample_players[4])
        draft_state.draft_board.drafted_players.add("RB1")
        draft_state.current_pick = 1

        # Mock counterfactual simulation since no hierarchy violation (different positions)
        with patch("optimal_adp.regret_optimizer.simulate_from_pick") as mock_simulate:
            mock_counterfactual_state = MagicMock()
            mock_counterfactual_team = MagicMock()
            mock_counterfactual_team.calculate_total_score.return_value = 20.0
            mock_counterfactual_state.teams = [mock_counterfactual_team, MagicMock()]
            mock_simulate.return_value = mock_counterfactual_state

            regret = calculate_pick_regret(draft_state, 0)

        # Should use counterfactual calculation, not hierarchy violation
        # regret = counterfactual (20.0) - original (18.0) = 2.0
        assert regret == 2.0


class TestCalculateAllRegrets:
    """Tests for calculate_all_regrets function."""

    @patch("optimal_adp.regret_optimizer.calculate_pick_regret")
    def test_calculate_all_regrets(
        self, mock_calculate_pick: MagicMock, completed_draft_state: DraftState
    ) -> None:
        """Test that regret is calculated for all picks in draft history."""
        # Mock individual regret calculations
        mock_calculate_pick.side_effect = [1.0, 2.0, 3.0, 4.0]

        regrets = calculate_all_regrets(completed_draft_state)

        # Should have regret for each player
        expected_regrets = {
            "QB1": 1.0,
            "RB1": 2.0,
            "QB2": 3.0,
            "RB2": 4.0,
        }
        assert regrets == expected_regrets

        # Should have called calculate_pick_regret for each pick
        assert mock_calculate_pick.call_count == 4


class TestUpdateAdpFromRegret:
    """Tests for update_adp_from_regret function."""

    def test_basic_update(self) -> None:
        """Test basic ADP update calculation using raw regret."""
        current_adp = {"Player1": 10.0, "Player2": 20.0}
        player_regrets = {
            "Player1": -2.0,
            "Player2": 1.5,
        }  # Player1 had negative regret (good pick)
        learning_rate = 0.5

        updated = update_adp_from_regret(current_adp, player_regrets, learning_rate)

        # Player1: 10.0 + 0.5 * (-2.0) = 9.0 (earlier pick due to negative regret)
        # Player2: 20.0 + 0.5 * 1.5 = 20.75 (later pick due to positive regret)
        assert updated["Player1"] == 9.0
        assert updated["Player2"] == 20.75

    def test_missing_player_in_adp(self) -> None:
        """Test that players not in current_adp are ignored."""
        current_adp = {"Player1": 10.0}
        player_regrets = {"Player1": 2.0, "Player2": -1.0}
        learning_rate = 0.5

        updated = update_adp_from_regret(current_adp, player_regrets, learning_rate)

        # Only Player1 should be updated
        assert len(updated) == 1
        assert "Player2" not in updated
        assert updated["Player1"] == 11.0  # 10.0 + 0.5 * 2.0

    def test_zero_regret_no_change(self) -> None:
        """Test that zero regret results in no ADP change."""
        current_adp = {"Player1": 15.0}
        player_regrets = {"Player1": 0.0}
        learning_rate = 1.0

        updated = update_adp_from_regret(current_adp, player_regrets, learning_rate)

        assert updated["Player1"] == 15.0  # No change


class TestRescaleAdpToPicks:
    """Tests for rescale_adp_to_picks function."""

    def test_empty_adp(self) -> None:
        """Test handling of empty ADP dictionary."""
        result = rescale_adp_to_picks({})
        assert result == {}

    def test_sequential_assignment(self) -> None:
        """Test that players are assigned sequential pick numbers."""
        updated_adp = {
            "Player1": -5.0,  # Best (lowest ADP)
            "Player2": 100.0,  # Worst (highest ADP)
            "Player3": 15.5,  # Middle
        }

        rescaled = rescale_adp_to_picks(updated_adp)

        assert rescaled["Player1"] == 1.0  # First pick
        assert rescaled["Player3"] == 2.0  # Second pick
        assert rescaled["Player2"] == 3.0  # Third pick

    def test_maintains_relative_order(self) -> None:
        """Test that relative ordering is maintained after rescaling."""
        updated_adp = {"A": 1.0, "B": 2.0, "C": 1.5}
        rescaled = rescale_adp_to_picks(updated_adp)

        # Order should be A (1.0), C (1.5), B (2.0)
        assert rescaled["A"] < rescaled["C"] < rescaled["B"]


class TestValidatePositionHierarchy:
    """Tests for validate_position_hierarchy function."""

    def test_valid_hierarchy(self, sample_players: list[Player]) -> None:
        """Test validation when hierarchy is maintained."""
        # QB1 (25.0 avg) has ADP 1, QB2 (20.0 avg) has ADP 2
        valid_adp = {
            "QB1": 1.0,
            "QB2": 2.0,
            "RB1": 3.0,
            "RB2": 4.0,
        }

        is_valid = validate_position_hierarchy(valid_adp, sample_players)
        assert is_valid is True

    def test_invalid_hierarchy(self, sample_players: list[Player]) -> None:
        """Test validation when hierarchy is violated."""
        # QB2 (20.0 avg) has earlier ADP than QB1 (25.0 avg) - violation!
        invalid_adp = {
            "QB1": 2.0,  # Lower avg but later pick
            "QB2": 1.0,  # Higher avg but earlier pick
            "RB1": 3.0,
            "RB2": 4.0,
        }

        is_valid = validate_position_hierarchy(invalid_adp, sample_players)
        assert is_valid is False

    def test_players_not_in_adp_ignored(self, sample_players: list[Player]) -> None:
        """Test that players not in ADP dict are ignored."""
        partial_adp = {
            "QB1": 1.0,  # Only include some players
            "RB1": 2.0,
        }

        # Should not crash and should validate the included players
        is_valid = validate_position_hierarchy(partial_adp, sample_players)
        assert is_valid is True


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_no_position_changes_converged(self) -> None:
        """Test that no position changes means convergence."""
        initial_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0}
        final_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0}

        position_changes = check_convergence(initial_adp, final_adp)
        assert position_changes == 0

    def test_single_position_change(self) -> None:
        """Test counting a single position change."""
        initial_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0}
        final_adp = {"Player1": 2.0, "Player2": 1.0, "Player3": 3.0}  # P1 and P2 swap

        position_changes = check_convergence(initial_adp, final_adp)
        assert position_changes == 2  # Both players that swapped count as changes

    def test_multiple_position_changes(self) -> None:
        """Test counting multiple position changes."""
        initial_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0, "Player4": 4.0}
        final_adp = {
            "Player1": 4.0,
            "Player2": 3.0,
            "Player3": 2.0,
            "Player4": 1.0,
        }  # All positions change

        position_changes = check_convergence(initial_adp, final_adp)
        # P1: rank 1→4 (3 moves), P2: rank 2→3 (1 move), P3: rank 3→2 (1 move), P4: rank 4→1 (3 moves)
        assert position_changes == 8  # Total magnitude of all position changes

    def test_adp_values_change_but_rankings_same(self) -> None:
        """Test that only ranking changes matter, not ADP value changes."""
        initial_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0}
        final_adp = {
            "Player1": 10.0,
            "Player2": 20.0,
            "Player3": 30.0,
        }  # Values change but order same

        position_changes = check_convergence(initial_adp, final_adp)
        assert position_changes == 0  # Rankings unchanged, so converged

    def test_empty_adp_dictionaries(self) -> None:
        """Test handling of empty ADP dictionaries."""
        position_changes = check_convergence({}, {})
        assert position_changes == 0

    def test_partial_overlap_players(self) -> None:
        """Test handling when player sets differ between initial and final."""
        initial_adp = {"Player1": 1.0, "Player2": 2.0, "Player3": 3.0}
        final_adp = {
            "Player1": 1.0,
            "Player2": 3.0,
            "Player4": 2.0,
        }  # Player3 gone, Player4 added

        position_changes = check_convergence(initial_adp, final_adp)
        # Player1: rank 1 → rank 1 (no change)
        # Player2: rank 2 → rank 3 (change)
        # Player3: not in final, ignored
        # Only count changes for players in both dictionaries
        assert position_changes == 1

    def test_magnitude_of_position_changes(self) -> None:
        """Test that position changes count the magnitude of moves."""
        initial_adp = {
            "Player1": 1.0,
            "Player2": 2.0,
            "Player3": 3.0,
            "Player4": 4.0,
            "Player5": 5.0,
        }
        final_adp = {
            "Player1": 3.0,
            "Player2": 2.0,
            "Player3": 1.0,
            "Player4": 5.0,
            "Player5": 4.0,
        }
        # Final rankings: P3=rank1 (1.0), P2=rank2 (2.0), P1=rank3 (3.0), P5=rank4 (4.0), P4=rank5 (5.0)
        # P1: rank 1→3 (2 moves), P2: rank 2→2 (0 moves), P3: rank 3→1 (2 moves)
        # P4: rank 4→5 (1 move), P5: rank 5→4 (1 move)

        position_changes = check_convergence(initial_adp, final_adp)
        assert position_changes == 6  # 2 + 0 + 2 + 1 + 1 = 6
