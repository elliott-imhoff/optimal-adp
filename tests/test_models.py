"""Tests for draft simulation engine."""

from optimal_adp.models import (
    DraftBoard,
    DraftState,
    Player,
    Team,
    simulate_full_draft,
)
from optimal_adp.config import NUM_TEAMS


def test_generate_snake_order() -> None:
    """Test snake order generation for 10 teams, 10 rounds."""
    # Create a DraftState with 10 teams to test the snake order generation
    players = [Player("Test Player", "QB", "TEST", 20.0, 340.0)]
    adp_mapping = {"Test Player": 1.0}

    draft_state = DraftState(players, adp_mapping, num_teams=10)
    order = draft_state.generate_snake_order()

    # Should have 100 picks total (10 teams × 10 rounds)
    assert len(order) == 100  # First round should be 0-9
    assert order[:10] == list(range(10))

    # Second round should be 9-0 (reversed)
    assert order[10:20] == list(range(9, -1, -1))

    # Third round should be 0-9 again
    assert order[20:30] == list(range(10))

    # Fourth round should be 9-0 (reversed)
    assert order[30:40] == list(range(9, -1, -1))


def test_generate_snake_order_small() -> None:
    """Test snake order for smaller draft (4 teams, 3 rounds)."""
    # Create a DraftState with 4 teams to test the snake order generation
    players = [Player("Test Player", "QB", "TEST", 20.0, 340.0)]
    adp_mapping = {"Test Player": 1.0}

    draft_state = DraftState(players, adp_mapping, num_teams=4)
    order = draft_state.generate_snake_order()

    # For 4 teams with our roster configuration, should get expected length
    # Test the pattern for first few rounds
    expected_start = [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]
    assert order[:12] == expected_start


def test_can_draft_player_qb() -> None:
    """Test QB draft eligibility."""
    qb = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    lamar_jackson = Player("Lamar Jackson", "QB", "MIA", 25.6, 434.4)
    dak_prescott = Player("Dak Prescott", "QB", "DAL", 20.1, 341.7)

    # Empty team should be able to draft QB
    empty_team = Team(
        team_id=0,
        qb_slots=[None, None],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )
    assert empty_team.can_draft_player(qb)

    # Team with 1 QB should be able to draft another
    team_with_one_qb = Team(
        team_id=0,
        qb_slots=[lamar_jackson, None],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )
    assert team_with_one_qb.can_draft_player(qb)

    # Team with full QB slots should not be able to draft QB
    team_full_qb = Team(
        team_id=0,
        qb_slots=[lamar_jackson, dak_prescott],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )
    assert not team_full_qb.can_draft_player(qb)


def test_can_draft_player_flex_eligible() -> None:
    """Test FLEX slot eligibility for RB/WR/TE."""
    rb = Player("Christian McCaffrey", "RB", "SF", 18.2, 310.0)
    wr = Player("Tyreek Hill", "WR", "MIA", 16.8, 285.0)
    te = Player("Travis Kelce", "TE", "KC", 12.4, 210.0)

    # Create some dummy players for filled slots
    josh_allen = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    lamar_jackson = Player("Lamar Jackson", "QB", "MIA", 25.6, 434.4)
    derrick_henry = Player("Derrick Henry", "RB", "TEN", 17.1, 290.7)
    nick_chubb = Player("Nick Chubb", "RB", "CLE", 16.8, 285.6)
    davante_adams = Player("Davante Adams", "WR", "LV", 15.9, 270.3)
    stefon_diggs = Player("Stefon Diggs", "WR", "BUF", 15.2, 258.4)
    deandre_hopkins = Player("DeAndre Hopkins", "WR", "ARI", 14.8, 251.6)
    mark_andrews = Player("Mark Andrews", "TE", "BAL", 11.2, 190.4)
    saquon_barkley = Player("Saquon Barkley", "RB", "NYG", 15.1, 256.7)
    cooper_kupp = Player("Cooper Kupp", "WR", "LAR", 14.6, 248.2)

    # Team with full position slots but open FLEX
    team_full_positions = Team(
        team_id=0,
        qb_slots=[josh_allen, lamar_jackson],
        rb_slots=[derrick_henry, nick_chubb],
        wr_slots=[davante_adams, stefon_diggs, deandre_hopkins],
        te_slots=[mark_andrews],
        flex_slots=[None, None],  # FLEX open
    )

    # RB/WR/TE should all be eligible for FLEX
    assert team_full_positions.can_draft_player(rb)
    assert team_full_positions.can_draft_player(wr)
    assert team_full_positions.can_draft_player(te)

    # Team with completely full roster
    team_full_roster = Team(
        team_id=0,
        qb_slots=[josh_allen, lamar_jackson],
        rb_slots=[derrick_henry, nick_chubb],
        wr_slots=[davante_adams, stefon_diggs, deandre_hopkins],
        te_slots=[mark_andrews],
        flex_slots=[saquon_barkley, cooper_kupp],
    )

    # No one should be draftable
    assert not team_full_roster.can_draft_player(rb)
    assert not team_full_roster.can_draft_player(wr)
    assert not team_full_roster.can_draft_player(te)


def test_draft_board_initialization() -> None:
    """Test DraftBoard initialization and player sorting."""
    players = [
        Player("Player A", "QB", "BUF", 20.0, 340.0),
        Player("Player B", "RB", "SF", 15.0, 255.0),
        Player("Player C", "WR", "MIA", 18.0, 306.0),
    ]

    # ADP mapping - lower ADP = higher priority
    adp_mapping = {
        "Player A": 1.5,  # Should be first
        "Player B": 3.2,  # Should be last
        "Player C": 2.1,  # Should be middle
    }

    board = DraftBoard(players, adp_mapping)

    # Players should be sorted by ADP
    assert board.available_players[0].name == "Player A"
    assert board.available_players[1].name == "Player C"
    assert board.available_players[2].name == "Player B"

    # No players drafted initially
    assert len(board.drafted_players) == 0


def test_draft_board_eligible_players() -> None:
    """Test getting eligible players for team needs."""
    players = [
        Player("QB1", "QB", "BUF", 22.0, 374.0),
        Player("RB1", "RB", "SF", 18.0, 306.0),
        Player("WR1", "WR", "MIA", 16.0, 272.0),
        Player("TE1", "TE", "KC", 12.0, 204.0),
    ]

    adp_mapping = {p.name: i + 1.0 for i, p in enumerate(players)}
    board = DraftBoard(players, adp_mapping)

    # Empty team should get all players as eligible
    empty_team = Team(
        team_id=0,
        qb_slots=[None, None],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )

    eligible = board.get_eligible_players(empty_team)
    assert len(eligible) == 4

    # Draft a player and test again
    board.draft_player(players[0])  # Draft QB1
    eligible = board.get_eligible_players(empty_team)
    assert len(eligible) == 3
    assert players[0] not in eligible


def test_draft_state_initialization() -> None:
    """Test DraftState initialization."""
    players = [
        Player("Player A", "QB", "BUF", 20.0, 340.0),
        Player("Player B", "RB", "SF", 15.0, 255.0),
    ]
    adp_mapping = {"Player A": 1.0, "Player B": 2.0}

    state = DraftState(players, adp_mapping)

    # Should have 10 teams
    assert len(state.teams) == 10

    # All teams should be empty
    for team in state.teams:
        assert team.get_open_slots()["QB"] == 2
        assert team.get_open_slots()["RB"] == 2
        assert team.get_open_slots()["WR"] == 3
        assert team.get_open_slots()["TE"] == 1
        assert team.get_open_slots()["FLEX"] == 2

    # Draft should be at beginning
    assert state.current_pick == 0
    assert len(state.draft_history) == 0

    # Pick order should be 100 picks
    assert len(state.pick_order) == 100


def test_make_greedy_pick() -> None:
    """Test greedy pick selection."""
    players = [
        Player("QB1", "QB", "BUF", 22.0, 374.0),  # ADP 1.0 - highest priority
        Player("RB1", "RB", "SF", 18.0, 306.0),  # ADP 2.0
        Player("WR1", "WR", "MIA", 16.0, 272.0),  # ADP 3.0
    ]

    adp_mapping = {"QB1": 1.0, "RB1": 2.0, "WR1": 3.0}
    state = DraftState(players, adp_mapping)

    # First pick should be QB1 (lowest ADP)
    picked_player = state.make_greedy_pick()
    assert picked_player.name == "QB1"

    # Draft state should be updated
    assert state.current_pick == 1
    assert len(state.draft_history) == 1
    assert state.draft_history[0] == (0, picked_player)

    # Team 0 should have QB1 player object
    team_0 = state.teams[0]
    assert picked_player in team_0.qb_slots


def test_greedy_pick_tie_breaker() -> None:
    """Test tie-breaker using highest average score."""
    players = [
        Player("QB1", "QB", "BUF", 20.0, 340.0),  # Same ADP, lower avg
        Player("QB2", "QB", "MIA", 22.0, 374.0),  # Same ADP, higher avg
    ]

    # Same ADP for both
    adp_mapping = {"QB1": 1.0, "QB2": 1.0}
    state = DraftState(players, adp_mapping)

    # Should pick QB2 (higher avg as tie-breaker)
    picked_player = state.make_greedy_pick()
    assert picked_player.name == "QB2"


def test_draft_state_cloning() -> None:
    """Test draft state cloning for counterfactual analysis."""
    players = [
        Player("QB1", "QB", "BUF", 22.0, 374.0),
        Player("RB1", "RB", "SF", 18.0, 306.0),
    ]
    adp_mapping = {"QB1": 1.0, "RB1": 2.0}

    original_state = DraftState(players, adp_mapping)

    # Make a pick
    original_state.make_greedy_pick()

    # Clone the state
    cloned_state = original_state.clone()

    # Should be independent copies
    assert cloned_state is not original_state
    assert cloned_state.teams is not original_state.teams
    assert cloned_state.draft_board is not original_state.draft_board

    # But should have same data
    assert cloned_state.current_pick == original_state.current_pick
    assert len(cloned_state.draft_history) == len(original_state.draft_history)

    # Modifying clone shouldn't affect original
    cloned_state.make_greedy_pick()
    assert cloned_state.current_pick == 2
    assert original_state.current_pick == 1


def test_draft_state_rewind() -> None:
    """Test rewinding draft state to previous pick."""
    players = [
        Player("QB1", "QB", "BUF", 22.0, 374.0),
        Player("RB1", "RB", "SF", 18.0, 306.0),
        Player("WR1", "WR", "MIA", 16.0, 272.0),
    ]
    adp_mapping = {"QB1": 1.0, "RB1": 2.0, "WR1": 3.0}

    state = DraftState(players, adp_mapping)

    # Make 3 picks
    state.make_greedy_pick()  # Pick 0
    state.make_greedy_pick()  # Pick 1
    state.make_greedy_pick()  # Pick 2

    assert state.current_pick == 3
    assert len(state.draft_history) == 3

    # Rewind to pick 1
    rewound_state = state.rewind_to_pick(1)

    # Should be back to state after pick 0 (before pick 1)
    assert rewound_state.current_pick == 1
    assert len(rewound_state.draft_history) == 1

    # Original state should be unchanged
    assert state.current_pick == 3
    assert len(state.draft_history) == 3


def test_simulate_from_pick() -> None:
    """Test continuing simulation from arbitrary pick."""
    players = [
        Player("QB1", "QB", "BUF", 22.0, 374.0),
        Player("RB1", "RB", "SF", 18.0, 306.0),
        Player("WR1", "WR", "MIA", 16.0, 272.0),
        Player("TE1", "TE", "KC", 12.0, 204.0),
    ]
    adp_mapping = {p.name: i + 1.0 for i, p in enumerate(players)}

    state = DraftState(players, adp_mapping)

    # Make 2 picks manually
    state.make_greedy_pick()  # Pick 0
    state.make_greedy_pick()  # Pick 1

    assert state.current_pick == 2

    # Continue simulation from pick 2
    final_state = state.simulate_from_pick(2)

    # Should have made 2 more picks (limited by available players)
    assert final_state.current_pick == 4
    assert len(final_state.draft_history) == 4


def test_simulate_full_draft_small() -> None:
    """Test complete draft simulation with small player pool."""
    # Create enough players for a few rounds
    players = []
    positions = ["QB", "RB", "WR", "TE"]

    for i in range(40):  # 40 players total
        pos = positions[i % 4]
        players.append(
            Player(f"{pos}{i//4 + 1}", pos, "TEAM", 15.0 + i * 0.1, 255.0 + i * 1.7)
        )

    adp_mapping = {p.name: i + 1.0 for i, p in enumerate(players)}

    final_state = simulate_full_draft(players, adp_mapping)

    # Should have made as many picks as possible
    # With 40 players and 10 teams, should make 40 picks (4 rounds)
    assert len(final_state.draft_history) == 40
    assert final_state.current_pick == 40

    # All players should be drafted
    assert len(final_state.draft_board.drafted_players) == 40


def test_team_add_player() -> None:
    """Test Team.add_player() method."""
    team = Team(
        team_id=0,
        qb_slots=[None, None],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )

    qb = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    rb = Player("Christian McCaffrey", "RB", "SF", 18.2, 310.0)

    # Should be able to add QB to QB slot
    assert team.add_player(qb)
    assert team.qb_slots[0] == qb

    # Should be able to add RB to RB slot
    assert team.add_player(rb)
    assert team.rb_slots[0] == rb


def test_team_get_open_slots() -> None:
    """Test Team.get_open_slots() method."""
    josh_allen = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    tyreek_hill = Player("Tyreek Hill", "WR", "MIA", 16.8, 285.0)
    davante_adams = Player("Davante Adams", "WR", "LV", 15.9, 270.3)

    team = Team(
        team_id=0,
        qb_slots=[josh_allen, None],  # 1 open
        rb_slots=[None, None],  # 2 open
        wr_slots=[tyreek_hill, davante_adams, None],  # 1 open
        te_slots=[None],  # 1 open
        flex_slots=[None, None],  # 2 open
    )

    open_slots = team.get_open_slots()

    assert open_slots["QB"] == 1
    assert open_slots["RB"] == 2
    assert open_slots["WR"] == 1
    assert open_slots["TE"] == 1
    assert open_slots["FLEX"] == 2


def test_team_is_roster_full() -> None:
    """Test Team.is_roster_full() method."""
    # Empty team
    empty_team = Team(
        team_id=0,
        qb_slots=[None, None],
        rb_slots=[None, None],
        wr_slots=[None, None, None],
        te_slots=[None],
        flex_slots=[None, None],
    )
    assert not empty_team.is_roster_full()

    # Full team - create player objects
    josh_allen = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    lamar_jackson = Player("Lamar Jackson", "QB", "MIA", 25.6, 434.4)
    christian_mccaffrey = Player("Christian McCaffrey", "RB", "SF", 18.2, 310.0)
    derrick_henry = Player("Derrick Henry", "RB", "TEN", 17.1, 290.7)
    tyreek_hill = Player("Tyreek Hill", "WR", "MIA", 16.8, 285.0)
    davante_adams = Player("Davante Adams", "WR", "LV", 15.9, 270.3)
    stefon_diggs = Player("Stefon Diggs", "WR", "BUF", 15.2, 258.4)
    travis_kelce = Player("Travis Kelce", "TE", "KC", 12.4, 210.0)
    saquon_barkley = Player("Saquon Barkley", "RB", "NYG", 15.1, 256.7)
    cooper_kupp = Player("Cooper Kupp", "WR", "LAR", 14.6, 248.2)

    full_team = Team(
        team_id=0,
        qb_slots=[josh_allen, lamar_jackson],
        rb_slots=[christian_mccaffrey, derrick_henry],
        wr_slots=[tyreek_hill, davante_adams, stefon_diggs],
        te_slots=[travis_kelce],
        flex_slots=[saquon_barkley, cooper_kupp],
    )
    assert full_team.is_roster_full()


def test_team_calculate_total_score() -> None:
    """Test Team.calculate_total_score() method."""
    josh_allen = Player("Josh Allen", "QB", "BUF", 22.6, 385.0)
    christian_mccaffrey = Player("Christian McCaffrey", "RB", "SF", 18.2, 310.0)
    tyreek_hill = Player("Tyreek Hill", "WR", "MIA", 16.8, 285.0)
    travis_kelce = Player("Travis Kelce", "TE", "KC", 12.4, 210.0)
    saquon_barkley = Player("Saquon Barkley", "RB", "NYG", 15.1, 257.0)

    team = Team(
        team_id=0,
        qb_slots=[josh_allen, None],  # 22.6 points
        rb_slots=[christian_mccaffrey, None],  # 18.2 points
        wr_slots=[tyreek_hill, None, None],  # 16.8 points
        te_slots=[travis_kelce],  # 12.4 points
        flex_slots=[saquon_barkley, None],  # 15.1 points
    )

    total_score = team.calculate_total_score()
    expected_score = 22.6 + 18.2 + 16.8 + 12.4 + 15.1

    assert (
        abs(total_score - expected_score) < 0.001
    )  # Account for floating point precision


def test_player_dataclass() -> None:
    """Test Player dataclass creation and attributes."""
    player = Player(name="Josh Allen", position="QB", team="BUF", avg=22.6, total=385.0)

    assert player.name == "Josh Allen"
    assert player.position == "QB"
    assert player.team == "BUF"
    assert player.avg == 22.6
    assert player.total == 385.0


def test_team_dataclass() -> None:
    """Test Team dataclass for roster tracking."""
    # Test creating empty team
    team = Team(
        team_id=0,
        qb_slots=[None, None],  # 2 QB slots
        rb_slots=[None, None],  # 2 RB slots
        wr_slots=[None, None, None],  # 3 WR slots
        te_slots=[None],  # 1 TE slot
        flex_slots=[None, None],  # 2 FLEX slots
    )

    assert team.team_id == 0
    assert len(team.qb_slots) == 2
    assert len(team.rb_slots) == 2
    assert len(team.wr_slots) == 3
    assert len(team.te_slots) == 1
    assert len(team.flex_slots) == 2
    assert all(slot is None for slot in team.qb_slots)


def test_team_auto_init() -> None:
    """Test Team dataclass auto-initialization of empty slots."""
    team = Team(
        team_id=1, qb_slots=[], rb_slots=[], wr_slots=[], te_slots=[], flex_slots=[]
    )

    assert len(team.qb_slots) == 2
    assert len(team.rb_slots) == 2
    assert len(team.wr_slots) == 3
    assert len(team.te_slots) == 1
    assert len(team.flex_slots) == 2
    assert all(
        slot is None
        for slot in team.qb_slots
        + team.rb_slots
        + team.wr_slots
        + team.te_slots
        + team.flex_slots
    )


def test_draft_functions() -> None:
    """Test draft configuration functions."""
    # Test global NUM_TEAMS
    assert NUM_TEAMS == 10

    # Test snake order generation through DraftState
    players = [Player("Test Player", "QB", "TEST", 20.0, 340.0)]
    adp_mapping = {"Test Player": 1.0}
    draft_state = DraftState(players, adp_mapping, num_teams=4)
    snake_order = draft_state.generate_snake_order()

    expected_length = 40  # 4 teams × 10 roster slots
    assert len(snake_order) == expected_length

    # Test first few picks are in correct snake pattern
    assert snake_order[0] == 0  # First pick: team 0
    assert snake_order[1] == 1  # Second pick: team 1
    assert snake_order[2] == 2  # Third pick: team 2
    assert snake_order[3] == 3  # Fourth pick: team 3
    # Snake back
    assert snake_order[4] == 3  # Fifth pick: team 3 (reverse order)
    assert snake_order[5] == 2  # Sixth pick: team 2
