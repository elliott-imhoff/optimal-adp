"""Tests for configuration data structures."""

from optimal_adp.config import (
    NUM_TEAMS,
    generate_snake_order,
    get_total_rounds,
    get_total_picks,
    OptimizationConfig,
    Player,
    Team,
    ROSTER_SLOTS,
)


def test_roster_slots_constant() -> None:
    """Test ROSTER_SLOTS constant has correct values."""
    assert ROSTER_SLOTS["QB"] == 2
    assert ROSTER_SLOTS["RB"] == 2
    assert ROSTER_SLOTS["WR"] == 3
    assert ROSTER_SLOTS["TE"] == 1
    assert ROSTER_SLOTS["FLEX"] == 2


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

    # Test total rounds calculation
    total_rounds = get_total_rounds()
    assert total_rounds == sum(ROSTER_SLOTS.values())

    # Test total picks calculation
    total_picks = get_total_picks(10)
    assert total_picks == 100

    # Test with custom number of teams
    assert get_total_picks(12) == 12 * total_rounds

    # Test snake order generation
    snake_order = generate_snake_order(4)
    expected_length = 4 * total_rounds
    assert len(snake_order) == expected_length

    # Test first few picks are in correct snake pattern
    assert snake_order[0] == 0  # First pick: team 0
    assert snake_order[1] == 1  # Second pick: team 1
    assert snake_order[2] == 2  # Third pick: team 2
    assert snake_order[3] == 3  # Fourth pick: team 3
    # Snake back
    assert snake_order[4] == 3  # Fifth pick: team 3 (reverse order)
    assert snake_order[5] == 2  # Sixth pick: team 2


def test_optimization_config() -> None:
    """Test OptimizationConfig dataclass with algorithm settings."""
    config = OptimizationConfig(
        learning_rate=0.5,
        convergence_threshold=0.25,
        consecutive_iterations=3,
        max_iterations=50,
    )

    assert config.learning_rate == 0.5
    assert config.convergence_threshold == 0.25
    assert config.consecutive_iterations == 3
    assert config.max_iterations == 50


def test_optimization_config_defaults() -> None:
    """Test OptimizationConfig uses correct defaults."""
    config = OptimizationConfig()

    assert config.learning_rate == 0.5
    assert config.convergence_threshold == 0.25
    assert config.consecutive_iterations == 3
    assert config.max_iterations == 50
