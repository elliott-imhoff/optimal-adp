"""Configuration data structures and settings for optimal ADP calculation."""

from dataclasses import dataclass

# Roster slot configuration - defines how many slots each position has
ROSTER_SLOTS = {
    "QB": 2,  # 2 QB slots
    "RB": 2,  # 2 RB slots
    "WR": 3,  # 3 WR slots
    "TE": 1,  # 1 TE slot
    "FLEX": 2,  # 2 FLEX slots (RB/WR/TE eligible)
}


@dataclass
class Player:
    """Player data structure containing fantasy football statistics.

    Attributes:
        name: Player's full name
        position: Position (QB, RB, WR, TE)
        team: NFL team abbreviation
        avg: Average weekly fantasy points
        total: Total season fantasy points
    """

    name: str
    position: str
    team: str
    avg: float
    total: float


@dataclass
class Team:
    """Team roster structure for tracking drafted players.

    Attributes:
        team_id: Unique team identifier (0-9)
        qb_slots: List of QB player names (ROSTER_SLOTS['QB'] slots)
        rb_slots: List of RB player names (ROSTER_SLOTS['RB'] slots)
        wr_slots: List of WR player names (ROSTER_SLOTS['WR'] slots)
        te_slots: List of TE player names (ROSTER_SLOTS['TE'] slot)
        flex_slots: List of FLEX player names (
            ROSTER_SLOTS['FLEX'] slots, RB/WR/TE eligible
        )
    """

    team_id: int
    qb_slots: list[str | None]
    rb_slots: list[str | None]
    wr_slots: list[str | None]
    te_slots: list[str | None]
    flex_slots: list[str | None]

    def __post_init__(self) -> None:
        """Initialize empty roster slots if not provided."""
        if not self.qb_slots:
            self.qb_slots = [None] * ROSTER_SLOTS["QB"]
        if not self.rb_slots:
            self.rb_slots = [None] * ROSTER_SLOTS["RB"]
        if not self.wr_slots:
            self.wr_slots = [None] * ROSTER_SLOTS["WR"]
        if not self.te_slots:
            self.te_slots = [None] * ROSTER_SLOTS["TE"]
        if not self.flex_slots:
            self.flex_slots = [None] * ROSTER_SLOTS["FLEX"]


@dataclass
class DraftConfig:
    """Configuration for draft league settings.

    Attributes:
        num_teams: Number of teams in league (10)

    Note: Roster slot configuration is defined in ROSTER_SLOTS constant.
    Total draft rounds automatically calculated from sum of ROSTER_SLOTS values.
    """

    num_teams: int = 10

    @property
    def total_rounds(self) -> int:
        """Calculate total draft rounds from roster slots."""
        return sum(ROSTER_SLOTS.values())

    @property
    def total_picks(self) -> int:
        """Calculate total picks in draft."""
        return self.num_teams * self.total_rounds


@dataclass
class OptimizationConfig:
    """Configuration for ADP optimization algorithm.

    Attributes:
        learning_rate: Learning rate η for ADP updates (0.5)
        convergence_threshold: Threshold ε for convergence detection (0.25)
        consecutive_iterations: Required consecutive stable iterations M (3)
        max_iterations: Maximum optimization iterations K (50)
    """

    learning_rate: float = 0.5
    convergence_threshold: float = 0.25
    consecutive_iterations: int = 3
    max_iterations: int = 50


# Baseline positions for VBR calculation
BASELINE_POSITIONS = {
    "QB": 21,  # QB21
    "RB": 29,  # RB29
    "WR": 43,  # WR43
    "TE": 11,  # TE11
}

# Positions eligible for FLEX slots
FLEX_POSITIONS = {"RB", "WR", "TE"}
