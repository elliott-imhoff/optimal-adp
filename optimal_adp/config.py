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
        qb_slots: List of QB players (ROSTER_SLOTS['QB'] slots)
        rb_slots: List of RB players (ROSTER_SLOTS['RB'] slots)
        wr_slots: List of WR players (ROSTER_SLOTS['WR'] slots)
        te_slots: List of TE players (ROSTER_SLOTS['TE'] slot)
        flex_slots: List of FLEX players (
            ROSTER_SLOTS['FLEX'] slots, RB/WR/TE eligible
        )
    """

    team_id: int
    qb_slots: list["Player | None"]
    rb_slots: list["Player | None"]
    wr_slots: list["Player | None"]
    te_slots: list["Player | None"]
    flex_slots: list["Player | None"]

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

    def add_player(self, player: "Player") -> bool:
        """Add player to appropriate roster slot.

        Args:
            player: Player to add to roster

        Returns:
            True if player was successfully added, False otherwise
        """
        position = player.position

        # Try position-specific slots first
        if position == "QB" and None in self.qb_slots:
            idx = self.qb_slots.index(None)
            self.qb_slots[idx] = player
            return True
        elif position == "RB" and None in self.rb_slots:
            idx = self.rb_slots.index(None)
            self.rb_slots[idx] = player
            return True
        elif position == "WR" and None in self.wr_slots:
            idx = self.wr_slots.index(None)
            self.wr_slots[idx] = player
            return True
        elif position == "TE" and None in self.te_slots:
            idx = self.te_slots.index(None)
            self.te_slots[idx] = player
            return True

        # Try FLEX slot for eligible positions
        if position in FLEX_POSITIONS and None in self.flex_slots:
            idx = self.flex_slots.index(None)
            self.flex_slots[idx] = player
            return True

        return False

    def get_open_slots(self) -> dict[str, int]:
        """Get count of available roster slots by position.

        Returns:
            Dictionary mapping position -> number of open slots
        """
        return {
            "QB": self.qb_slots.count(None),
            "RB": self.rb_slots.count(None),
            "WR": self.wr_slots.count(None),
            "TE": self.te_slots.count(None),
            "FLEX": self.flex_slots.count(None),
        }

    def is_roster_full(self) -> bool:
        """Check if all starter slots are filled.

        Returns:
            True if no open slots remain, False otherwise
        """
        open_slots = self.get_open_slots()
        return all(count == 0 for count in open_slots.values())

    def calculate_total_score(self) -> float:
        """Calculate total team score from all starters.

        Returns:
            Sum of AVG scores across all drafted players
        """
        total_score = 0.0

        # Sum scores from all roster slots
        all_players: list["Player"] = []
        all_players.extend(player for player in self.qb_slots if player is not None)
        all_players.extend(player for player in self.rb_slots if player is not None)
        all_players.extend(player for player in self.wr_slots if player is not None)
        all_players.extend(player for player in self.te_slots if player is not None)
        all_players.extend(player for player in self.flex_slots if player is not None)

        for player in all_players:
            total_score += player.avg

        return total_score

    def can_draft_player(self, player: "Player") -> bool:
        """Check if a player can be drafted to fill this team's roster needs.

        Args:
            player: Player to check eligibility for

        Returns:
            True if player can be added to team roster, False otherwise

        Logic:
            - QB can only fill QB slots
            - RB can fill RB slots or FLEX slots
            - WR can fill WR slots or FLEX slots
            - TE can fill TE slots or FLEX slots
        """
        position = player.position

        # Check position-specific slots first
        if position == "QB" and None in self.qb_slots:
            return True
        elif position == "RB" and None in self.rb_slots:
            return True
        elif position == "WR" and None in self.wr_slots:
            return True
        elif position == "TE" and None in self.te_slots:
            return True

        # Check FLEX slots for eligible positions
        if position in FLEX_POSITIONS and None in self.flex_slots:
            return True

        return False


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

    def generate_snake_order(self) -> list[int]:
        """Generate snake draft order for this draft configuration.

        Returns:
            List of team indices representing snake draft order

        Example:
            For 4 teams, 3 rounds:
            Round 1: [0, 1, 2, 3]
            Round 2: [3, 2, 1, 0]
            Round 3: [0, 1, 2, 3]
            Result: [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]
        """
        pick_order: list[int] = []

        for round_num in range(self.total_rounds):
            if round_num % 2 == 0:  # Even rounds: normal order
                pick_order.extend(range(self.num_teams))
            else:  # Odd rounds: reverse order
                pick_order.extend(range(self.num_teams - 1, -1, -1))

        return pick_order


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
