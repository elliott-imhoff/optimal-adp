"""Draft simulation engine for optimal ADP calculation."""

import copy
import logging
from dataclasses import dataclass

from .config import (
    ROSTER_SLOTS,
    NUM_TEAMS,
    FLEX_POSITIONS,
)

logger = logging.getLogger(__name__)


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


class DraftBoard:
    """Manages available players and draft state during simulation.

    Attributes:
        available_players: List of players sorted by ADP (lowest first)
        drafted_players: Set of player names who have been drafted
        adp_mapping: Dictionary mapping player names to their ADP values
    """

    def __init__(self, players: list[Player], adp_mapping: dict[str, float]):
        """Initialize draft board with players sorted by ADP.

        Args:
            players: List of all eligible players
            adp_mapping: Dictionary of player_name -> ADP value
        """
        self.adp_mapping = adp_mapping.copy()
        self.drafted_players: set[str] = set()

        # Sort players by ADP (lowest ADP = highest priority)
        self.available_players = sorted(
            players, key=lambda p: adp_mapping.get(p.name, float("inf"))
        )

    def get_eligible_players(self, team: Team) -> list[Player]:
        """Get list of available players that can fill team's roster needs.

        Args:
            team: Team to check roster needs for

        Returns:
            List of available players that can be drafted by team
        """
        eligible = []
        for player in self.available_players:
            if player.name not in self.drafted_players:
                if team.can_draft_player(player):
                    eligible.append(player)
        return eligible

    def draft_player(self, player: Player) -> None:
        """Remove player from available pool (mark as drafted).

        Args:
            player: Player who was drafted
        """
        self.drafted_players.add(player.name)
        logger.debug(f"Drafted {player.name} ({player.position})")


class DraftState:
    """Complete draft simulation state including teams and draft board.

    Attributes:
        teams: List of all 10 team rosters
        draft_board: Current state of available players
        pick_order: Snake draft order (team indices)
        current_pick: Current pick number (0-99)
        draft_history: Log of all picks made [(pick_num, player), ...]
    """

    def __init__(
        self,
        players: list[Player],
        adp_mapping: dict[str, float],
        num_teams: int = NUM_TEAMS,
    ):
        """Initialize draft state with empty teams and full player pool.

        Args:
            players: List of all eligible players
            adp_mapping: Dictionary of player_name -> ADP value
            num_teams: Number of teams in the draft (defaults to NUM_TEAMS)
        """
        self.num_teams = num_teams
        self.total_picks = sum(ROSTER_SLOTS.values()) * num_teams
        self.teams = [
            Team(
                team_id=i,
                qb_slots=[None] * ROSTER_SLOTS["QB"],
                rb_slots=[None] * ROSTER_SLOTS["RB"],
                wr_slots=[None] * ROSTER_SLOTS["WR"],
                te_slots=[None] * ROSTER_SLOTS["TE"],
                flex_slots=[None] * ROSTER_SLOTS["FLEX"],
            )
            for i in range(num_teams)
        ]

        self.draft_board = DraftBoard(players, adp_mapping)
        self.pick_order = self.generate_snake_order()
        self.current_pick = 0
        self.draft_history: list[tuple[int, Player]] = []

    def generate_snake_order(self) -> list[int]:
        """Generate snake draft order for a given number of teams.

        Args:
            num_teams: Number of teams in the draft

        Returns:
            List of team indices representing snake draft order

        Example:
            For 4 teams, 3 rounds:
            Round 1: [0, 1, 2, 3]
            Round 2: [3, 2, 1, 0]
            Round 3: [0, 1, 2, 3]
            Result: [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]
        """
        total_rounds = sum(ROSTER_SLOTS.values())
        pick_order: list[int] = []

        for round_num in range(total_rounds):
            if round_num % 2 == 0:  # Even rounds: normal order
                pick_order.extend(range(self.num_teams))
            else:  # Odd rounds: reverse order
                pick_order.extend(range(self.num_teams - 1, -1, -1))

        return pick_order

    def clone(self) -> "DraftState":
        """Create deep copy of draft state for counterfactual analysis.

        Returns:
            Complete copy of current draft state
        """
        return copy.deepcopy(self)

    def rewind_to_pick(self, pick_number: int) -> "DraftState":
        """Restore draft state to before specified pick was made.

        Args:
            pick_number: Pick number to rewind to (0-based)

        Returns:
            New DraftState with state restored to before pick_number
        """
        if pick_number < 0 or pick_number >= len(self.draft_history):
            raise ValueError(f"Invalid pick number: {pick_number}")

        # Create a new draft state from scratch
        original_players = self.draft_board.available_players.copy()
        for drafted_name in self.draft_board.drafted_players:
            # Add back drafted players
            for player in original_players:
                if player.name == drafted_name:
                    original_players.append(player)
                    break

        rewound_state = DraftState(
            original_players, self.draft_board.adp_mapping, self.num_teams
        )

        # Replay picks up to (but not including) the target pick
        for i in range(pick_number):
            if i < len(self.draft_history):
                _, player = self.draft_history[i]
                team_idx = rewound_state.pick_order[i]
                team = rewound_state.teams[team_idx]

                # Add player to team
                team.add_player(player)

                # Update draft board
                rewound_state.draft_board.draft_player(player)
                rewound_state.draft_history.append((i, player))
                rewound_state.current_pick = i + 1

        return rewound_state

    def make_greedy_pick(self) -> Player:
        """Make greedy pick for current team (lowest ADP eligible player).

        Returns:
            Player that was drafted

        Raises:
            ValueError: If no eligible players available for current team
        """
        current_team_idx = self.pick_order[self.current_pick]
        current_team = self.teams[current_team_idx]

        # Get eligible players for current team
        eligible_players = self.draft_board.get_eligible_players(current_team)

        if not eligible_players:
            raise ValueError(f"No eligible players for team {current_team_idx}")

        # Pick lowest ADP player (first in list since sorted by ADP)
        # Tie-breaker: highest average score
        selected_player = min(
            eligible_players,
            key=lambda p: (
                self.draft_board.adp_mapping.get(p.name, float("inf")),
                -p.avg,  # Negative for descending order
            ),
        )

        # Update draft state
        current_team.add_player(selected_player)
        self.draft_board.draft_player(selected_player)
        self.draft_history.append((self.current_pick, selected_player))
        self.current_pick += 1

        logger.debug(
            f"Pick {self.current_pick}: Team {current_team_idx} "
            f"drafts {selected_player.name} ({selected_player.position})"
        )

        return selected_player

    def simulate_full_draft(self) -> "DraftState":
        """Simulate complete snake draft using greedy algorithm.

        Returns:
            Self reference after completing the draft
        """
        logger.info("Starting draft simulation...")

        # Execute all picks
        for pick_num in range(self.total_picks):
            try:
                self.make_greedy_pick()
            except ValueError as e:
                logger.warning(f"Could not complete pick {pick_num}: {e}")
                break

        logger.info(f"Draft simulation complete: {len(self.draft_history)} picks made")
        return self

    def simulate_from_pick(self, start_pick: int) -> "DraftState":
        """Continue draft simulation from given pick number.

        Args:
            start_pick: Pick number to start simulation from

        Returns:
            Self reference after continuing simulation from start_pick
        """
        if start_pick < self.current_pick:
            raise ValueError(
                f"Cannot simulate from pick {start_pick}, "
                f"already at pick {self.current_pick}"
            )

        # Update current pick to start_pick
        self.current_pick = start_pick

        # Continue simulation from current state
        while self.current_pick < self.total_picks:
            try:
                self.make_greedy_pick()
            except ValueError as e:
                logger.warning(f"Could not complete pick {self.current_pick}: {e}")
                break

        return self


# Convenience function for backward compatibility
def simulate_full_draft(
    players: list[Player],
    initial_adp: dict[str, float],
    num_teams: int = NUM_TEAMS,
) -> DraftState:
    """Simulate complete snake draft using greedy algorithm.

    Args:
        players: List of all eligible players
        initial_adp: Dictionary of player_name -> ADP value
        num_teams: Number of teams in the draft (defaults to NUM_TEAMS)

    Returns:
        Final draft state with all rosters filled
    """
    draft_state = DraftState(players, initial_adp, num_teams)
    return draft_state.simulate_full_draft()
