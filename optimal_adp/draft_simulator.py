"""Draft simulation engine for optimal ADP calculation."""

import copy
import logging

from .config import Player, Team, ROSTER_SLOTS, DraftConfig

logger = logging.getLogger(__name__)


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
        draft_config: DraftConfig | None = None,
    ):
        """Initialize draft state with empty teams and full player pool.

        Args:
            players: List of all eligible players
            adp_mapping: Dictionary of player_name -> ADP value
            draft_config: Draft configuration (defaults to DraftConfig())
        """
        if draft_config is None:
            draft_config = DraftConfig()

        self.draft_config = draft_config
        self.teams = [
            Team(
                team_id=i,
                qb_slots=[None] * ROSTER_SLOTS["QB"],
                rb_slots=[None] * ROSTER_SLOTS["RB"],
                wr_slots=[None] * ROSTER_SLOTS["WR"],
                te_slots=[None] * ROSTER_SLOTS["TE"],
                flex_slots=[None] * ROSTER_SLOTS["FLEX"],
            )
            for i in range(draft_config.num_teams)
        ]

        self.draft_board = DraftBoard(players, adp_mapping)
        self.pick_order = draft_config.generate_snake_order()
        self.current_pick = 0
        self.draft_history: list[tuple[int, Player]] = []

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
            original_players, self.draft_board.adp_mapping, self.draft_config
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


def make_greedy_pick(draft_state: DraftState) -> Player:
    """Make greedy pick for current team (lowest ADP eligible player).

    Args:
        draft_state: Current draft state

    Returns:
        Player that was drafted

    Raises:
        ValueError: If no eligible players available for current team
    """
    current_team_idx = draft_state.pick_order[draft_state.current_pick]
    current_team = draft_state.teams[current_team_idx]

    # Get eligible players for current team
    eligible_players = draft_state.draft_board.get_eligible_players(current_team)

    if not eligible_players:
        raise ValueError(f"No eligible players for team {current_team_idx}")

    # Pick lowest ADP player (first in list since sorted by ADP)
    # Tie-breaker: highest average score
    selected_player = min(
        eligible_players,
        key=lambda p: (
            draft_state.draft_board.adp_mapping.get(p.name, float("inf")),
            -p.avg,  # Negative for descending order
        ),
    )

    # Update draft state
    current_team.add_player(selected_player)
    draft_state.draft_board.draft_player(selected_player)
    draft_state.draft_history.append((draft_state.current_pick, selected_player))
    draft_state.current_pick += 1

    logger.debug(
        f"Pick {draft_state.current_pick}: Team {current_team_idx} "
        f"drafts {selected_player.name} ({selected_player.position})"
    )

    return selected_player


def simulate_full_draft(
    players: list[Player],
    initial_adp: dict[str, float],
    draft_config: DraftConfig | None = None,
) -> DraftState:
    """Simulate complete snake draft using greedy algorithm.

    Args:
        players: List of all eligible players
        initial_adp: Dictionary of player_name -> ADP value
        draft_config: Draft configuration (defaults to DraftConfig())

    Returns:
        Final draft state with all rosters filled
    """
    if draft_config is None:
        draft_config = DraftConfig()

    draft_state = DraftState(players, initial_adp, draft_config)

    logger.info("Starting draft simulation...")

    # Execute all picks
    total_picks = draft_config.total_picks
    for pick_num in range(total_picks):
        try:
            make_greedy_pick(draft_state)
        except ValueError as e:
            logger.warning(f"Could not complete pick {pick_num}: {e}")
            break

    logger.info(
        f"Draft simulation complete: {len(draft_state.draft_history)} picks made"
    )
    return draft_state


def simulate_from_pick(draft_state: DraftState, start_pick: int) -> DraftState:
    """Continue draft simulation from given pick number.

    Args:
        draft_state: Current draft state to continue from
        start_pick: Pick number to start simulation from

    Returns:
        Updated draft state with simulation continued from start_pick
    """
    if start_pick < draft_state.current_pick:
        raise ValueError(
            f"Cannot simulate from pick {start_pick}, "
            f"already at pick {draft_state.current_pick}"
        )

    # Update current pick to start_pick
    draft_state.current_pick = start_pick

    # Continue simulation from current state
    total_picks = draft_state.draft_config.total_picks
    while draft_state.current_pick < total_picks:
        try:
            make_greedy_pick(draft_state)
        except ValueError as e:
            logger.warning(f"Could not complete pick {draft_state.current_pick}: {e}")
            break

    return draft_state
