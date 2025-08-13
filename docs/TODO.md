# TODO: Fantasy Football ADP Optimizer Implementation

Based on the problem statement and coding guidelines, here's the comprehensive implementation plan:

## Phase 0: Repository Setup & Development Environment

### 0.1 Pre-commit & Code Quality Setup
- [x] **Configure pre-commit hooks** with black, flake8, mypy
- [x] **Setup test coverage** reporting with pytest-cov
- [x] **Configure pyproject.toml** with tool settings for black, flake8, mypy
- [x] **Create .pre-commit-config.yaml** with appropriate hooks
- [x] **Initialize test directory structure** (`tests/` with `__init__.py`)

### 0.2 Definition of Done
✅ **Phase 0 Complete When:**
- [x] Pre-commit hooks installed and running on commit
- [x] Black, flake8, mypy all pass on existing code
- [x] Test coverage reporting configured
- [x] Development environment ready for TDD workflow

## Phase 1: Data Processing & Setup

### 1.1 Data I/O Module (`optimal_adp/data_io.py`)
- [ ] **`load_player_data()`**: Load and filter 2024 stats from CSV
  - Filter out K and DEF positions
  - Exclude players outside top 150 by TTL
  - Exclude players with <10 games played
  - Return typed player data structure
- [ ] **`compute_initial_adp()`**: Calculate VBR-based initial ADP
  - Use baseline positions: QB21, RB29, WR43, TE11
  - VBR = player AVG - baseline AVG for position
  - Sort by VBR descending to get ADP order
- [ ] **`save_adp_results()`**: Save ADP iteration results to CSV
- [ ] **`save_regret_results()`**: Save regret scores to CSV
- [ ] **`save_convergence_metrics()`**: Save convergence info to JSON

### 1.2 Configuration Module (`optimal_adp/config.py`)
- [ ] **`DraftConfig`** dataclass: League settings
  - Teams: 10
  - Roster limits (QB:2, RB:2, WR:3, TE:1, FLEX:2)
  - Total picks: 100
- [ ] **`OptimizationConfig`** dataclass: Algorithm settings
  - Learning rate η = 0.5
  - Convergence threshold ε = 0.25
  - Consecutive iterations M = 3
  - Max iterations K = 50
- [ ] **`Player`** dataclass: Player data structure
- [ ] **`Team`** dataclass: Team roster tracking

### 1.3 Tests for Phase 1
- [ ] Test data loading and filtering
- [ ] Test VBR calculation accuracy
- [ ] Test I/O functions with fixture data

### 1.4 Definition of Done
✅ **Phase 1 Complete When:**
- [ ] All Phase 1.3 tests pass
- [ ] >90% test coverage for Phase 1 modules
- [ ] Pre-commit hooks pass (black, flake8, mypy)
- [ ] Data structures properly typed with docstrings
- [ ] Can load, filter, and compute initial ADP from real 2024 data
- [ ] Can save/load configuration and results to/from files

## Phase 2: Draft Simulation Engine

### 2.1 Draft State Management (`optimal_adp/draft_simulator.py`)

**Core Data Structures:**
- [ ] **`DraftBoard`** class: Manages available players and ADP ordering
  - `available_players: List[Player]` - sorted by current ADP
  - `drafted_players: Set[str]` - track who's been picked
  - `get_eligible_players(team: Team) -> List[Player]` - filter by roster needs
  - `draft_player(player: Player) -> None` - remove from available pool
  - `reset_from_pick(pick_number: int) -> DraftBoard` - restore state for regret calculation

- [ ] **`DraftState`** class: Complete draft simulation state
  - `teams: List[Team]` - all 10 teams with current rosters
  - `draft_board: DraftBoard` - current player availability
  - `pick_order: List[int]` - snake draft order (team indices)
  - `current_pick: int` - which pick we're on (0-99)
  - `draft_history: List[Tuple[int, Player]]` - (pick_num, player) log

**Core Draft Logic:**
- [ ] **`generate_snake_order(num_teams: int, num_rounds: int) -> List[int]`**
  - Create deterministic snake draft order
  - Round 1: [0,1,2,...,9], Round 2: [9,8,7,...,0], etc.

- [ ] **`can_draft_player(player: Player, team: Team) -> bool`**
  - Check if player position fits team's open roster slots
  - Handle FLEX eligibility (RB/WR/TE can fill FLEX)
  - Account for positional limits (QB:2, RB:2, WR:3, TE:1, FLEX:2)

- [ ] **`make_greedy_pick(draft_state: DraftState) -> Player`**
  - Find lowest ADP player that fits current team's needs
  - Tie-breaker: Highest average player score
  - Update draft state (remove player, add to team roster)

- [ ] **`simulate_full_draft(players: List[Player], initial_adp: Dict[str, float]) -> DraftState`**
  - Initialize empty teams and draft board
  - Execute 100 picks using greedy algorithm
  - Return final draft state with all rosters filled

### 2.2 Draft Replay & State Restoration
- [ ] **`DraftState.clone() -> DraftState`**: Deep copy for counterfactual analysis
- [ ] **`DraftState.rewind_to_pick(pick_number: int) -> DraftState`**
  - Restore draft state as it was before specific pick
  - Used for regret calculation "what-if" scenarios
- [ ] **`simulate_from_pick(draft_state: DraftState, start_pick: int) -> DraftState`**
  - Continue draft simulation from given pick number
  - Used for counterfactual regret analysis

### 2.3 Team Management & Scoring
- [ ] **`Team.add_player(player: Player) -> bool`**: Add to appropriate roster slot
- [ ] **`Team.get_open_slots() -> Dict[str, int]`**: Return available positions
- [ ] **`Team.is_roster_full() -> bool`**: Check if all starter slots filled
- [ ] **`Team.calculate_total_score() -> float`**: Sum AVG across all starters

### 2.4 Tests for Phase 2
- [ ] Test snake order generation (10 teams, 10 rounds)
- [ ] Test roster eligibility logic (especially FLEX slot handling)
- [ ] Test greedy pick selection with various ADP scenarios
- [ ] Test draft state cloning and rewinding
- [ ] Test complete draft simulation with known player pool
- [ ] Test team scoring calculation
- [ ] Test draft replay from arbitrary pick numbers

### 2.5 Definition of Done
✅ **Phase 2 Complete When:**
- [ ] All Phase 2.4 tests pass
- [ ] >90% test coverage for Phase 2 modules
- [ ] Pre-commit hooks pass (black, flake8, mypy)
- [ ] Can simulate complete 100-pick snake draft
- [ ] Draft state can be cloned and rewound to any pick
- [ ] Team rosters properly enforce position limits and FLEX rules
- [ ] Greedy algorithm consistently picks lowest ADP eligible player

## Phase 3: Regret Calculation & ADP Updates (Combined)

### 3.1 Regret Calculation (`optimal_adp/regret_optimizer.py`)
- [ ] **`calculate_pick_regret(original_draft: DraftState, pick_number: int) -> float`**
  - Clone draft state and rewind to before the specified pick
  - Remove the originally drafted player from available pool
  - Simulate draft forward from that pick using greedy algorithm (will pick different position than original)
  - Compare team scores: regret = counterfactual_score - original_score
  - **Position hierarchy constraint**: If a player was drafted when a same-position player with higher AVG was still available on the draft board, assign high regret score to prevent worse players being picked before better ones

- [ ] **`calculate_all_regrets(draft_state: DraftState) -> Dict[str, float]`**
  - Compute regret for every pick in the draft (picks 0-99)
  - Return mapping of player_name -> regret_score (since each player drafted only once)

### 3.2 ADP Update Logic
- [ ] **`normalize_regret_scores(player_regrets: Dict[str, float]) -> Dict[str, float]`**
  - Convert raw regret scores to normalized ranks (0-1 scale)
  - Lower regret → higher rank → earlier ADP

- [ ] **`update_adp_raw(current_adp: Dict[str, float], normalized_regrets: Dict[str, float], learning_rate: float) -> Dict[str, float]`**
  - Apply: New_ADP = Old_ADP - η × normalized_regret_rank
  - Initial result may be outside valid range

- [ ] **`rescale_adp_to_picks(updated_adp: Dict[str, float]) -> Dict[str, float]`**
  - Sort all players by updated ADP values (lower = better)
  - Assign ADP as their rank position: 1st best gets ADP=1, 2nd gets ADP=2, etc.
  - Maintains relative ordering for all players, including those beyond pick 100
  - Example: 200 players total → ADPs from 1 to 200, even though only top 100 are draftable

- [ ] **`validate_position_hierarchy(updated_adp: Dict[str, float], players: List[Player]) -> bool`**
  - Validate that within each position, higher AVG players still have lower ADP
  - Return True if hierarchy is maintained, False if violated
  - Use for debugging/validation only - hierarchy should be enforced in regret calculation

- [ ] **`check_convergence(adp_history: List[Dict[str, float]], threshold: float = 0.25, consecutive_rounds: int = 3) -> bool`**
  - Check if max |ADP change| < threshold for M consecutive iterations
  - Return True if converged, False otherwise

### 3.3 Tests for Phase 3
- [ ] Test regret calculation with simple 2-team, 4-pick scenarios
- [ ] Test counterfactual simulation logic
- [ ] Test ADP update calculations with known regret inputs
- [ ] Test ADP rescaling maintains relative order for all players (including undrafted)
- [ ] Test that undrafted players still have meaningful relative rankings
- [ ] Test position hierarchy validation function
- [ ] Test convergence detection with various ADP change patterns
- [ ] Test that regret calculation assigns max regret when a player is drafted while a better same-position player (higher AVG) was still available

### 3.4 Definition of Done
✅ **Phase 3 Complete When:**
- [ ] All Phase 3.3 tests pass
- [ ] >90% test coverage for Phase 3 modules
- [ ] Pre-commit hooks pass (black, flake8, mypy)
- [ ] Can calculate regret for any pick in a draft
- [ ] Position hierarchy constraints prevent same-position inversions
- [ ] ADP updates properly scale to meaningful pick positions
- [ ] Convergence detection works with configurable thresholds

## Phase 4: Main Optimization Loop

### 4.1 Main Optimizer (`optimal_adp/optimizer.py`)
- [ ] **`optimize_adp()`**: Main iteration loop
  1. Load and filter player data
  2. Compute initial VBR-based ADP
  3. For each iteration:
     - Simulate draft with current ADP
     - Calculate regret scores for all picks
     - Update ADP with hierarchy constraints
     - Check convergence
     - Save iteration results
  4. Return final ADP and convergence metrics

### 4.2 CLI Runner (`optimal_adp/main.py`)
- [ ] Command-line interface for running optimization
- [ ] Argument parsing for configuration overrides
- [ ] Progress reporting and logging

### 4.3 Tests for Phase 4
- [ ] Integration test with small fixture dataset
- [ ] Test full optimization loop convergence
- [ ] Test output file generation

### 4.4 Definition of Done
✅ **Phase 4 Complete When:**
- [ ] All Phase 4.3 tests pass
- [ ] >90% test coverage for Phase 4 modules
- [ ] Pre-commit hooks pass (black, flake8, mypy)
- [ ] Full optimization loop runs end-to-end with real data
- [ ] CLI interface works with argument parsing
- [ ] Convergence achieved within reasonable iterations
- [ ] Progress reporting and logging functional

## Phase 5: Artifacts & Output

### 5.1 Directory Structure
- [ ] Create `artifacts/` directory
- [ ] Implement artifact file naming conventions:
  - `artifacts/adp_round_k.csv`
  - `artifacts/regret_by_pick_round_k.csv`
  - `artifacts/convergence.json`

### 5.2 Validation & Analysis
- [ ] **Sanity check functions**:
  - Verify top players are at top of draft board
  - Verify ADP order matches AVG scores within positions
  - Check that all roster constraints are satisfied
- [ ] **Analysis utilities** (`optimal_adp/analysis.py`):
  - Compare initial vs. final ADP rankings
  - Identify biggest movers by position
  - Generate summary statistics

### 5.3 Definition of Done
✅ **Phase 5 Complete When:**
- [ ] >90% test coverage for Phase 5 modules
- [ ] Pre-commit hooks pass (black, flake8, mypy)
- [ ] All artifact files generate correctly with proper naming
- [ ] Analysis utilities provide meaningful insights
- [ ] Sanity checks validate optimization results
- [ ] Full dataset optimization produces reasonable ADP changes

## Phase 6: Code Quality & Documentation

### 6.1 Code Quality
- [ ] Add comprehensive docstrings (Python 3.10 style)
- [ ] Type hints for all functions
- [ ] Configure pre-commit hooks
- [ ] Run black, flake8, mypy
- [ ] Achieve >90% test coverage

### 6.2 Documentation
- [ ] Update README.md with usage instructions
- [ ] Add examples of running the optimizer
- [ ] Document configuration options
- [ ] Add troubleshooting guide

### 6.3 Definition of Done
✅ **Phase 6 Complete When:**
- [ ] >90% test coverage achieved
- [ ] All code passes black, flake8, mypy checks
- [ ] Pre-commit hooks configured and working
- [ ] README.md has clear usage instructions with examples
- [ ] All functions have comprehensive docstrings

## Implementation Order Priority

**Follow phases in order, with tests written first for each phase:**

0. **Phase 0**: Repository Setup & Development Environment
   - Set up pre-commit hooks, test coverage, and development tools
   - Establish code quality baseline

1. **Phase 1**: Data Processing & Setup
   - Write Phase 1.3 tests first (using fixture data)
   - Implement Phase 1.2 (Config) - Foundation data structures
   - Implement Phase 1.1 (Data I/O) - Core data loading
   - Validate tests pass and pre-commit hooks pass

2. **Phase 2**: Draft Simulation Engine
   - Write Phase 2.4 tests first (draft logic with known scenarios)
   - Implement Phase 2.1 & 2.2 (Draft Simulation & State Management)
   - Implement Phase 2.3 (Team Management & Scoring)
   - Validate tests pass and pre-commit hooks pass

3. **Phase 3**: Regret Calculation & ADP Updates
   - Write Phase 3.3 tests first (regret scenarios with simple cases)
   - Implement Phase 3.1 (Regret Calculation)
   - Implement Phase 3.2 (ADP Update Logic)
   - Validate tests pass and pre-commit hooks pass

4. **Phase 4**: Main Optimization Loop
   - Write Phase 4.3 tests first (integration with small fixture)
   - Implement Phase 4.1 & 4.2 (Main Optimizer & CLI)
   - Validate tests pass and pre-commit hooks pass

5. **Phase 5**: Artifacts & Output
   - Implement artifact generation and analysis utilities
   - Test with full dataset
   - Validate pre-commit hooks pass

6. **Phase 6**: Code Quality & Documentation
   - Final polish, documentation, and comprehensive testing

## Notes
- Follow test-first development: write tests for each module before implementation
- Use small, pure functions with no global state
- Pass configuration explicitly through function parameters
- Use fixed RNG seeds for reproducible results
- Keep modules separated by concern (I/O, scoring, simulation, regret, updates)
