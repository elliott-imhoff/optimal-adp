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
- [x] **`load_player_data()`**: Load and filter 2024 stats from CSV
  - Filter out K and DEF positions
  - Exclude players outside top 150 by TTL
  - Exclude players with <10 games played
  - Return typed player data structure
- [x] **`compute_initial_adp()`**: Calculate VBR-based initial ADP
  - Use baseline positions: QB21, RB29, WR43, TE11
  - VBR = player AVG - baseline AVG for position
  - Sort by VBR descending to get ADP order
- [x] **`save_adp_results()`**: Save ADP iteration results to CSV
- [x] **`save_regret_results()`**: Save regret scores to CSV

### 1.2 Configuration Module (`optimal_adp/config.py`)
- [x] **`DraftConfig`** dataclass: League settings
  - Teams: 10
  - Roster limits (QB:2, RB:2, WR:3, TE:1, FLEX:2)
  - Total picks: 100
- [x] **`OptimizationConfig`** dataclass: Algorithm settings
  - Learning rate η = 0.5
  - Convergence threshold ε = 0.25
  - Consecutive iterations M = 3
  - Max iterations K = 50
- [x] **`Player`** dataclass: Player data structure
- [x] **`Team`** dataclass: Team roster tracking

### 1.3 Tests for Phase 1
- [x] Test data loading and filtering
- [x] Test VBR calculation accuracy
- [x] Test I/O functions with fixture data

### 1.4 Definition of Done
✅ **Phase 1 Complete When:**
- [x] All Phase 1.3 tests pass
- [x] >90% test coverage for Phase 1 modules
- [x] Pre-commit hooks pass (black, flake8, mypy)
- [x] Data structures properly typed with docstrings
- [x] Can load, filter, and compute initial ADP from real 2024 data
- [x] Can save/load configuration and results to/from files

## Phase 2: Draft Simulation Engine

### 2.1 Draft State Management (`optimal_adp/draft_simulator.py`)

**Core Data Structures:**
- [x] **`DraftBoard`** class: Manages available players and ADP ordering
  - `available_players: List[Player]` - sorted by current ADP
  - `drafted_players: Set[str]` - track who's been picked
  - `get_eligible_players(team: Team) -> List[Player]` - filter by roster needs
  - `draft_player(player: Player) -> None` - remove from available pool
  - `reset_from_pick(pick_number: int) -> DraftBoard` - restore state for regret calculation

- [x] **`DraftState`** class: Complete draft simulation state
  - `teams: List[Team]` - all 10 teams with current rosters
  - `draft_board: DraftBoard` - current player availability
  - `pick_order: List[int]` - snake draft order (team indices)
  - `current_pick: int` - which pick we're on (0-99)
  - `draft_history: List[Tuple[int, Player]]` - (pick_num, player) log

**Core Draft Logic:**
- [x] **`generate_snake_order(num_teams: int, num_rounds: int) -> List[int]`**
  - Create deterministic snake draft order
  - Round 1: [0,1,2,...,9], Round 2: [9,8,7,...,0], etc.

- [x] **`can_draft_player(player: Player, team: Team) -> bool`**
  - Check if player position fits team's open roster slots
  - Handle FLEX eligibility (RB/WR/TE can fill FLEX)
  - Account for positional limits (QB:2, RB:2, WR:3, TE:1, FLEX:2)

- [x] **`make_greedy_pick(draft_state: DraftState) -> Player`**
  - Find lowest ADP player that fits current team's needs
  - Tie-breaker: Highest average player score
  - Update draft state (remove player, add to team roster)

- [x] **`simulate_full_draft(players: List[Player], initial_adp: Dict[str, float]) -> DraftState`**
  - Initialize empty teams and draft board
  - Execute 100 picks using greedy algorithm
  - Return final draft state with all rosters filled

### 2.2 Draft Replay & State Restoration
- [x] **`DraftState.clone() -> DraftState`**: Deep copy for counterfactual analysis
- [x] **`DraftState.rewind_to_pick(pick_number: int) -> DraftState`**
  - Restore draft state as it was before specific pick
  - Used for regret calculation "what-if" scenarios
- [x] **`simulate_from_pick(draft_state: DraftState, start_pick: int) -> DraftState`**
  - Continue draft simulation from given pick number
  - Used for counterfactual regret analysis

### 2.3 Team Management & Scoring
- [x] **`Team.add_player(player: Player) -> bool`**: Add to appropriate roster slot
- [x] **`Team.get_open_slots() -> Dict[str, int]`**: Return available positions
- [x] **`Team.is_roster_full() -> bool`**: Check if all starter slots filled
- [x] **`Team.calculate_total_score() -> float`**: Sum AVG across all starters

### 2.4 Tests for Phase 2
- [x] Test snake order generation (10 teams, 10 rounds)
- [x] Test roster eligibility logic (especially FLEX slot handling)
- [x] Test greedy pick selection with various ADP scenarios
- [x] Test draft state cloning and rewinding
- [x] Test complete draft simulation with known player pool
- [x] Test team scoring calculation
- [x] Test draft replay from arbitrary pick numbers

### 2.5 Definition of Done
✅ **Phase 2 Complete When:**
- [x] All Phase 2.4 tests pass
- [x] >90% test coverage for Phase 2 modules
- [x] Pre-commit hooks pass (black, flake8, mypy)
- [x] Can simulate complete 100-pick snake draft
- [x] Draft state can be cloned and rewound to any pick
- [x] Team rosters properly enforce position limits and FLEX rules
- [x] Greedy algorithm consistently picks lowest ADP eligible player

## Phase 3: Regret Calculation & ADP Updates (Combined) ✅

### 3.1 Regret Calculation (`optimal_adp/regret.py`)
- [x] **`calculate_pick_regret(original_draft: DraftState, pick_number: int) -> float`**
  - Clone draft state and rewind to before the specified pick
  - Remove the originally drafted player from available pool
  - Simulate draft forward from that pick using greedy algorithm (will pick different position than original)
  - Compare team scores: regret = counterfactual_score - original_score
  - **Position hierarchy constraint**: If a player was drafted when a same-position player with higher AVG was still available on the draft board, assign high regret score to prevent worse players being picked before better ones

- [x] **`calculate_all_regrets(draft_state: DraftState) -> Dict[str, float]`**
  - Compute regret for every pick in the draft (picks 0-99)
  - Return mapping of player_name -> regret_score (since each player drafted only once)

### 3.2 ADP Update Logic
- [x] **`update_adp_from_regret(current_adp: Dict[str, float], player_regrets: Dict[str, float], learning_rate: float) -> Dict[str, float]`**
  - Apply: New_ADP = Old_ADP + η × regret_score (use raw regret directly)
  - Higher regret → later pick (higher ADP number)
  - No normalization - let raw regret magnitudes drive the changes

- [x] **`rescale_adp_to_picks(updated_adp: Dict[str, float]) -> Dict[str, float]`**
  - Sort all players by updated ADP values (lower = better)
  - Assign ADP as their rank position: 1st best gets ADP=1, 2nd gets ADP=2, etc.
  - Maintains relative ordering for all players, including those beyond pick 100
  - Example: 200 players total → ADPs from 1 to 200, even though only top 100 are draftable

- [x] **`validate_position_hierarchy(updated_adp: Dict[str, float], players: List[Player]) -> bool`**
  - Validate that within each position, higher AVG players still have lower ADP
  - Return True if hierarchy is maintained, False if violated
  - Use for debugging/validation only - hierarchy should be enforced in regret calculation

- [x] **`check_convergence(initial_adp: Dict[str, float], final_adp: Dict[str, float]) -> int`**
  - Calculate total magnitude of position changes across all players between initial (pre-update) and final (post-regret-update + rescaling) ADP
  - Return position_changes_count where 0 means converged
  - Example: Player moving from rank 21→23 contributes 2, player 5→5 contributes 0
  - Log the position_changes_count each iteration to track convergence progress
  - Optimization is complete when position_changes_count reaches 0 (or max_iterations reached)

### 3.3 Tests for Phase 3
- [x] Test regret calculation with simple 2-team, 4-pick scenarios
- [x] Test counterfactual simulation logic
- [x] Test ADP update calculations with raw regret inputs (no normalization)
- [x] Test ADP rescaling maintains relative order for all players (including undrafted)
- [x] Test that undrafted players still have meaningful relative rankings
- [x] Test position hierarchy validation function
- [x] Test convergence detection by counting position changes between initial and final ADP rankings
- [x] Test that regret calculation assigns max regret when a player is drafted while a better same-position player (higher AVG) was still available
- [x] Test that raw regret scores translate appropriately to ADP changes with different learning rates

### 3.4 Definition of Done
✅ **Phase 3 Complete When:**
- [x] All Phase 3.3 tests pass
- [x] >90% test coverage for Phase 3 modules
- [x] Pre-commit hooks pass (black, flake8, mypy)
- [x] Can calculate regret for any pick in a draft
- [x] Position hierarchy constraints prevent same-position inversions
- [x] ADP updates properly scale to meaningful pick positions
- [x] Convergence detection works with magnitude-based position change counting

## Phase 4: Main Optimization Loop

### 4.1 Main Optimizer (`optimal_adp/optimizer.py`)
- [x] **`optimize_adp()`**: Main iteration loop
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
- [x] Command-line interface for running optimization
- [x] Argument parsing for configuration overrides
- [x] Progress reporting and logging

### 4.3 Tests for Phase 4
- [x] Integration test with small fixture dataset
- [x] Test full optimization loop convergence
- [x] Test output file generation

### 4.4 Validation Module (`optimal_adp/validation.py`)
- [x] **Main Validation Functions** accessible via CLI:
  
  **Primary Validation:**
  - [x] `validate_convergence()`: Test convergence with configurable learning rate (default 0.1)
  - [x] `validate_final_rankings()`: Check final rankings quality
  - [x] `validate_optimization()`: Main entry point combining all validations
  - [x] Track convergence metrics (position changes per iteration)
  - [x] Ensure no divergence or oscillation patterns
  - [x] **Optional perturbation testing**: parameter to randomly vary initial VBR rankings slightly
  
  **Final Rankings Quality Checks:**
  - [x] **Position hierarchy validation**: Same-position players ranked by AVG score (no inversions)
  - [x] **Elite positional players validation**: Top QB, top RB, and top WR all drafted in first round
  - [x] **No catastrophic inversions**: No clearly inferior players drafted before vastly superior ones
  
  **CLI Integration:**
  - [x] Add `validate` subcommand to main CLI
  - [x] Support `--perturb` flag for perturbation testing
  - [x] Support `--learning-rate` parameter for validation runs
  - [x] Simple pass/fail reporting with summary
  
  **Tests for Validation Module:**
  - [x] Test validation functions with known good/bad optimization results
  - [x] Test perturbation logic produces varied initial conditions
  - [x] Test position hierarchy validation catches violations
  - [x] Test elite player validation works correctly
  - [x] Test CLI integration for validation subcommand

  **Success Criteria:**
  - [x] Validation module passes all checks with default parameters
  - [x] Convergence achieved within 1000 iterations
  - [x] No position hierarchy violations in final rankings
  - [x] Top QB, RB, and WR all land in first round (picks 1-10)

### 4.5 Definition of Done
✅ **Phase 4 Complete When:**
- [x] All Phase 4.3 tests pass
- [x] >90% test coverage for Phase 4 modules
- [x] Pre-commit hooks pass (black, flake8, mypy)
- [x] Full optimization loop runs end-to-end with real data
- [x] CLI interface works with argument parsing
- [x] **Validation module (4.4) passes all convergence and ranking quality checks**
- [x] Progress reporting and logging functional

## Phase 5: Code Refactoring & Organization ✅

### 5.1 Module Restructuring
- [x] **Consolidate models**: Move `Player`, `Team`, `DraftBoard`, `DraftState` from config.py to new models.py
- [x] **Eliminate draft_simulator.py**: Move all draft logic into models.py for better OOP design
- [x] **Enhance data_io.py**: Add comprehensive artifact saving functions (create_run_directory, save_final_adp_csv, save_team_scores_csv, etc.)
- [x] **Clean up imports**: Update all modules to import from new structure

### 5.2 Object-Oriented Refactoring  
- [x] **DraftState methods**: Convert standalone functions to instance methods
  - `generate_snake_order()` → `DraftState.generate_snake_order()`
  - `simulate_from_pick()` → `DraftState.simulate_from_pick()`
- [x] **Method integration**: Ensure methods use self properties correctly (self.num_teams, etc.)
- [x] **Remove helper functions**: Remove standalone `update_adp_from_regret()` in favor of constrained version

### 5.3 Architectural Separation of Concerns
- [x] **Pure optimization function**: Refactor `optimize_adp()` to accept prepared data (players, initial_adp) and return only algorithmic results
- [x] **I/O coordination function**: Create `run_optimization_with_validation_and_io()` that handles:
  - **Step 1**: Load and prepare data (centralized I/O via data_io functions)
  - **Step 2**: Run pure optimization algorithm  
  - **Step 3**: Run validation using simplified validation functions
  - **Step 4**: Save artifacts if requested (centralized I/O)
  - **Step 5**: Print results and return boolean success/failure
- [x] **Function migration**: Move `run_optimization_with_validation_and_io()` from process.py to optimizer.py
- [x] **Validation refactoring**: Split validation into focused helper functions (validate_convergence_criteria, validate_position_hierarchy_results, validate_elite_players_placement)

### 5.4 Test File Consolidation
- [x] **Merge test files**: Combine test_process.py into test_optimizer.py (17 tests total)
- [x] **Clean imports**: Resolve duplicate imports from file mergers  
- [x] **Update test references**: Fix all tests to use new function signatures (optimize_adp now requires players + initial_adp params)
- [x] **Remove obsolete tests**: Delete test_config.py and test_draft_simulator.py after migration
- [x] **Add comprehensive I/O tests**: Test artifact generation, parameter validation, exception handling

### 5.5 Code Quality Improvements
- [x] **Pre-commit compliance**: Ensure all refactored code passes black, flake8, mypy
- [x] **Import organization**: Clean up circular imports and unused imports
- [x] **Type annotations**: Maintain proper typing through refactoring (Mock instead of MagicMock, etc.)
- [x] **Documentation**: Update docstrings for new module organization and function separation
- [x] **Method patching**: Update test mocks to use patch.object() for instance methods

### 5.6 Definition of Done
✅ **Phase 5 Complete When:**
- [x] All 99 tests pass with >90% coverage after refactoring
- [x] Pre-commit hooks pass on all refactored code  
- [x] Module structure follows better OOP principles with clear separation of concerns
- [x] No circular imports or dependency issues
- [x] Test files properly organized and consolidated
- [x] Pure optimization algorithm separated from I/O operations
- [x] Comprehensive artifact generation and validation pipeline established
- [x] Code organization improved for maintainability and testability

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

## Notes
- Follow test-first development: write tests for each module before implementation
- Use small, pure functions with no global state
- Pass configuration explicitly through function parameters
- Use fixed RNG seeds for reproducible results
- Keep modules separated by concern (I/O, scoring, simulation, regret, updates)
