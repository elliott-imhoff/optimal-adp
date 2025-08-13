# Problem Statement: Fantasy Football ADP Optimizer

## 1. Objective
Optimize the Average Draft Position (ADP) rankings for the 2024 fantasy football draft by using the 2024 season data to simulate drafts under my league settings, compute regret scores for each pick, adjust ADP based on regret, and repeat until convergence. The goal is to compute the real value of players across positions for my league settings.

---

## 2. Inputs
- **Primary dataset:** `data/2024_player_stats.csv`
- **Format:** One row per player-season with columns:
  - `Player` — player name
  - `Pos` — position (`QB`, `RB`, `WR`, `TE`)
  - `AVG` — average weekly fantasy points
  - `TTL` — total season fantasy points
- **Initial ADP:** Computed using **Value-Based Ranking (VBR)**:
  - Baseline positions for replacement level:
    - QB: QB21
    - RB: RB29
    - WR: WR43
    - TE: TE11
  - VBR = player’s `AVG` minus the baseline `AVG` for their position.
  - Higher VBR → lower ADP (earlier pick).
- **Filtering:**
  - Exclude `K` and `DEF` positions.
  - Exclude players outside the top 150 by `TTL`.
  - Exclude players with fewer than 10 games played.

---

## 3. League Settings
- **Teams:** 10
- **Draft type:** Snake draft
- **Roster limits (starters only):**
  - QB: 2
  - RB: 2
  - WR: 3
  - TE: 1
  - FLEX: 2 (RB/WR/TE eligible)
- **Bench players:** ignored (not drafted)
- **Total rounds:** 10 rostered starters × 10 teams = 100 picks
- **Pick order:** Fixed per round (snake reversal each round)
- **Draft eligibility:** Player can only be drafted if they fit an unfilled starting slot.

---

## 4. Scoring Rules
Scoring rules are not computed here — all point values are pre-compiled in the dataset (`AVG` and `TTL` columns).
When computing a team’s season score, use the **average weekly points (`AVG`)** for each drafted starter.

---

## 5. Simulation Process

### 5.1 Greedy Draft Algorithm
1. For each pick:
   - Filter remaining players:
     - Not yet drafted
     - Position fits an open starting slot for the current team
   - Select the available player with the **lowest numerical ADP**.
   - Tie-breaker: lowest `Player` name lexicographically.
2. Continue until all starting rosters are full.

### 5.2 Team Scoring
- After the draft, calculate each team’s total season score by:
  - Summing `AVG` across all starters.

---

## 6. Regret Calculation

### 6.1 Definition
For a given pick:
1. Remove that pick from the draft board.
2. Re-simulate the entire draft **from that pick forward** using the greedy algorithm.
3. Compare the final team’s total score in this counterfactual draft to the original draft.
4. Regret score = counterfactual score − actual score.

### 6.2 Constraints
- When updating ADP, enforce **position hierarchy preservation**:
  - Within the same position, a higher-scoring player (`AVG`) must always have a lower (earlier) ADP than any lower-scoring player of that position.
  - Prevents flip-flopping where a worse player at the same position jumps ahead of a better one.

### 6.3 Aggregation
- For each player: compute mean regret across all simulated drafts where they were selected.

---

## 7. ADP Update Process

### 7.1 Update Rule
- New_ADP = Old_ADP − η × normalized_regret_rank
  - η = learning rate (e.g., 0.5)
  - Lower regret → earlier pick (lower ADP number)
  - Higher regret → later pick (higher ADP number)
- Clip ADPs to [1, number_of_players].
- Apply **position hierarchy constraint** after update (see §6.2).

### 7.2 Convergence Criteria
- Stop if:
  - Maximum |ADP change| < ε for M consecutive iterations
    - ε = 0.25 picks
    - M = 3
- Or after K max iterations (default K = 50).

---

## 8. Outputs
- `artifacts/adp_round_k.csv`: ADP list after each iteration k.
- `artifacts/regret_by_pick_round_k.csv`: regret scores for each pick.
- `artifacts/convergence.json`: iteration count, convergence metrics, and parameters used.

---

## 9. Example Iteration
1. Load filtered 2024 player stats and compute initial ADP via VBR.
2. Simulate full greedy draft.
3. Compute regret per pick by re-running draft from each pick onward.
4. Update ADPs with hierarchy constraints.
5. Repeat until convergence.

---

## 10. Assumptions
- Snake draft order is deterministic and repeatable given a seed.
- FLEX slot accepts RB/WR/TE only.
- Bench slots are ignored entirely.

---

## 11. Tests
- Implement common sense unit tests to ensure draft order makes sense e.g. top scoring players are at the top of the draft board, ADP order matches season average scores within same position
