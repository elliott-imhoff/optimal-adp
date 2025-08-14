# Optimal ADP: Fantasy Football Draft Position Optimizer

A Python tool that optimizes Average Draft Position (ADP) rankings for fantasy football drafts using actual season data, league-specific settings, and regret minimization algorithms.

## 🎯 Project Overview

This tool uses 2024 NFL season data to simulate thousands of fantasy football drafts under your specific league settings, calculates regret scores for each draft pick, and iteratively optimizes ADP rankings until convergence. The result is a data-driven ADP ranking that reflects the true value of players in your league format.

### Key Features

- **Data-Driven Optimization**: Uses actual NFL statistics to determine optimal draft positions
- **League-Specific Customization**: Configurable team count, roster requirements, and scoring settings
- **Regret Minimization**: Advanced algorithm that minimizes draft regret through iterative improvement
- **Value-Based Ranking (VBR)**: Initial ADP calculation based on position-specific baselines
- **Snake Draft Simulation**: Realistic draft simulation with position eligibility constraints
- **Comprehensive Output**: Detailed artifacts including convergence history, team scores, and regret analysis

## 🏈 How It Works

1. **Data Loading**: Load player statistics from 2024 season data
2. **Initial ADP**: Calculate starting ADP using Value-Based Ranking (VBR) with position baselines:
   - QB21, RB29, WR43, TE11 as replacement-level thresholds
3. **Draft Simulation**: Simulate complete snake drafts using greedy ADP-based selection
4. **Regret Calculation**: For each pick, simulate counterfactual drafts to measure regret
5. **ADP Updates**: Adjust ADP values based on regret scores with position hierarchy constraints  
6. **Convergence**: Repeat until ADP changes stabilize or maximum iterations reached

## 📋 Requirements

- Python 3.9-3.11
- Poetry (for dependency management)

## 🚀 Installation

### Option 1: Install as a CLI Tool (Recommended)

```bash
# Clone the repository
git clone https://github.com/elliott-imhoff/optimal-adp.git
cd optimal-adp

# Install with Poetry
poetry install

# The CLI command is now available
poetry run optimal-adp --help
```

### Option 2: Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/elliott-imhoff/optimal-adp.git  
cd optimal-adp

# Install dependencies
poetry install

# Install pre-commit hooks (optional)
poetry run pre-commit install
```

## 💻 Usage

### Basic Usage

```bash
# Run optimization with default settings
poetry run optimal-adp data/2024_stats.csv

# Run with custom parameters
poetry run optimal-adp data/2024_stats.csv \
  --learning-rate 0.5 \
  --max-iterations 50 \
  --num-teams 12 \
  --verbose
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--learning-rate` | Learning rate for ADP updates (0.0-1.0) | 0.3 |
| `--max-iterations` | Maximum optimization iterations | 50 |
| `--num-teams` | Number of teams in draft | 10 |
| `--perturb` | Apply random perturbation to initial ADP | False |
| `--no-artifacts` | Skip saving detailed output files | False |
| `--verbose` | Enable verbose logging | False |

### Examples

```bash
# Quick test run with minimal iterations
poetry run optimal-adp data/2024_stats.csv --max-iterations 5 --verbose

# High-precision optimization for 12-team league  
poetry run optimal-adp data/2024_stats.csv \
  --learning-rate 0.2 \
  --max-iterations 100 \
  --num-teams 12

# Experimental run with perturbation
poetry run optimal-adp data/2024_stats.csv --perturb --verbose
```

## 📊 Output Files

When optimization completes, the following files are generated in `artifacts/run_[timestamp]/`:

### Core Results
- **`final_adp.csv`**: Optimized ADP rankings with draft positions
- **`team_scores.csv`**: Final team scores from optimized draft
- **`regrets.csv`**: Regret scores for each player
- **`initial_vbr_adp.csv`**: Starting ADP values before optimization

### Analysis Files  
- **`convergence_history.csv`**: Position changes per iteration
- **`run_parameters.txt`**: Configuration and run summary

## 🏆 League Settings

The tool is preconfigured for a standard 10-team league with the following roster requirements:

- **Starters**: 2 QB, 2 RB, 3 WR, 1 TE, 2 FLEX (RB/WR/TE)
- **Total Rounds**: 10 (100 total picks)
- **Draft Format**: Snake draft with position eligibility constraints

These settings can be modified in `optimal_adp/config.py`.

## 📈 Data Format

Input CSV should contain player statistics with these columns:

| Column | Description |
|--------|-------------|
| `Player` | Player name |
| `Pos` | Position (QB, RB, WR, TE) |
| `Team` | NFL team |
| `AVG` | Average weekly fantasy points |  
| `TTL` | Total season fantasy points |

## 🔧 Development

### Running Tests

```bash
poetry run pytest
poetry run pytest --cov=optimal_adp  # With coverage
```

### Code Quality

```bash
# Pre-commit hooks (run all checks)
poetry run pre-commit run --all-files
```
