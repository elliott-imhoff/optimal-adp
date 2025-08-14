"""Configuration data structures and settings for optimal ADP calculation."""

# Roster slot configuration - defines how many slots each position has
ROSTER_SLOTS = {
    "QB": 2,  # 2 QB slots
    "RB": 2,  # 2 RB slots
    "WR": 3,  # 3 WR slots
    "TE": 1,  # 1 TE slot
    "FLEX": 2,  # 2 FLEX slots (RB/WR/TE eligible)
}

# Baseline positions for VBR calculation
BASELINE_POSITIONS = {
    "QB": 21,  # QB21
    "RB": 29,  # RB29
    "WR": 43,  # WR43
    "TE": 11,  # TE11
}

# Positions eligible for FLEX slots
FLEX_POSITIONS = {"RB", "WR", "TE"}

# Global draft configuration
NUM_TEAMS = 10  # Number of teams in the league
