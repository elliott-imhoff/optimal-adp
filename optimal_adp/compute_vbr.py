import csv
from collections import defaultdict

INPUT_CSV = "data/2024_stats.csv"
OUTPUT_CSV = "data/2024_init_vbr.csv"

NAME_COL = "Player"
POS_COL = "Pos"
AVG_COL = "AVG"
TOTAL_COL = "TTL"

BASELINE_RANK = {"QB": 21, "RB": 28, "WR": 42, "TE": 11}


with open(INPUT_CSV, "r") as f:
    players = list(csv.DictReader(f))
players = [p for p in players if p[AVG_COL] is not None]
players = sorted(players, key=lambda p: float(p[AVG_COL]), reverse=True)
players = [p for p in players if float(p[TOTAL_COL]) > 50 and p[POS_COL] in BASELINE_RANK]

position_vals = defaultdict(list)
for p in players:
    position_vals[p[POS_COL]].append(float(p[AVG_COL]))

baseline_points = {}
for pos, rank in BASELINE_RANK.items():
    baseline_points[pos] = position_vals[pos][rank]

for p in players:
    p["VBR"] = float(float(p[AVG_COL]) - baseline_points[p[POS_COL]])

players = sorted(players, key=lambda p: p["VBR"], reverse=True)

fieldnames = list(players[0].keys())
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(players)
