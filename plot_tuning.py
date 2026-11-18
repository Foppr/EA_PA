import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WINNER_COLOR = "#2a78d6"   # blue: the winning configuration
OTHER_COLOR = "#b4b2a9"    # gray: all other configurations
GRID_COLOR = "#e5e4df"
TEXT_COLOR = "#52514e"

# --- Load data ---
with open("tuning_results.csv", newline="") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["round"] = int(r["round"])
    r["run_budget"] = int(r["run_budget"])
    r["F18"] = float(r["F18"])
    r["F23"] = float(r["F23"])
    r["kept"] = r["kept"] == "True"


def config_key(r):
    return (r["population_size"], r["mutation_rate"], r["crossover_rate"], r["n_elites"])


# Group the rows per configuration: one curve per config, ordered by round
curves = defaultdict(list)
for r in rows:
    curves[config_key(r)].append(r)
for pts in curves.values():
    pts.sort(key=lambda r: r["round"])

last_round = max(r["round"] for r in rows)
winner = next(config_key(r) for r in rows if r["round"] == last_round and r["kept"])

# Number of configs and budget per round (for the annotations)
rounds = sorted({r["round"] for r in rows})
budgets = [next(r["run_budget"] for r in rows if r["round"] == k) for k in rounds]
n_configs = [sum(r["round"] == k for r in rows) for k in rounds]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, prob, title in zip(axes, ["F18", "F23"], ["F18 (LABS)", "F23 (N-Queens)"]):
    # Draw the gray curves first, the winner last so it is on top
    for key, pts in sorted(curves.items(), key=lambda kv: kv[0] == winner):
        xs = [p["run_budget"] for p in pts]
        ys = [p[prob] for p in pts]
        is_winner = key == winner
        ax.plot(xs, ys,
                color=WINNER_COLOR if is_winner else OTHER_COLOR,
                linewidth=2.5 if is_winner else 1,
                marker="o", markersize=6 if is_winner else 4,
                zorder=3 if is_winner else 2)

    # Mark each round with a vertical line and the number of configs
    for b, n in zip(budgets, n_configs):
        ax.axvline(b, color=GRID_COLOR, linewidth=1, zorder=1)
        ax.annotate(f"{n} configs", (b, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Evaluations per run", fontsize=9)
    ax.set_ylabel("Best fitness (mean over runs)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=18)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

handles = [
    Line2D([], [], color=WINNER_COLOR, linewidth=2.5, marker="o", label="Selected configuration"),
    Line2D([], [], color=OTHER_COLOR, linewidth=1, marker="o", label="Other configurations"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 1])

fig.savefig("successive_halving_curves.pdf")
fig.savefig("successive_halving_curves.png", dpi=200)
plt.show()
