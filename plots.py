import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_ioh_dat(filepath: str, budget: int = 5000) -> np.ndarray:
    """
    Parse an IOHprofiler .dat file containing multiple runs and return a 2D array
    of shape (n_runs, budget) with the best-so-far raw_y at every evaluation 1..budget.
    """
    runs_raw = []
    current_evals, current_ys = [], []

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # New run header line (e.g., "evaluations" "raw_y")
            if stripped.startswith('"') or stripped[0].isalpha():
                if current_evals:
                    runs_raw.append((current_evals, current_ys))
                    current_evals, current_ys = [], []
                continue

            parts = stripped.split()
            current_evals.append(int(float(parts[0])))
            current_ys.append(float(parts[1]))

    if current_evals:
        runs_raw.append((current_evals, current_ys))

    curves = np.full((len(runs_raw), budget), np.nan, dtype=float)
    for run_idx, (evals, ys) in enumerate(runs_raw):
        best_so_far = -np.inf
        prev_eval = 0
        for ev, y in zip(evals, ys):
            ev_clamped = min(max(ev, 1), budget)
            if prev_eval > 0 and prev_eval < ev_clamped:
                curves[run_idx, prev_eval:ev_clamped] = best_so_far
            best_so_far = max(best_so_far, y)
            curves[run_idx, ev_clamped - 1] = best_so_far
            prev_eval = ev_clamped

        # Forward-fill the final best fitness until the end of the budget
        if prev_eval < budget:
            curves[run_idx, prev_eval:budget] = best_so_far

    return curves


def plot_convergence(
    baseline_dat: str,
    tuned_dat: str,
    title: str = "GA Convergence Curve",
    budget: int = 5000,
    save_path: str = "convergence.pdf",
) -> None:
    """Plot mean ± 1 SD convergence curves for baseline vs. tuned .dat files."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    evaluations = np.arange(1, budget + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        ("Baseline", baseline_dat, "#1f77b4"),
        ("Tuned", tuned_dat, "#d95f02"),
    ]

    for label, filepath, color in configs:
        curves = parse_ioh_dat(filepath, budget=budget)
        mean_curve = np.nanmean(curves, axis=0)
        sd_curve = np.nanstd(curves, axis=0, ddof=1)

        ax.plot(evaluations, mean_curve, label=f"{label} (Mean)", color=color, linewidth=2)
        ax.fill_between(
            evaluations,
            mean_curve - sd_curve,
            mean_curve + sd_curve,
            color=color,
            alpha=0.2,
            label=f"{label} (±1 SD)",
        )

    ax.set_title(title)
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best-so-far Fitness ($y$)")
    ax.set_xlim(1, budget)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Example usage for F18 (LABS)
    plot_convergence(
        baseline_dat="data/run_f18_baseline/data_f18_LABS/IOHprofiler_f18_DIM50.dat",
        tuned_dat="data/run_f18_tuned/data_f18_LABS/IOHprofiler_f18_DIM50.dat",
        title="F18 (LABS, $d=50$) — Baseline vs. Tuned GA",
        budget=5000,
        save_path="f18_convergence.pdf",
    )

    # Example usage for F23 (N-Queens)
    plot_convergence(
        baseline_dat="data/run_f23_baseline/data_f23_NQueens/IOHprofiler_f23_DIM49.dat",
        tuned_dat="data/run_f23_tuned/data_f23_NQueens/IOHprofiler_f23_DIM49.dat",
        title="F23 (N-Queens, $d=49$) — Baseline vs. Tuned GA",
        budget=5000,
        save_path="f23_convergence.pdf",
    )