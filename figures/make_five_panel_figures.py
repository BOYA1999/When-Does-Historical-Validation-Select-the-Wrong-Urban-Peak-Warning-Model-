from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiment"
ENERGY = ROOT / "artifacts" / "energy_case" / "gate_17_buildings"
OUT = Path(__file__).resolve().parent
plt.style.use(OUT / "academic.mplstyle")

BLUE = "#3B6C8E"
GREEN = "#5B8E7D"
RED = "#B85C5C"
GOLD = "#B28A4A"
PURPLE = "#756A91"
GRAY = "#8B8B8B"
LIGHT = "#ECE9E4"
INK = "#374151"
ALPHAS = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", fontsize=9.2, pad=6)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def short_model(name):
    return {
        "logistic_indicator": "LR",
        "random_forest_indicator": "RF",
        "hist_gradient_boosting": "HGB",
        "tabpfn_v2": "TabPFN",
    }[name]


def figure1():
    summary = pd.read_csv(ENERGY / "building_summary.csv")
    selection = pd.read_csv(ENERGY / "selection.csv")
    diagnostics = pd.read_csv(ENERGY / "diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), gridspec_kw={"height_ratios": [0.82, 1.18]})
    ax = axes[0, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, "a", "The auditable failure")
    boxes = [
        (0.01, BLUE, "Develop", "Peak-linked gaps\nselect HGB"),
        (0.35, GOLD, "Change", "Gap--peak link\nremoved\nloads retained"),
        (0.69, RED, "Deploy", "Another candidate\nis empirical target\noracle"),
    ]
    for x, color, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.20), 0.29, 0.50, boxstyle="round,pad=.02",
                                    facecolor="white", edgecolor=color, linewidth=1.4))
        ax.text(x + 0.145, 0.59, title, ha="center", fontweight="bold", color=color, fontsize=8.5)
        ax.text(x + 0.145, 0.38, body, ha="center", va="center", color=INK, fontsize=5.7, linespacing=1.35)
    for x1, x2 in ((0.305, 0.345), (0.645, 0.685)):
        ax.add_patch(FancyArrowPatch((x1, 0.45), (x2, 0.45), arrowstyle="-|>", mutation_scale=10,
                                     color=GRAY, linewidth=1.1))

    ax = axes[0, 1]
    panel(ax, "b", "17/17 buildings exceed 0.02")
    x = np.arange(len(summary))
    for i, row in summary.iterrows():
        ax.plot([i, i], [row.matched_regret, row.shifted_regret], color=LIGHT, linewidth=1.2, zorder=1)
    ax.scatter(x, summary.matched_regret, color=GRAY, s=19, label="Matched", zorder=3)
    ax.scatter(x, summary.shifted_regret, color=RED, s=23, label="Association removed", zorder=3)
    ax.axhline(0.02, color=RED, linestyle="--", linewidth=1.0)
    ax.set_xticks(x[::2], [f"B{i}" for i in summary.building.iloc[::2]])
    ax.set_ylabel("Median selection regret")
    ax.set_ylim(-0.03, 1.13)
    ax.legend(fontsize=7.2, loc="upper left")

    ax = axes[1, 0]
    panel(ax, "c", "34/34 historical winners flip")
    order = ["logistic_indicator", "random_forest_indicator", "hist_gradient_boosting", "tabpfn_v2"]
    matrix = pd.crosstab(selection.selected_model, selection.shifted_oracle).reindex(index=order, columns=order, fill_value=0)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.to_numpy().max()))
    for i in range(4):
        for j in range(4):
            value = int(matrix.iloc[i, j])
            ax.text(j, i, value, ha="center", va="center", color="white" if value > matrix.to_numpy().max() / 2 else INK,
                    fontsize=9, fontweight="bold")
    ax.set_xticks(range(4), [short_model(x) for x in order], rotation=25, ha="right")
    ax.set_yticks(range(4), [short_model(x) for x in order])
    ax.set_xlabel("Empirical target oracle after removal")
    ax.set_ylabel("Historical validation winner")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Building--fold count")

    ax = axes[1, 1]
    panel(ax, "d", "The telemetry shortcut disappears")
    groups = [
        diagnostics[diagnostics.split == "matched_target"].missing_count_label_corr.to_numpy(),
        diagnostics[diagnostics.split == "shifted_target"].missing_count_label_corr.to_numpy(),
    ]
    rng = np.random.default_rng(4)
    bp = ax.boxplot(groups, positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [GRAY, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    for i, (values, color) in enumerate(zip(groups, [GRAY, RED])):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(values)), values, s=14, color=color, alpha=0.55)
        ax.text(i, 0.08, f"median {np.median(values):.3f}", ha="center", fontsize=7.4)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=0.9)
    ax.set_xticks([0, 1], ["Matched", "Association\nremoved"])
    ax.set_ylabel("Missing-count--peak correlation")
    ax.set_ylim(-0.12, 1.05)
    fig.tight_layout()
    save(fig, "fig1_citylearn_failure")


def stage6_dir():
    full = EXP / "stage6_adversarial_ablation_full"
    pilot = EXP / "stage6_adversarial_ablation_pilot"
    return full if (full / "gate.json").exists() else pilot


def figure2():
    source = stage6_dir()
    phase = pd.read_csv(source / "phase_summary.csv")
    dataset = pd.read_csv(source / "phase_dataset_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    pivots = [
        ("median_dataset_regret", "a", "Selection-regret phase map", "YlOrRd", 0, None, ".2f"),
        ("median_rank_spearman", "b", "Ranking-stability phase map", "RdBu_r", -1, 1, ".1f"),
    ]
    for ax, (value, letter, title, cmap, vmin, vmax, fmt) in zip(axes[0], pivots):
        table = phase.pivot(index="validation_alpha", columns="target_alpha", values=value).reindex(index=ALPHAS, columns=ALPHAS)
        im = ax.imshow(table, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        limit = np.nanmax(np.abs(table.to_numpy()))
        for i in range(7):
            for j in range(7):
                val = table.iloc[i, j]
                color = "white" if abs(val) > 0.58 * limit else INK
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=6.5, color=color)
        ax.set_xticks(range(7), [f"{x:g}" for x in ALPHAS])
        ax.set_yticks(range(7), [f"{x:g}" for x in ALPHAS])
        ax.set_xlabel("Target association")
        ax.set_ylabel("Validation association")
        panel(ax, letter, title)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    mismatch = dataset.assign(distance=(dataset.validation_alpha - dataset.target_alpha).abs())
    within = mismatch.groupby(["dataset", "distance"], as_index=False).median_regret.median()
    aggregate = within.groupby("distance").median_regret.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).reset_index()
    aggregate.columns = ["distance", "median", "q25", "q75"]
    ax = axes[1, 0]
    panel(ax, "c", "Regret rises with mismatch")
    ax.fill_between(aggregate.distance, aggregate.q25, aggregate.q75, color=BLUE, alpha=0.17)
    ax.plot(aggregate.distance, aggregate["median"], marker="o", color=BLUE)
    ax.axhline(0.02, color=RED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Absolute association mismatch")
    ax.set_ylabel("Median dataset regret")
    ax.set_ylim(bottom=0)

    threshold = mismatch.assign(hit=mismatch.median_regret >= 0.02).groupby("distance", as_index=False).hit.mean()
    ax = axes[1, 1]
    panel(ax, "d", "Boundary crossings increase")
    ax.plot(threshold.distance, 100 * threshold.hit, marker="o", color=RED)
    ax.set_xlabel("Absolute association mismatch")
    ax.set_ylabel("Dataset--cell pairs >=0.02 (%)")
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    save(fig, "fig2_phase_boundary")


def figure3():
    path = EXP / "stage3_multiplicity_holdout"
    selection = pd.read_csv(path / "selection_results.csv")
    dose = pd.read_csv(path / "dose_response.csv")
    selection["fraction"] = selection.linked / selection.features
    median_curve = selection.groupby(["dataset", "linked", "features"], as_index=False).logloss_regret.median()
    median_curve["fraction"] = median_curve.linked / median_curve.features
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    colors = [BLUE, GREEN, RED, GOLD, PURPLE, "#A0835D", "#5F7775"]
    ax = axes[0, 0]
    panel(ax, "a", "Held-out dose curves")
    for color, (name, cell) in zip(colors, median_curve.groupby("dataset")):
        cell = cell.sort_values("fraction")
        ax.plot(cell.fraction, cell.logloss_regret, marker="o", color=color, alpha=.88,
                label=name.replace("_", " ").title())
    ax.axhline(0.02, color=RED, linestyle="--", linewidth=.9)
    ax.set_xlabel("Outcome-linked channel fraction")
    ax.set_ylabel("Median selection regret")
    ax.legend(fontsize=6.0, ncol=2)

    ax = axes[0, 1]
    panel(ax, "b", "Cell-level dose direction")
    values = dose.multiplicity_regret_spearman.dropna().to_numpy()
    rng = np.random.default_rng(7)
    ax.boxplot(values, positions=[0], widths=.35, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#DCE7ED", "edgecolor": BLUE}, medianprops={"color": RED, "linewidth": 1.7})
    ax.scatter(rng.uniform(-.10, .10, len(values)), values, s=17, color=BLUE, alpha=.65)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=.9)
    ax.set_xticks([])
    ax.set_ylabel("Within-cell Spearman")
    ax.text(.05, .96, f"median = {np.median(values):.3f}\n{np.sum(values > 0)}/{len(values)} positive\n2 undefined retained",
            transform=ax.transAxes, va="top", fontsize=7.5)

    ax = axes[1, 0]
    panel(ax, "c", "Boundary crossings increase")
    bins = pd.cut(selection.fraction, [-.001, .10, .25, .50, .75, 1.01], labels=[.05, .175, .375, .625, .875])
    crossing = selection.assign(bin=bins, hit=selection.logloss_regret >= .02).groupby("bin", observed=True).hit.mean()
    ax.plot(crossing.index.astype(float), 100 * crossing.values, marker="o", color=RED)
    ax.set_xlabel("Binned linked-channel fraction")
    ax.set_ylabel("Dataset--seed cells >=0.02 (%)")
    ax.set_ylim(-3, 103)

    ax = axes[1, 1]
    panel(ax, "d", "A retained non-monotonic exception")
    worst = dose.sort_values("multiplicity_regret_spearman").iloc[0]
    cell = selection[(selection.dataset == worst.dataset) & (selection.seed == worst.seed)].sort_values("fraction")
    ax.plot(cell.fraction, cell.logloss_regret, marker="o", color=GOLD)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=.9)
    ax.set_xlabel("Outcome-linked channel fraction")
    ax.set_ylabel("Selection regret")
    ax.text(.04, .95, f"{worst.dataset.replace('_', ' ').title()}, seed {int(worst.seed)}\nSpearman = {worst.multiplicity_regret_spearman:.1f}",
            transform=ax.transAxes, va="top", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_repeated_channels")


def figure4():
    pilot_summary = pd.read_csv(EXP / "pilot_v1" / "scenario_summary.csv")
    proxy = pd.read_csv(EXP / "stage2_v3_proxy" / "summary.csv")
    latent = pd.read_csv(EXP / "stage2_v4_latent" / "summary.csv")
    dataset = pd.read_csv(EXP / "stage5_statistics" / "dataset_alpha_summary.csv")
    selection = pd.read_csv(ENERGY / "selection.csv")
    diagnostics = pd.read_csv(ENERGY / "diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    labels = ["Rate shift", "Proxy onset", "Observed proxy", "Withheld proxy", "Latent rho=.75", "Strong linked rho=1"]
    values = [
        float(pilot_summary.loc[pilot_summary.scenario == "rate_shift", "median_shifted_regret"].iloc[0]),
        float(pilot_summary.loc[pilot_summary.scenario == "mechanism_shift", "median_shifted_regret"].iloc[0]),
        float(proxy.loc[proxy.family == "observed_proxy", "median_logloss_regret"].max()),
        float(proxy.loc[proxy.family == "withheld_proxy", "median_logloss_regret"].max()),
        float(latent.loc[np.isclose(latent.source_rho, .75), "median_logloss_regret"].iloc[0]),
        float(latent.loc[np.isclose(latent.source_rho, 1), "median_logloss_regret"].iloc[0]),
    ]
    ax = axes[0, 0]
    panel(ax, "a", "Most changes are negative controls")
    colors = [GRAY] * 4 + [GOLD, RED]
    ax.barh(np.arange(6), values, color=colors, alpha=.78)
    ax.scatter(values, np.arange(6), color=colors, s=20)
    ax.axvline(.02, color=RED, linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(6), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Median shifted-target regret")

    ax = axes[0, 1]
    panel(ax, "b", "Latent association crosses late")
    ax.plot(latent.source_rho, latent.median_logloss_regret, marker="o", color=GOLD)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("Latent outcome association")
    ax.set_ylabel("Median selection regret")
    ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    panel(ax, "c", "Dataset counterexamples")
    zero = dataset[np.isclose(dataset.target_alpha, 0)].copy()
    ax.scatter(zero.median_rank_spearman, zero.median_logloss_regret,
               s=35 + 60 * zero.winner_flip_rate, c=np.where(zero.median_logloss_regret >= .02, RED, GRAY), alpha=.82)
    label_rows = zero.nlargest(3, "median_logloss_regret")
    for _, row in label_rows.iterrows():
        ax.annotate(row.dataset.replace("_", " "), (row.median_rank_spearman, row.median_logloss_regret),
                    xytext=(3, 2), textcoords="offset points", fontsize=6.0)
    low_count = int((zero.median_logloss_regret < .02).sum())
    ax.text(.98, .96, f"{low_count} datasets remain below 0.02", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color=INK)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=.9)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=.9)
    ax.set_xlabel("Rank Spearman")
    ax.set_ylabel("Median selection regret")

    rates = diagnostics.pivot_table(index=["building", "fold"], columns="split", values="peak_rate").reset_index()
    matched = selection[["building", "fold", "matched_regret"]]
    rates = rates.merge(matched, on=["building", "fold"])
    rates["peak_rate_change"] = rates.matched_target - rates.train
    ax = axes[1, 1]
    panel(ax, "d", "Building 4 outlier retained")
    colors = np.where(rates.building == 4, RED, GRAY)
    ax.scatter(100 * rates.peak_rate_change, rates.matched_regret, c=colors, s=np.where(rates.building == 4, 48, 24), alpha=.82)
    b4 = rates[rates.building == 4].sort_values("matched_regret").iloc[-1]
    ax.annotate("Building 4", (100 * b4.peak_rate_change, b4.matched_regret), xytext=(5, -2), textcoords="offset points", fontsize=7.4)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=.9)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=.9)
    ax.set_xlabel("Peak-rate change (pp)")
    ax.set_ylabel("Matched-condition regret")
    fig.tight_layout()
    save(fig, "fig4_failure_atlas")


def figure5():
    summary = pd.read_csv(stage6_dir() / "equal_budget_summary.csv")
    selectors = ["standard", "neutral", "reversed"]
    labels = ["Historical", "Neutral", "Reversed"]
    colors = [GRAY, BLUE, GREEN]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, "a", "Equal-budget contract")
    for i, (label, color, alpha) in enumerate(zip(labels, colors, [1.5, 0, -1.5])):
        x = .02 + i * .33
        ax.add_patch(FancyBboxPatch((x, .24), .28, .45, boxstyle="round,pad=.02", facecolor="white", edgecolor=color, linewidth=1.4))
        ax.text(x + .14, .58, label, ha="center", color=color, fontweight="bold", fontsize=8.2)
        ax.text(x + .14, .40, f"one mask\nalpha = {alpha:g}", ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(.50, .10, "Same trained models, parameters, split, metric and validation size", ha="center", fontsize=7.5, color=INK)

    ax = axes[0, 1]
    panel(ax, "b", "Matched versus changed")
    x = np.arange(2)
    width = .24
    for i, (selector, label, color) in enumerate(zip(selectors, labels, colors)):
        vals = [
            summary[(summary.selector == selector) & np.isclose(summary.target_alpha, 0)].median_dataset_regret.iloc[0],
            summary[(summary.selector == selector) & np.isclose(summary.target_alpha, 1.5)].median_dataset_regret.iloc[0],
        ]
        positions = x + (i - 1) * width
        ax.bar(positions, vals, width, color=color, label=label)
        ax.scatter(positions, vals, marker="_", s=38, color=color, zorder=4)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=1)
    ax.set_xticks(x, ["Association\nremoved", "Matched"])
    ax.set_ylabel("Median dataset regret")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    panel(ax, "c", "Risk-transfer plane")
    markers = ["o", "s", "^"]
    offsets = [(4, 3), (5, 6), (5, -12)]
    for selector, label, color, marker, offset in zip(selectors, labels, colors, markers, offsets):
        matched = summary[(summary.selector == selector) & np.isclose(summary.target_alpha, 1.5)].median_dataset_regret.iloc[0]
        shifted = summary[(summary.selector == selector) & np.isclose(summary.target_alpha, 0)].median_dataset_regret.iloc[0]
        face = "none" if selector == "neutral" else color
        ax.scatter(matched, shifted, s=55, marker=marker, facecolor=face, edgecolor=color, linewidth=1.4, zorder=4)
        ax.annotate(label, (matched, shifted), xytext=offset, textcoords="offset points", fontsize=7.2)
    ax.axhline(.02, color=RED, linestyle="--", linewidth=.9)
    ax.axvline(.02, color=RED, linestyle="--", linewidth=.9)
    ax.set_xlabel("Matched-target regret")
    ax.set_ylabel("Association-removed regret")
    ax.set_xlim(left=-.01)
    ax.set_ylim(bottom=-.01)

    ax = axes[1, 1]
    panel(ax, "d", "Choice changes")
    rates = summary[np.isclose(summary.target_alpha, 0)].set_index("selector").choice_change_rate_vs_standard
    ax.bar([0, 1, 2], 100 * rates.reindex(selectors), color=colors)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylabel("Choice change vs historical (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    save(fig, "fig5_equal_budget_ablation")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
