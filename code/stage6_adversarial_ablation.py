import argparse
import gc
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import pilot
from stage2_boundary import ALPHAS, label_mask


HOLDOUT = {
    "diabetes": 37,
    "sonar": 40,
    "blood_transfusion": 1464,
    "qsar_biodeg": 1494,
    "kc1": 1067,
    "pc1": 1068,
    "hill_valley": 1479,
}
SELECTORS = {"standard": 1.5, "neutral": 0.0, "reversed": -1.5}


def models(seed, device):
    return {
        "logistic_indicator": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
        "random_forest_indicator": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=4, random_state=seed),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "tabpfn_v2": TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=1,
            device=device,
            fit_mode="low_memory",
            memory_saving_mode=True,
            keep_cache_on_device=False,
            random_state=seed,
            show_progress_bar=False,
        ),
    }


def summarize(scores, output):
    selected = []
    for (dataset, seed), cell in scores.groupby(["dataset", "seed"]):
        validation = cell[cell.split == "validation"].pivot(index="model", columns="association", values="log_loss")
        targets = cell[cell.split == "target"].pivot(index="model", columns="association", values="log_loss")
        for validation_alpha in ALPHAS:
            validation_loss = validation[validation_alpha]
            chosen = validation_loss.idxmin()
            for target_alpha in ALPHAS:
                target_loss = targets[target_alpha]
                oracle = target_loss.idxmin()
                selected.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "validation_alpha": validation_alpha,
                        "target_alpha": target_alpha,
                        "selected_model": chosen,
                        "oracle": oracle,
                        "logloss_regret": target_loss[chosen] - target_loss.min(),
                        "rank_spearman": validation_loss.corr(target_loss, method="spearman"),
                        "winner_flip": chosen != oracle,
                    }
                )
    selection = pd.DataFrame(selected)
    selection.to_csv(output / "selection_results.csv", index=False)
    dataset_summary = (
        selection.groupby(["dataset", "validation_alpha", "target_alpha"], as_index=False)
        .agg(
            median_regret=("logloss_regret", "median"),
            median_rank_spearman=("rank_spearman", "median"),
            winner_flip_rate=("winner_flip", "mean"),
        )
    )
    dataset_summary.to_csv(output / "phase_dataset_summary.csv", index=False)
    phase = (
        dataset_summary.groupby(["validation_alpha", "target_alpha"], as_index=False)
        .agg(
            median_dataset_regret=("median_regret", "median"),
            median_rank_spearman=("median_rank_spearman", "median"),
            threshold_fraction=("median_regret", lambda x: np.mean(x >= 0.02)),
        )
    )
    phase.to_csv(output / "phase_summary.csv", index=False)

    selector_rows = []
    choice_rows = selection[np.isclose(selection.target_alpha, 1.5)].pivot(
        index=["dataset", "seed"], columns="validation_alpha", values="selected_model"
    )
    for selector, validation_alpha in SELECTORS.items():
        for target_alpha in [1.5, 0.0]:
            part = dataset_summary[
                np.isclose(dataset_summary.validation_alpha, validation_alpha)
                & np.isclose(dataset_summary.target_alpha, target_alpha)
            ]
            selector_rows.append(
                {
                    "selector": selector,
                    "validation_alpha": validation_alpha,
                    "target_alpha": target_alpha,
                    "median_dataset_regret": part.median_regret.median(),
                    "datasets_at_threshold": int((part.median_regret >= 0.02).sum()),
                    "choice_change_rate_vs_standard": float(
                        0 if selector == "standard" else (choice_rows[validation_alpha] != choice_rows[1.5]).mean()
                    ),
                }
            )
    equal_budget = pd.DataFrame(selector_rows)
    equal_budget.to_csv(output / "equal_budget_summary.csv", index=False)

    standard_shift = dataset_summary[
        np.isclose(dataset_summary.validation_alpha, 1.5) & np.isclose(dataset_summary.target_alpha, 0)
    ].set_index("dataset").median_regret
    alternatives = equal_budget[(equal_budget.selector != "standard") & np.isclose(equal_budget.target_alpha, 0)]
    best_selector = alternatives.sort_values("median_dataset_regret").iloc[0].selector
    best_alpha = SELECTORS[best_selector]
    best_shift = dataset_summary[
        np.isclose(dataset_summary.validation_alpha, best_alpha) & np.isclose(dataset_summary.target_alpha, 0)
    ].set_index("dataset").median_regret
    mismatch = dataset_summary.assign(distance=(dataset_summary.validation_alpha - dataset_summary.target_alpha).abs())
    mismatch_corr = mismatch.groupby("dataset").apply(
        lambda x: x.distance.corr(x.median_regret, method="spearman"), include_groups=False
    )
    mismatch_corr.rename("mismatch_regret_spearman").to_csv(output / "mismatch_correlations.csv")
    flip_rate = float((choice_rows[best_alpha] != choice_rows[1.5]).mean())
    median_improvement = float(standard_shift.median() - best_shift.median())
    improved_datasets = int(((standard_shift - best_shift) >= 0.02).sum())
    crossing_cells = int((phase.median_dataset_regret >= 0.02).sum())
    stable_datasets = int((mismatch_corr > 0.4).sum())
    gate = {
        "best_equal_budget_selector": best_selector,
        "choice_change_rate_vs_standard": flip_rate,
        "shifted_target_median_improvement": median_improvement,
        "datasets_improved_by_0.02": improved_datasets,
        "phase_cells_at_0.02": crossing_cells,
        "datasets_with_mismatch_spearman_gt_0.4": stable_datasets,
        "equal_budget_pass": flip_rate >= 0.30 and median_improvement >= 0.02 and improved_datasets >= 3,
        "phase_boundary_pass": crossing_cells >= 4 and stable_datasets >= 3,
    }
    gate["full_expansion_go"] = gate["equal_budget_pass"] and gate["phase_boundary_pass"]
    (output / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)


def run(dataset_names, seeds, output, device, threads):
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "model_scores.csv"
    existing = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    if len(existing):
        counts = existing.groupby(["dataset", "seed"]).size()
        completed = set(counts[counts == 56].index)
        existing = existing.set_index(["dataset", "seed"])
        existing = existing[existing.index.isin(completed)].reset_index()
    else:
        completed = set()
    rows = existing.to_dict("records")
    for dataset_name in dataset_names:
        x, y = pilot.load_data(dataset_name)
        for seed in seeds:
            if (dataset_name, seed) in completed:
                continue
            x_train, x_hold, y_train, y_hold = train_test_split(
                x, y, test_size=0.4, stratify=y, random_state=seed
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed
            )
            train = label_mask(x_train, y_train, 0.20, 1.5, seed + 10)
            validations = {a: label_mask(x_val, y_val, 0.20, a, seed + 20) for a in ALPHAS}
            targets = {a: label_mask(x_test, y_test, 0.20, a, seed + 40) for a in ALPHAS}
            for model_name, model in models(seed, device).items():
                model.fit(train, y_train)
                for split, collections, labels in (
                    ("validation", validations, y_val),
                    ("target", targets, y_test),
                ):
                    associations = list(collections)
                    sizes = [len(collections[a]) for a in associations]
                    probabilities = model.predict_proba(np.concatenate([collections[a] for a in associations]))
                    offset = 0
                    for association, size in zip(associations, sizes):
                        ll, brier, auc = pilot.metrics(labels, probabilities[offset:offset + size])
                        offset += size
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "model": model_name,
                                "split": split,
                                "association": association,
                                "log_loss": ll,
                                "brier": brier,
                                "auc": auc,
                            }
                        )
            pd.DataFrame(rows).to_csv(score_path, index=False)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            print(dataset_name, seed, "complete", flush=True)
    scores = pd.DataFrame(rows)
    expected = len(dataset_names) * len(seeds) * 56
    if len(scores) != expected or scores.log_loss.isna().any():
        raise RuntimeError(f"invalid score matrix: rows={len(scores)}, expected={expected}")
    summarize(scores, output)
    manifest = {
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "threads": threads,
        "datasets": dataset_names,
        "seeds": seeds,
        "training_alpha": 1.5,
        "validation_and_target_alphas": ALPHAS,
        "validation_evaluations_per_selector": 1,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    all_datasets = dict(pilot.DATASETS)
    all_datasets.update(HOLDOUT)
    pilot.DATASETS.update(HOLDOUT)
    if args.mode == "smoke":
        names, seeds = ["breast_cancer"], [0]
    elif args.mode == "pilot":
        names, seeds = list(pilot.DATASETS)[:5], [0, 1, 2]
    else:
        names, seeds = list(all_datasets), list(range(10))
    run(names, seeds, Path(f"artifacts/experiment/stage6_adversarial_ablation_{args.mode}"), args.device, args.threads)
