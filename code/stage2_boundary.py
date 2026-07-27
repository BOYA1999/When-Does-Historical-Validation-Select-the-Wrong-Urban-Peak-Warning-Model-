import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pilot


ALPHAS = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
OUTPUT = Path("artifacts/experiment/stage2_v1")


def label_mask(x, y, rate, alpha, seed):
    score = (y - y.mean()) / (y.std() + 1e-8)
    p0 = pilot.calibrated_probability(alpha * score, rate)
    p = np.repeat(p0[:, None], x.shape[1], axis=1)
    out = x.copy()
    out[np.random.default_rng(seed).random(x.shape) < p] = np.nan
    return out


def run(dataset_names=None, seeds=None, output=OUTPUT):
    dataset_names = list(pilot.DATASETS) if dataset_names is None else dataset_names
    seeds = [0, 1, 2] if seeds is None else seeds
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "model_scores.csv"
    rows = pd.read_csv(score_path).to_dict("records") if score_path.exists() else []
    counts = pd.DataFrame(rows).groupby(["dataset", "seed"]).size() if rows else pd.Series(dtype=int)
    completed = set(counts[counts == 32].index)

    for dataset_name in dataset_names:
        x, y = pilot.load_data(dataset_name)
        for seed in seeds:
            if (dataset_name, seed) in completed:
                continue
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            train = label_mask(x_train, y_train, 0.20, 1.5, seed + 10)
            val = label_mask(x_val, y_val, 0.20, 1.5, seed + 20)
            targets = {alpha: label_mask(x_test, y_test, 0.20, alpha, seed + 40) for alpha in ALPHAS}

            for model_name, model in pilot.models(seed).items():
                model.fit(train, y_train)
                ll, brier, auc = pilot.metrics(y_val, model.predict_proba(val))
                rows.append({"dataset": dataset_name, "seed": seed, "model": model_name, "split": "validation", "target_alpha": 1.5, "log_loss": ll, "brier": brier, "auc": auc})
                for alpha, target in targets.items():
                    ll, brier, auc = pilot.metrics(y_test, model.predict_proba(target))
                    rows.append({"dataset": dataset_name, "seed": seed, "model": model_name, "split": "target", "target_alpha": alpha, "log_loss": ll, "brier": brier, "auc": auc})

            pd.DataFrame(rows).to_csv(score_path, index=False)
            gc.collect()
            torch.cuda.empty_cache()
            print(dataset_name, seed, "complete", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == len(dataset_names) * len(seeds) * 32
    selected = []
    for (dataset, seed), cell in scores.groupby(["dataset", "seed"]):
        validation = cell[cell.split == "validation"].set_index("model")
        logloss_choice = validation.log_loss.idxmin()
        auc_choice = validation.auc.idxmax()
        for alpha, target in cell[cell.split == "target"].groupby("target_alpha"):
            target = target.set_index("model")
            selected.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "target_alpha": alpha,
                    "logloss_selected_model": logloss_choice,
                    "logloss_oracle": target.log_loss.idxmin(),
                    "logloss_regret": target.loc[logloss_choice, "log_loss"] - target.log_loss.min(),
                    "logloss_rank_spearman": validation.log_loss.corr(target.log_loss, method="spearman"),
                    "auc_selected_model": auc_choice,
                    "auc_oracle": target.auc.idxmax(),
                    "auc_regret": target.auc.max() - target.loc[auc_choice, "auc"],
                }
            )
    selection = pd.DataFrame(selected)
    selection.to_csv(output / "selection_results.csv", index=False)
    summary = (
        selection.assign(winner_flip=selection.logloss_selected_model != selection.logloss_oracle)
        .groupby("target_alpha", as_index=False)
        .agg(
            median_logloss_regret=("logloss_regret", "median"),
            median_rank_spearman=("logloss_rank_spearman", "median"),
            winner_flip_rate=("winner_flip", "mean"),
            median_auc_regret=("auc_regret", "median"),
        )
        .sort_values("target_alpha", ascending=False)
    )
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
