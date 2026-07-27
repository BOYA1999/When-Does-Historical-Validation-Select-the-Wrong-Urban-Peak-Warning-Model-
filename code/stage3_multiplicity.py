import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pilot


OUTPUT = Path("artifacts/experiment/stage3_multiplicity")


def source_mask(x, y, linked, order, seed):
    rng = np.random.default_rng(seed)
    p = np.full(x.shape, 0.20)
    y_score = (y - y.mean()) / (y.std() + 1e-8)
    p0 = pilot.calibrated_probability(1.5 * y_score, 0.20)
    p[:, order[:linked]] = p0[:, None]
    out = x.copy()
    out[rng.random(x.shape) < p] = np.nan
    return out


def target_mask(x, seed):
    out = x.copy()
    out[np.random.default_rng(seed).random(x.shape) < 0.20] = np.nan
    return out


def run(dataset_names=None, seeds=None, count_grid=None, output=OUTPUT):
    dataset_names = list(pilot.DATASETS) if dataset_names is None else dataset_names
    seeds = [0, 1, 2] if seeds is None else seeds
    count_grid = [1, 2, 4, 8, 16, 32] if count_grid is None else count_grid
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = 0
    for dataset_name in dataset_names:
        x, y = pilot.load_data(dataset_name)
        linked_counts = sorted(set(min(value, x.shape[1]) for value in [*count_grid, x.shape[1]]))
        expected += len(linked_counts) * len(seeds) * 8
        for seed in seeds:
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            order = np.random.default_rng(seed + 5).permutation(x.shape[1])
            target = target_mask(x_test, seed + 40)
            for linked in linked_counts:
                train = source_mask(x_train, y_train, linked, order, seed + 10)
                val = source_mask(x_val, y_val, linked, order, seed + 20)
                for model_name, model in pilot.models(seed).items():
                    model.fit(train, y_train)
                    for split, features, labels in [("validation", val, y_val), ("target", target, y_test)]:
                        ll, brier, auc = pilot.metrics(labels, model.predict_proba(features))
                        rows.append({"dataset": dataset_name, "seed": seed, "features": x.shape[1], "linked": linked, "model": model_name, "split": split, "log_loss": ll, "brier": brier, "auc": auc})
                pd.DataFrame(rows).to_csv(output / "model_scores.csv", index=False)
                gc.collect()
                torch.cuda.empty_cache()
                print(dataset_name, seed, linked, "complete", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == expected
    selected = []
    for (dataset, seed, linked), cell in scores.groupby(["dataset", "seed", "linked"]):
        validation = cell[cell.split == "validation"].set_index("model")
        target = cell[cell.split == "target"].set_index("model")
        choice = validation.log_loss.idxmin()
        selected.append({"dataset": dataset, "seed": seed, "linked": linked, "features": int(cell.features.iloc[0]), "selected_model": choice, "oracle": target.log_loss.idxmin(), "logloss_regret": target.loc[choice, "log_loss"] - target.log_loss.min()})
    selection = pd.DataFrame(selected)
    selection.to_csv(output / "selection_results.csv", index=False)
    summary = selection.groupby("linked", as_index=False).agg(median_logloss_regret=("logloss_regret", "median"), cells=("logloss_regret", "size"))
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
