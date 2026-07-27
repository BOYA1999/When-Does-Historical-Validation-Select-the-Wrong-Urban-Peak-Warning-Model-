import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pilot


ALPHAS = [1.5, 0.0, -0.5, -1.5]
OUTPUT = Path("artifacts/experiment/stage2_v3_proxy")


def policy_mask(x, score, rate, alpha, seed, protected_index=None):
    p0 = pilot.calibrated_probability(alpha * score, rate)
    p = np.repeat(p0[:, None], x.shape[1], axis=1)
    if protected_index is not None:
        p[:, protected_index] = 0
    out = x.copy()
    out[np.random.default_rng(seed).random(x.shape) < p] = np.nan
    return out


def standardized_anchor(train, other, index):
    mean = train[:, index].mean()
    scale = train[:, index].std() + 1e-8
    return (other[:, index] - mean) / scale


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in pilot.DATASETS:
        x, y = pilot.load_data(dataset_name)
        for seed in [0, 1, 2]:
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            correlations = [abs(np.corrcoef(x_train[:, j], y_train)[0, 1]) for j in range(x_train.shape[1])]
            anchor = int(np.nanargmax(correlations))
            train_score = standardized_anchor(x_train, x_train, anchor)
            val_score = standardized_anchor(x_train, x_val, anchor)
            test_score = standardized_anchor(x_train, x_test, anchor)

            for family in ["observed_proxy", "withheld_proxy"]:
                if family == "observed_proxy":
                    train_base, val_base, test_base = x_train, x_val, x_test
                    protected = anchor
                else:
                    train_base = np.delete(x_train, anchor, axis=1)
                    val_base = np.delete(x_val, anchor, axis=1)
                    test_base = np.delete(x_test, anchor, axis=1)
                    protected = None
                train = policy_mask(train_base, train_score, 0.20, 1.5, seed + 10, protected)
                val = policy_mask(val_base, val_score, 0.20, 1.5, seed + 20, protected)
                targets = {alpha: policy_mask(test_base, test_score, 0.20, alpha, seed + 40, protected) for alpha in ALPHAS}

                for model_name, model in pilot.models(seed).items():
                    model.fit(train, y_train)
                    ll, brier, auc = pilot.metrics(y_val, model.predict_proba(val))
                    rows.append({"dataset": dataset_name, "seed": seed, "family": family, "anchor": anchor, "model": model_name, "split": "validation", "target_alpha": 1.5, "log_loss": ll, "brier": brier, "auc": auc})
                    for alpha, target in targets.items():
                        ll, brier, auc = pilot.metrics(y_test, model.predict_proba(target))
                        rows.append({"dataset": dataset_name, "seed": seed, "family": family, "anchor": anchor, "model": model_name, "split": "target", "target_alpha": alpha, "log_loss": ll, "brier": brier, "auc": auc})
                pd.DataFrame(rows).to_csv(OUTPUT / "model_scores.csv", index=False)
                gc.collect()
                torch.cuda.empty_cache()
                print(dataset_name, seed, family, "complete", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == 600
    selected = []
    for (dataset, seed, family), cell in scores.groupby(["dataset", "seed", "family"]):
        validation = cell[cell.split == "validation"].set_index("model")
        choice = validation.log_loss.idxmin()
        for alpha, target in cell[cell.split == "target"].groupby("target_alpha"):
            target = target.set_index("model")
            selected.append({"dataset": dataset, "seed": seed, "family": family, "target_alpha": alpha, "selected_model": choice, "oracle": target.log_loss.idxmin(), "logloss_regret": target.loc[choice, "log_loss"] - target.log_loss.min(), "rank_spearman": validation.log_loss.corr(target.log_loss, method="spearman")})
    selection = pd.DataFrame(selected)
    selection.to_csv(OUTPUT / "selection_results.csv", index=False)
    summary = selection.assign(winner_flip=selection.selected_model != selection.oracle).groupby(["family", "target_alpha"], as_index=False).agg(median_logloss_regret=("logloss_regret", "median"), median_rank_spearman=("rank_spearman", "median"), winner_flip_rate=("winner_flip", "mean"))
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
