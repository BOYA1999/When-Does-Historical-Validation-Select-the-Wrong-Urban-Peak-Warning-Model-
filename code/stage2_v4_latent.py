import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pilot


RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTPUT = Path("artifacts/experiment/stage2_v4_latent")


def latent_mask(x, y, rho, seed):
    rng = np.random.default_rng(seed)
    y_score = (y - y.mean()) / (y.std() + 1e-8)
    latent = rho * y_score + np.sqrt(1 - rho**2) * rng.normal(size=len(y))
    p0 = pilot.calibrated_probability(1.5 * latent, 0.20)
    p = np.repeat(p0[:, None], x.shape[1], axis=1)
    out = x.copy()
    out[rng.random(x.shape) < p] = np.nan
    return out


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in pilot.DATASETS:
        x, y = pilot.load_data(dataset_name)
        for seed in [0, 1, 2]:
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            target = latent_mask(x_test, y_test, 0.0, seed + 40)
            for rho in RHOS:
                train = latent_mask(x_train, y_train, rho, seed + 10)
                val = latent_mask(x_val, y_val, rho, seed + 20)
                for model_name, model in pilot.models(seed).items():
                    model.fit(train, y_train)
                    for split, features, labels in [("validation", val, y_val), ("target", target, y_test)]:
                        ll, brier, auc = pilot.metrics(labels, model.predict_proba(features))
                        rows.append({"dataset": dataset_name, "seed": seed, "source_rho": rho, "model": model_name, "split": split, "log_loss": ll, "brier": brier, "auc": auc})
                pd.DataFrame(rows).to_csv(OUTPUT / "model_scores.csv", index=False)
                gc.collect()
                torch.cuda.empty_cache()
                print(dataset_name, seed, rho, "complete", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == 600
    selected = []
    for (dataset, seed, rho), cell in scores.groupby(["dataset", "seed", "source_rho"]):
        validation = cell[cell.split == "validation"].set_index("model")
        target = cell[cell.split == "target"].set_index("model")
        choice = validation.log_loss.idxmin()
        selected.append({"dataset": dataset, "seed": seed, "source_rho": rho, "selected_model": choice, "oracle": target.log_loss.idxmin(), "logloss_regret": target.loc[choice, "log_loss"] - target.log_loss.min(), "rank_spearman": validation.log_loss.corr(target.log_loss, method="spearman")})
    selection = pd.DataFrame(selected)
    selection.to_csv(OUTPUT / "selection_results.csv", index=False)
    summary = selection.assign(winner_flip=selection.selected_model != selection.oracle).groupby("source_rho", as_index=False).agg(median_logloss_regret=("logloss_regret", "median"), median_rank_spearman=("rank_spearman", "median"), winner_flip_rate=("winner_flip", "mean"))
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
