import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


DATASETS = {
    "breast_cancer": None,
    "ionosphere": 59,
    "spambase": 44,
    "banknote": 1462,
    "phoneme": 1489,
}

SCENARIOS = {
    "rate_shift": (("mcar", 0.10), ("mcar", 0.30)),
    "mechanism_shift": (("mcar", 0.20), ("mar", 0.20)),
    "shortcut_reversal": (("label_pos", 0.20), ("label_neg", 0.20)),
}


def load_data(name):
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype("float32"), data.target.astype(int)
    data = fetch_openml(data_id=DATASETS[name], as_frame=False, parser="auto")
    x = np.asarray(data.data, dtype="float32")
    y = LabelEncoder().fit_transform(np.asarray(data.target).astype(str))
    keep = np.isfinite(x).all(axis=1)
    return x[keep], y[keep]


def calibrated_probability(score, rate):
    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(score + mid)))
        if p.mean() < rate:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(score + (lo + hi) / 2)))


def mask_data(x, y, kind, rate, seed):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if kind == "mcar":
        p = np.full((n, d), rate)
    elif kind == "mar":
        anchor = (x[:, np.argmax(np.var(x, axis=0))] - np.mean(x[:, np.argmax(np.var(x, axis=0))]))
        anchor /= np.std(anchor) + 1e-8
        signs = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        p = np.column_stack([calibrated_probability(1.5 * anchor * s, rate) for s in signs])
    else:
        score = (y - y.mean()) / (y.std() + 1e-8)
        if kind == "label_neg":
            score = -score
        p0 = calibrated_probability(1.5 * score, rate)
        p = np.repeat(p0[:, None], d, axis=1)
    out = x.copy()
    out[rng.random((n, d)) < p] = np.nan
    return out


def models(seed):
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
            device="cuda",
            fit_mode="low_memory",
            memory_saving_mode=True,
            keep_cache_on_device=False,
            random_state=seed,
            show_progress_bar=False,
        ),
    }


def metrics(y, probability):
    classes = np.arange(probability.shape[1])
    ll = log_loss(y, probability, labels=classes)
    onehot = np.eye(len(classes))[y]
    brier = np.mean(np.sum((probability - onehot) ** 2, axis=1))
    auc = roc_auc_score(y, probability[:, 1]) if len(classes) == 2 else roc_auc_score(y, probability, multi_class="ovr", average="weighted")
    return ll, brier, auc


def run(dataset_names, seeds, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "model_scores.csv"
    rows = pd.read_csv(score_path).to_dict("records") if score_path.exists() else []
    completed = {(r["dataset"], int(r["seed"]), r["scenario"]) for r in rows}
    for dataset_name in dataset_names:
        x, y = load_data(dataset_name)
        for seed in seeds:
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            for scenario, (source, target) in SCENARIOS.items():
                if (dataset_name, seed, scenario) in completed:
                    continue
                source_kind, source_rate = source
                target_kind, target_rate = target
                train = mask_data(x_train, y_train, source_kind, source_rate, seed + 10)
                val = mask_data(x_val, y_val, source_kind, source_rate, seed + 20)
                iid_test = mask_data(x_test, y_test, source_kind, source_rate, seed + 30)
                shifted_test = mask_data(x_test, y_test, target_kind, target_rate, seed + 40)
                for model_name, model in models(seed).items():
                    model.fit(train, y_train)
                    for split, features, labels in (("validation", val, y_val), ("iid_test", iid_test, y_test), ("shifted_test", shifted_test, y_test)):
                        ll, brier, auc = metrics(labels, model.predict_proba(features))
                        rows.append({"dataset": dataset_name, "seed": seed, "scenario": scenario, "model": model_name, "split": split, "log_loss": ll, "brier": brier, "auc": auc})
                pd.DataFrame(rows).to_csv(score_path, index=False)
                gc.collect()
                torch.cuda.empty_cache()
                print(dataset_name, seed, scenario, "complete", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(score_path, index=False)

    selected = []
    for keys, cell in scores.groupby(["dataset", "seed", "scenario"]):
        val = cell[cell.split == "validation"].set_index("model").log_loss
        chosen = val.idxmin()
        record = dict(zip(("dataset", "seed", "scenario"), keys))
        record["selected_model"] = chosen
        for split in ("iid_test", "shifted_test"):
            target = cell[cell.split == split].set_index("model").log_loss
            record[f"{split}_oracle"] = target.idxmin()
            record[f"{split}_regret"] = target[chosen] - target.min()
            record[f"{split}_rank_spearman"] = val.corr(target, method="spearman")
        record["excess_regret"] = record["shifted_test_regret"] - record["iid_test_regret"]
        selected.append(record)
    selection = pd.DataFrame(selected)
    selection.to_csv(output_dir / "selection_results.csv", index=False)
    print(selection.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    names = ["breast_cancer"] if args.smoke else list(DATASETS)
    seeds = [0] if args.smoke else [0, 1, 2]
    run(names, seeds, Path("artifacts/experiment/smoke" if args.smoke else "artifacts/experiment/pilot_v1"))
