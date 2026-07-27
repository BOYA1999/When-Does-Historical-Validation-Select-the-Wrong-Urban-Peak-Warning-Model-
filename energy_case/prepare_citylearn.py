import hashlib
import json
from pathlib import Path

import pandas as pd


DATASET = "citylearn_challenge_2022_phase_all"
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT / "data" / "citylearn_v2.5.0"
DATA_ROOT = REPO_ROOT / "data" / "datasets" / DATASET
OUT = ROOT / "artifacts" / "energy_case" / "baseline"


def main():
    schema_path = DATA_ROOT / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError("Clone the official CityLearn v2.5.0 tag into data/citylearn_v2.5.0 first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    dataset_root = schema_path.parent
    files = []
    for path in sorted(p for p in dataset_root.rglob("*") if p.is_file()):
        record = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            record.update(
                rows=len(frame),
                columns=list(frame.columns),
                missing_cells=int(frame.isna().sum().sum()),
            )
        files.append(record)

    buildings = [name for name, cfg in schema["buildings"].items() if cfg.get("include", True)]
    report = {
        "dataset": DATASET,
        "citylearn_version": "2.5.0",
        "source_tag": "v2.5.0",
        "source_commit": "29062af6d077409e1c37a3e53a6cac30fd4d02bc",
        "license_path": str((REPO_ROOT / "LICENSE").relative_to(ROOT)).replace("\\", "/"),
        "schema_path": str(schema_path.relative_to(ROOT)).replace("\\", "/"),
        "building_count": len(buildings),
        "buildings": buildings,
        "file_count": len(files),
        "files": files,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
