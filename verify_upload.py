import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLOCKED_DIRS = {"data", "artifacts", "results", "outputs", "checkpoints", "models", "paper", ".git", ".venv", ".venv-energy", ".venv_energy", "__pycache__"}
BLOCKED_SUFFIXES = {".csv", ".parquet", ".feather", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".onnx", ".log", ".zip", ".7z", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".docx"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".mplstyle", ""}
PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.I),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/", re.I),
    "workspace username": re.compile(r"\bADMIN\b"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def main():
    problems = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in BLOCKED_DIRS for part in relative.parts):
            problems.append(f"blocked directory: {relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"blocked file type: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            problems.append(f"unexpected file type: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    problems.append(f"{label}: {relative}")
            if path.suffix.lower() == ".py":
                try:
                    ast.parse(text, filename=str(relative))
                except SyntaxError as error:
                    problems.append(f"Python syntax: {relative}: {error}")
    if problems:
        raise SystemExit("release check failed:\n" + "\n".join(sorted(set(problems))))
    print(f"release check passed: {len(files)} code/documentation files; no blocked content found")


if __name__ == "__main__":
    main()
