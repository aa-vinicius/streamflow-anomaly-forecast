from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".joblib", ".pkl", ".pickle"}
WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")


def main() -> None:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden data/model format: {relative}")
        if "data" not in relative.parts and path.suffix.lower() in {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".cff"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if WINDOWS_PATH.search(text):
                problems.append(f"absolute Windows path: {relative}")
    if problems:
        print("Repository validation FAILED:")
        print("\n".join(f"- {item}" for item in problems))
        sys.exit(1)
    print("Repository validation passed: no fitted-model binaries or absolute local paths detected.")



if __name__ == "__main__":
    main()
