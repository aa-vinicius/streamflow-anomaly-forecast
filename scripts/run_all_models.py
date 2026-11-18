from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ten frozen outlet-lead workflows.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    script = Path(__file__).with_name("train_hybrid_v2.py")
    for outlet in ("19094", "19097"):
        for lead in range(1, 6):
            command = [
                sys.executable,
                str(script),
                "--outlet", outlet,
                "--h", str(lead),
                "--data-root", str(args.data_root),
                "--output-root", str(args.output_root),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

