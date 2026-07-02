from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ac4mpc.sb3_wrappers import export_sb3_sac_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SB3 SAC actor/critic weights to NumPy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = export_sb3_sac_weights(args.model, args.out)
    print(output)


if __name__ == "__main__":
    main()
