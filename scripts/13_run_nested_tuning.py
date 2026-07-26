from __future__ import annotations

import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STUDY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adsorption_reliability_study.nested_tuning import run_nested_tuning


if __name__ == "__main__":
    print(run_nested_tuning())
