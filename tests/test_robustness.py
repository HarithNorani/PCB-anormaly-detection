from pathlib import Path

import pytest

from pcb_inspection.robustness import run_robustness


def test_run_robustness_rejects_unknown_condition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown robustness condition"):
        run_robustness(tmp_path, conditions=("unknown",))
