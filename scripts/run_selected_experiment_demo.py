from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_model_forward_demo import run_forward_demo


def _check_close(name: str, observed: float, expected: float, tolerance: float = 0.001) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(f"{name}: observed={observed:.6f}, expected={expected:.6f}")


def main() -> None:
    data_dir = ROOT / "data" / "paper_results"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    primary = pd.read_csv(data_dir / "helixpair_primary_metrics.csv")
    display = primary[["section", "slice", "metric", "helixpair_value", "paper_display", "source"]].copy()
    display.to_csv(out_dir / "selected_paper_metric_check.csv", index=False)

    expected = {
        ("biochemical", "held_out_tf_pairs", "auprc"): 0.718,
        ("biochemical", "held_out_family_combinations", "auprc"): 0.956,
        ("deployment", "staged_standard", "auprc"): 0.827,
        ("deployment", "staged_unseen_pair", "auprc"): 0.718,
        ("deployment", "staged_unseen_state", "auprc"): 0.763,
        ("reporter", "lentimpra_k562", "auprc"): 0.836,
        ("reporter", "lentimpra_hepg2", "auprc"): 0.893,
    }
    indexed = primary.set_index(["section", "slice", "metric"])
    for key, value in expected.items():
        _check_close(" / ".join(key), indexed.loc[key, "paper_display"], value)

    forward_path = run_forward_demo(out_dir)
    print("HelixPair selected full-method demo completed.")
    print(f"Metric check: {out_dir / 'selected_paper_metric_check.csv'}")
    print(f"Forward-pass output: {forward_path}")
    for key in expected:
        print(f"{key[0]} | {key[1]} | {key[2]} = {indexed.loc[key, 'paper_display']:.3f}")


if __name__ == "__main__":
    main()
