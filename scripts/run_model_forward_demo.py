from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helixpair.model import HelixPairModel


def _ids(batch: int, offset: int = 0) -> dict[str, torch.Tensor]:
    base = torch.arange(batch, dtype=torch.long)
    return {
        "family_id": (base + offset) % 3,
        "subfamily_id": (base + offset) % 5,
        "paralog_id": (base + offset) % 7,
        "tf_id": (base + offset) % 11,
    }


def run_forward_demo(out_dir: Path | None = None) -> Path:
    torch.manual_seed(11)
    batch = 4
    model = HelixPairModel(
        num_families=3,
        num_subfamilies=5,
        num_paralogs=7,
        num_tfs=11,
        geometry_dim=19,
        availability_dim=32,
        state_dim=32,
        sequence_channels=9,
        interface_channels=13,
        embedding_dim=32,
        rank=8,
        num_gap_bins=25,
        helical_order=2,
        use_hierarchy_embedding=True,
        use_anchor_refinement=True,
        use_geometry_head=True,
        use_bridge_head=True,
        use_state_gate=True,
        use_partition=True,
        use_availability=True,
        disable_helical_basis=False,
    )
    model.eval()

    left_anchor_offsets = torch.randn(batch, 5, 9, 24)
    right_anchor_offsets = torch.randn(batch, 5, 9, 24)
    geometry_features = torch.randn(batch, 19)
    gap = torch.linspace(0.0, 18.0, batch)
    geometry_features[:, 7] = torch.sin(2.0 * torch.pi * gap / 10.5)
    geometry_features[:, 8] = torch.cos(2.0 * torch.pi * gap / 10.5)
    interface_tensor = torch.randn(batch, 13, 32)
    availability = torch.rand(batch, 32)
    state_context = torch.randn(batch, 32)

    with torch.no_grad():
        outputs = model(
            left_anchor_offsets=left_anchor_offsets,
            right_anchor_offsets=right_anchor_offsets,
            left_ids=_ids(batch, 0),
            right_ids=_ids(batch, 3),
            geometry_features=geometry_features,
            interface_tensor=interface_tensor,
            availability=availability,
            state_context=state_context,
            compatibility=torch.ones(batch),
        )

    target = out_dir or (ROOT / "outputs")
    target.mkdir(exist_ok=True)
    out_path = target / "full_model_forward_demo.csv"
    pd.DataFrame(
        {
            "example": list(range(batch)),
            "usage_probability": outputs.usage_probability.numpy(),
            "availability_only_probability": outputs.availability_only_probability.numpy(),
            "geometry_residual": outputs.geometry_residual.numpy(),
            "bridge_residual": outputs.bridge_residual.numpy(),
            "state_gate": outputs.state_gate.numpy(),
            "state_correction": outputs.state_correction.numpy(),
            "cooperative_gain": outputs.cooperative_gain.numpy(),
        }
    ).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(run_forward_demo())
