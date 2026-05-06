# HelixPair NeurIPS 2026 Full Clean Code Package

This package contains the clean HelixPair method code corresponding to
`HelixPair_NeurIPS2026_reviewed.docx`.

Included:
- Full HelixPair method implementation: data construction, feature extraction, split construction,
  tensor-bundle building, model architecture, losses, staged training, inference, metrics, audits,
  robustness utilities, public-state utilities, and result summarization.
- Selected training and demo entrypoints under `scripts/`.
- Clean configuration files for the HelixPair method training stages.
- Locked paper-result CSV files used by the selected demo.

Excluded:
- External comparator implementations and probe code.
- Historical search scripts, optional ablation runners, checkpoints, caches, manuscript drafts, and
  bulky raw/intermediate datasets.

Quick checks:

```bash
python scripts/run_selected_experiment_demo.py
python scripts/run_model_forward_demo.py
```

Training entrypoints:

```bash
# Train one staged phase when the required tensor bundles are present.
python scripts/train_helixpair_phase.py --config configs/phase2_pair/default.yaml --phase phase2

# Run the selected hematopoietic phase4/phase5 multiseed training plan when full data bundles are present.
python scripts/run_phase45_multiseed_clean.py --inventory-only
```

This package includes a cleaned mini processed dataset under
`data_processed/real/phase2/default/`, plus a phase1 warm-start checkpoint, so the phase2 command above
runs without CAP-SELEX FASTQ files. These mini bundles are for code-path verification, not for reproducing
paper-scale AUPRC values. Paper-scale training requires the full processed bundles or locally rebuilt inputs
from the raw resources.
