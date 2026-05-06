from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = [11, 17, 23, 29, 47]
DNA_ALPHABET = "ACGTN"
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}
ORIENTATION_TO_INDEX = {"++": 0, "+-": 1, "-+": 2, "--": 3}
INDEX_TO_ORIENTATION = {value: key for key, value in ORIENTATION_TO_INDEX.items()}
DEFAULT_HELICAL_PERIOD = 10.5
DEFAULT_WINDOW = 96
DEFAULT_ANCHOR_WINDOW = 24
DEFAULT_INTERFACE_FLANK = 4
DEFAULT_OFFSETS = (-2, -1, 0, 1, 2)
DEFAULT_PHASES = ("phase1", "phase2", "phase3", "phase4", "phase5")
DEFAULT_SCENARIOS = ("synthetic", "ablation", "real")
DEFAULT_GAP_BINS = tuple(range(-4, 21))
CANONICAL_REAL_SPLITS = ("default", "unseen_pair", "unseen_state")
PHASE4_REAL_EVAL_SPLITS = (
    "default",
    "unseen_pair",
    "unseen_state",
    "unseen_family",
    "unseen_subfamily",
    "sequence_locus",
)
