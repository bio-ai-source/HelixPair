from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from helixpair.constants import (
    DEFAULT_ANCHOR_WINDOW,
    DEFAULT_HELICAL_PERIOD,
    DEFAULT_INTERFACE_FLANK,
    DEFAULT_OFFSETS,
    DNA_ALPHABET,
    DNA_TO_INDEX,
    ORIENTATION_TO_INDEX,
)
from helixpair.types import AnchorRecord, PairRecord

ROLL_LOOKUP = {
    "AA": -0.6, "AC": -0.2, "AG": 0.1, "AT": 0.4,
    "CA": -0.1, "CC": -0.3, "CG": 0.8, "CT": 0.2,
    "GA": 0.2, "GC": 0.7, "GG": -0.2, "GT": -0.1,
    "TA": 0.5, "TC": 0.0, "TG": -0.1, "TT": -0.5,
}
HELT_LOOKUP = {
    "AA": -0.3, "AC": 0.1, "AG": 0.3, "AT": 0.2,
    "CA": 0.1, "CC": -0.4, "CG": 0.6, "CT": 0.0,
    "GA": 0.2, "GC": 0.7, "GG": -0.5, "GT": 0.1,
    "TA": 0.3, "TC": 0.0, "TG": 0.1, "TT": -0.2,
}
_ASCII_TO_INDEX = np.full(256, DNA_TO_INDEX["N"], dtype=np.int64)
for _base, _index in DNA_TO_INDEX.items():
    _ASCII_TO_INDEX[ord(_base.upper())] = int(_index)
    _ASCII_TO_INDEX[ord(_base.lower())] = int(_index)
_REVERSE_COMPLEMENT_ROWS = np.asarray(
    [
        DNA_TO_INDEX["T"],
        DNA_TO_INDEX["G"],
        DNA_TO_INDEX["C"],
        DNA_TO_INDEX["A"],
    ],
    dtype=np.int64,
)


def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(table)[::-1]


def one_hot_encode(sequence: str) -> np.ndarray:
    tokens = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    indices = _ASCII_TO_INDEX[tokens]
    encoded = np.zeros((len(DNA_ALPHABET), len(indices)), dtype=np.float32)
    encoded[indices, np.arange(len(indices), dtype=np.int64)] = 1.0
    return encoded


def random_background(length: int, rng: np.random.Generator, gc: float = 0.5) -> str:
    at = (1.0 - gc) / 2.0
    gc_prob = gc / 2.0
    alphabet = np.array(list("ACGT"))
    probs = np.array([at, gc_prob, gc_prob, at], dtype=np.float64)
    return "".join(rng.choice(alphabet, size=length, p=probs))


def embed_sequence(sequence: str, insert: str, start: int) -> str:
    if start < 0 or start + len(insert) > len(sequence):
        raise ValueError("insert out of bounds")
    return f"{sequence[:start]}{insert}{sequence[start + len(insert):]}"


def pfm_to_pwm(pfm: np.ndarray, pseudocount: float = 1e-3, background: float = 0.25) -> np.ndarray:
    normalized = (pfm + pseudocount) / (pfm + pseudocount).sum(axis=0, keepdims=True)
    return np.log2(normalized / background)


def _prepare_scan_tracks(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    forward = one_hot_encode(sequence)[:4]
    reverse = forward[_REVERSE_COMPLEMENT_ROWS, ::-1]
    return forward, reverse


def _score_pwm_track(track: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    motif_len = int(pwm.shape[1])
    if track.shape[1] < motif_len:
        return np.zeros((0,), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(track, motif_len, axis=1)
    return np.einsum("bpm,bm->p", windows, pwm, optimize=True).astype(np.float32, copy=False)


def _scan_pwm_with_tracks(
    sequence: str,
    pwm: np.ndarray,
    tf_id: str,
    top_k: int,
    cutoff: float,
    *,
    forward_track: np.ndarray,
    reverse_track: np.ndarray,
    score_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[AnchorRecord]:
    motif_len = int(pwm.shape[1])
    if top_k <= 0 or len(sequence) < motif_len:
        return []
    if score_cache is not None and motif_len in score_cache:
        forward_scores, reverse_scores = score_cache[motif_len]
    else:
        forward_scores = _score_pwm_track(forward_track, pwm)
        reverse_scores = _score_pwm_track(reverse_track, pwm)
        if score_cache is not None:
            score_cache[motif_len] = (forward_scores, reverse_scores)
    if forward_scores.size == 0:
        return []

    reverse_scores = reverse_scores[::-1]
    forward_starts = np.flatnonzero(forward_scores >= cutoff)
    reverse_starts = np.flatnonzero(reverse_scores >= cutoff)
    if forward_starts.size == 0 and reverse_starts.size == 0:
        return []

    starts = np.concatenate([forward_starts, reverse_starts], axis=0)
    strands = np.concatenate(
        [
            np.full(forward_starts.shape, "+", dtype=object),
            np.full(reverse_starts.shape, "-", dtype=object),
        ],
        axis=0,
    )
    scores = np.concatenate([forward_scores[forward_starts], reverse_scores[reverse_starts]], axis=0)
    stable_order = np.concatenate([2 * forward_starts, 2 * reverse_starts + 1], axis=0)
    ranked = np.lexsort((stable_order, -scores))[: int(top_k)]
    anchors: list[AnchorRecord] = []
    for index in ranked.tolist():
        start = int(starts[index])
        anchors.append(
            AnchorRecord(
                sequence,
                tf_id,
                start,
                start + motif_len,
                str(strands[index]),
                float(scores[index]),
                2,
            )
        )
    return anchors


def scan_pwm(sequence: str, pwm: np.ndarray, tf_id: str, top_k: int, cutoff: float) -> list[AnchorRecord]:
    forward_track, reverse_track = _prepare_scan_tracks(sequence)
    return _scan_pwm_with_tracks(
        sequence,
        pwm,
        tf_id,
        top_k,
        cutoff,
        forward_track=forward_track,
        reverse_track=reverse_track,
    )


def local_offset_softmin(energies: Iterable[float], temperature: float = 1.0) -> float:
    values = np.asarray(list(energies), dtype=np.float32)
    if values.size == 0:
        return float("nan")
    scaled = -values / max(temperature, 1e-6)
    max_term = scaled.max()
    return float(-temperature * (math.log(np.exp(scaled - max_term).sum()) + max_term))


def offset_slice(sequence: str, center: int, width: int) -> str:
    left = center - width // 2
    right = left + width
    prefix = ""
    suffix = ""
    if left < 0:
        prefix = "N" * abs(left)
        left = 0
    if right > len(sequence):
        suffix = "N" * (right - len(sequence))
        right = len(sequence)
    return prefix + sequence[left:right] + suffix


def approximate_shape_channels(sequence: str, window: int = 5) -> np.ndarray:
    sequence = sequence.upper()
    length = len(sequence)
    encoded = one_hot_encode(sequence)[:4]
    purine = np.isin(np.array(list(sequence)), np.array(list("AG"))).astype(np.float32)
    pyrimidine = np.isin(np.array(list(sequence)), np.array(list("CT"))).astype(np.float32)
    mgw = np.zeros((length,), dtype=np.float32)
    prot = np.zeros((length,), dtype=np.float32)
    roll = np.zeros((length,), dtype=np.float32)
    helt = np.zeros((length,), dtype=np.float32)
    radius = max(window // 2, 1)
    for index in range(length):
        left = max(0, index - radius)
        right = min(length, index + radius + 1)
        local = encoded[:, left:right]
        at_fraction = float(local[[0, 3]].sum() / max(local.sum(), 1.0))
        gc_fraction = float(local[[1, 2]].sum() / max(local.sum(), 1.0))
        mgw[index] = at_fraction
        helt[index] = gc_fraction
        if index < length - 1:
            dinuc = sequence[index : index + 2]
            roll[index] = float(ROLL_LOOKUP.get(dinuc, 0.0))
            if dinuc[0] in "CT" and dinuc[1] in "AG":
                prot[index] = 1.0
            elif dinuc[0] in "AG" and dinuc[1] in "CT":
                prot[index] = -1.0
        else:
            prot[index] = prot[index - 1] if index > 0 else 0.0
            roll[index] = roll[index - 1] if index > 0 else 0.0
    shape = np.stack([mgw, roll, prot, helt], axis=0)
    mean = shape.mean(axis=1, keepdims=True)
    std = shape.std(axis=1, keepdims=True)
    return (shape - mean) / np.clip(std, 1e-6, None)


def build_offset_anchor_tensor(
    sequence: str,
    start: int,
    end: int,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
    anchor_window: int = DEFAULT_ANCHOR_WINDOW,
    include_shape_channels: bool = True,
) -> np.ndarray:
    center = (start + end) // 2
    windows = []
    for offset in offsets:
        anchor_sequence = offset_slice(sequence, center + int(offset), anchor_window)
        one_hot = one_hot_encode(anchor_sequence)
        if include_shape_channels:
            one_hot = np.concatenate([one_hot, approximate_shape_channels(anchor_sequence)], axis=0)
        windows.append(one_hot)
    return np.stack(windows, axis=0)


def helical_basis(distance: np.ndarray | float, order: int = 2, period: float = DEFAULT_HELICAL_PERIOD) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float32)
    features = []
    for harmonic in range(1, order + 1):
        angle = 2.0 * np.pi * harmonic * distance / period
        features.append(np.sin(angle))
        features.append(np.cos(angle))
    return np.stack(features, axis=-1)


def spline_basis(distance: np.ndarray | float, bins: int = 8, minimum: float = -4.0, maximum: float = 20.0) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float32)
    knots = np.linspace(minimum, maximum, bins, dtype=np.float32)
    scale = max((maximum - minimum) / bins, 1e-6)
    features = [np.maximum(1.0 - np.abs(distance - knot) / scale, 0.0) for knot in knots]
    return np.stack(features, axis=-1)


def geometry_features(center_distance: float, edge_gap: float, overlap_len: float, orientation: str, order: int = 2, bins: int = 8) -> np.ndarray:
    distance = np.asarray([center_distance], dtype=np.float32)
    gap = np.asarray([edge_gap], dtype=np.float32)
    overlap = np.asarray([overlap_len], dtype=np.float32)
    orientation_one_hot = np.zeros((1, len(ORIENTATION_TO_INDEX)), dtype=np.float32)
    orientation_one_hot[0, ORIENTATION_TO_INDEX[orientation]] = 1.0
    helical = helical_basis(distance, order=order)
    spline = spline_basis(distance, bins=bins)
    return np.concatenate(
        [
            distance[:, None],
            gap[:, None],
            overlap[:, None],
            orientation_one_hot,
            helical,
            spline,
        ],
        axis=-1,
    )[0]


def make_interface_mask(sequence_length: int, left_start: int, left_end: int, right_start: int, right_end: int, flank: int = 4) -> np.ndarray:
    mask = np.zeros((4, sequence_length), dtype=np.float32)
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    gap_start = min(left_end, right_end)
    gap_end = max(left_start, right_start)
    inner_left_start = max(left_end - flank, left_start)
    inner_left_end = left_end
    inner_right_start = right_start
    inner_right_end = min(right_start + flank, right_end)
    if overlap_end > overlap_start:
        mask[0, overlap_start:overlap_end] = 1.0
    if gap_end > gap_start:
        mask[1, gap_start:gap_end] = 1.0
    mask[2, inner_left_start:inner_left_end] = 1.0
    mask[3, inner_right_start:inner_right_end] = 1.0
    return mask


def build_interface_tensor(
    sequence: str,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
    shape_channels: np.ndarray | None = None,
    flank: int = DEFAULT_INTERFACE_FLANK,
    use_shape_channels: bool = True,
    mask_exclusive_anchor_core: bool = True,
) -> np.ndarray:
    one_hot = one_hot_encode(sequence)
    masked = one_hot.copy()
    if mask_exclusive_anchor_core:
        if right_start >= left_end:
            masked[:, left_start:left_end] = 0.0
            masked[:, right_start:right_end] = 0.0
        else:
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            masked[:, left_start:overlap_start] = 0.0
            masked[:, overlap_end:right_end] = 0.0
    region_mask = make_interface_mask(len(sequence), left_start, left_end, right_start, right_end, flank=flank)
    if shape_channels is None and use_shape_channels:
        shape_channels = approximate_shape_channels(sequence)
    if shape_channels is None:
        shape_channels = np.zeros((0, len(sequence)), dtype=np.float32)
    return np.concatenate([masked, shape_channels.astype(np.float32), region_mask], axis=0)


def interface_bounds(left_start: int, left_end: int, right_start: int, right_end: int, flank: int = DEFAULT_INTERFACE_FLANK) -> tuple[int, int]:
    interface_start = max(min(left_end, right_start) - flank, 0)
    interface_end = max(left_end, right_start) + flank
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    if overlap_end > overlap_start:
        interface_start = max(overlap_start - flank, 0)
        interface_end = overlap_end + flank
    return interface_start, interface_end


def _edge_counts(sequence: str) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    counts: dict[str, dict[str, int]] = {base: {other: 0 for other in DNA_ALPHABET[:4]} for base in DNA_ALPHABET[:4]}
    indegree = {base: 0 for base in DNA_ALPHABET[:4]}
    outdegree = {base: 0 for base in DNA_ALPHABET[:4]}
    for current, nxt in zip(sequence, sequence[1:]):
        if current not in counts or nxt not in counts:
            continue
        counts[current][nxt] += 1
        outdegree[current] += 1
        indegree[nxt] += 1
    return counts, indegree, outdegree


def _remaining_nodes(counts: dict[str, dict[str, int]]) -> set[str]:
    nodes = set()
    for left, targets in counts.items():
        if any(value > 0 for value in targets.values()):
            nodes.add(left)
        for right, value in targets.items():
            if value > 0:
                nodes.add(right)
    return nodes


def _connected_undirected(counts: dict[str, dict[str, int]], current: str, end: str) -> bool:
    nodes = _remaining_nodes(counts) | {current, end}
    if not nodes:
        return True
    start = next(iter(nodes))
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        neighbors = {other for other, value in counts.get(node, {}).items() if value > 0}
        neighbors |= {left for left, targets in counts.items() if targets.get(node, 0) > 0}
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return nodes.issubset(seen)


def _has_eulerian_completion(counts: dict[str, dict[str, int]], current: str, end: str) -> bool:
    indegree = {base: 0 for base in DNA_ALPHABET[:4]}
    outdegree = {base: 0 for base in DNA_ALPHABET[:4]}
    total_edges = 0
    for left, targets in counts.items():
        for right, value in targets.items():
            if value <= 0:
                continue
            outdegree[left] += value
            indegree[right] += value
            total_edges += value
    if total_edges == 0:
        return current == end
    if not _connected_undirected(counts, current, end):
        return False
    for node in DNA_ALPHABET[:4]:
        diff = outdegree[node] - indegree[node]
        if node == current and node == end:
            if diff != 0:
                return False
        elif node == current:
            if diff != 1:
                return False
        elif node == end:
            if diff != -1:
                return False
        elif diff != 0:
            return False
    return True


def dinucleotide_shuffle(sequence: str, rng: np.random.Generator | None = None) -> str:
    if len(sequence) < 3 or any(base not in "ACGT" for base in sequence):
        return sequence
    rng = rng or np.random.default_rng()
    counts, _, _ = _edge_counts(sequence)
    current = sequence[0]
    end = sequence[-1]
    shuffled = [current]
    for _ in range(len(sequence) - 1):
        candidates = [base for base, value in counts[current].items() if value > 0]
        rng.shuffle(candidates)
        chosen = None
        for candidate in candidates:
            counts[current][candidate] -= 1
            if _has_eulerian_completion(counts, candidate, end):
                chosen = candidate
                break
            counts[current][candidate] += 1
        if chosen is None:
            return sequence
        shuffled.append(chosen)
        current = chosen
    return "".join(shuffled)


def shuffle_interface_region(
    sequence: str,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
    flank: int = DEFAULT_INTERFACE_FLANK,
    rng: np.random.Generator | None = None,
) -> str:
    start, end = interface_bounds(left_start, left_end, right_start, right_end, flank=flank)
    start = max(0, start)
    end = min(len(sequence), end)
    region = sequence[start:end]
    if not region:
        return sequence
    shuffled = dinucleotide_shuffle(region, rng=rng)
    return sequence[:start] + shuffled + sequence[end:]


def rerandomize_exclusive_cores(
    sequence: str,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
    rng: np.random.Generator | None = None,
) -> str:
    rng = rng or np.random.default_rng()
    chars = list(sequence)
    left_overlap = max(left_start, right_start)
    right_overlap = min(left_end, right_end)
    left_exclusive = list(range(left_start, min(left_end, left_overlap)))
    right_exclusive = list(range(max(right_start, right_overlap), right_end))
    for index in left_exclusive + right_exclusive:
        chars[index] = str(rng.choice(np.array(list("ACGT"))))
    return "".join(chars)


def build_pair_record(seq_id: str, left: AnchorRecord, right: AnchorRecord) -> PairRecord:
    left_center = 0.5 * (left.start + left.end)
    right_center = 0.5 * (right.start + right.end)
    return PairRecord(
        seq_id=seq_id,
        left_tf=left.tf_id,
        right_tf=right.tf_id,
        left_start=left.start,
        left_end=left.end,
        right_start=right.start,
        right_end=right.end,
        orientation=f"{left.strand}{right.strand}",
        center_distance=float(right_center - left_center),
        edge_gap=float(right.start - left.end),
        overlap_len=float(max(0, min(left.end, right.end) - max(left.start, right.start))),
        coarse_additive_score=float(left.score + right.score),
    )
