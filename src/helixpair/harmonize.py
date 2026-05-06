from __future__ import annotations

import json
from pathlib import Path

from helixpair.io_utils import resolve_path, write_table
from helixpair.parsers import parse_cap_selex_batch_pairs, parse_cap_selex_curated_prey


def _read_seq_lines(path: str | Path, limit: int | None) -> list[str]:
    path = resolve_path(path)
    if not path.exists():
        return []
    sequences = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            sequence = line.strip().upper()
            if sequence:
                sequences.append(sequence)
    return sequences


def _tf_file_registry(curated, available_files: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    registry: dict[str, dict[str, list[str]]] = {}
    for row in curated.itertuples(index=False):
        left_tf = str(row.left_tf)
        right_tf = str(row.right_tf)
        left_baseline = f"{row.tf1_seq_prefix}0u_sig.seq" if str(row.tf1_seq_prefix) and str(row.tf1_seq_prefix).lower() != "nan" else ""
        right_baseline = f"{row.tf2_seq_prefix}0u_sig.seq" if str(row.tf2_seq_prefix) and str(row.tf2_seq_prefix).lower() != "nan" else ""
        left_treated = str(row.left_seq_file) if str(row.left_seq_file) and str(row.left_seq_file) != "nan" else ""
        right_treated = str(row.right_seq_file) if str(row.right_seq_file) and str(row.right_seq_file) != "nan" else ""
        for tf, baseline_file, treated_file in [
            (left_tf, left_baseline, left_treated),
            (right_tf, right_baseline, right_treated),
        ]:
            if not tf:
                continue
            info = registry.setdefault(tf, {"baseline_files": [], "treated_files": []})
            if baseline_file in available_files and baseline_file not in info["baseline_files"]:
                info["baseline_files"].append(baseline_file)
            if treated_file in available_files and treated_file not in info["treated_files"]:
                info["treated_files"].append(treated_file)
    return registry


def harmonize_cap_selex(
    project_root: str | Path,
    sample_limit_per_file: int | None = None,
    minimum_pair_edges: int = 1,
    minimum_pairs: int = 1,
):
    import pandas as pd

    project_root = resolve_path(project_root)
    intermediate_root = resolve_path(project_root / "data_intermediate")
    selex_root = resolve_path(project_root / "data_raw" / "cap_selex" / "relative_affinity" / "SELEX")
    ena_selex_root = resolve_path(project_root / "data_raw" / "cap_selex" / "ena" / "SELEX")
    curated_path = resolve_path(project_root / "data_raw" / "cap_selex" / "relative_affinity" / "data" / "Curated_Prey_Final.txt")
    secondary_pairs_path = resolve_path(project_root / "data_raw" / "cap_selex" / "relative_affinity" / "Batch_test.txt")
    ena_downloaded_inventory_path = resolve_path(project_root / "data_intermediate" / "cap_selex_ena_downloaded_inventory.tsv")

    curated = parse_cap_selex_curated_prey(curated_path)
    secondary_pairs = parse_cap_selex_batch_pairs(secondary_pairs_path) if secondary_pairs_path.exists() else pd.DataFrame()
    available_files = {path.name: path for path in selex_root.glob("*_sig.seq")}
    if ena_selex_root.exists():
        available_files.update({path.name: path for path in ena_selex_root.glob("*_sig.seq")})
    tf_registry = _tf_file_registry(curated, available_files)
    records = []
    sequence_records = []
    reference_edges = []

    for row in curated.itertuples(index=False):
        tf1_prefix = "" if pd.isna(row.tf1_seq_prefix) else str(row.tf1_seq_prefix)
        tf2_prefix = "" if pd.isna(row.tf2_seq_prefix) else str(row.tf2_seq_prefix)
        left_baseline_file = f"{tf1_prefix}0u_sig.seq" if tf1_prefix else ""
        right_baseline_file = f"{tf2_prefix}0u_sig.seq" if tf2_prefix else ""
        pair_files = [file_name for file_name in [row.left_seq_file, row.right_seq_file] if file_name]
        left_baseline_exists = left_baseline_file in available_files if left_baseline_file else False
        right_baseline_exists = right_baseline_file in available_files if right_baseline_file else False
        existing_pair_files = [file_name for file_name in pair_files if file_name in available_files]
        left_treated_exists = row.left_seq_file in available_files if row.left_seq_file else False
        right_treated_exists = row.right_seq_file in available_files if row.right_seq_file else False
        monomer_usable = (not row.is_pair) and (left_treated_exists or right_treated_exists) and (left_baseline_exists or right_baseline_exists)
        baseline_candidates = [
            file_name
            for file_name in [left_baseline_file, right_baseline_file, row.left_seq_file, row.right_seq_file]
            if file_name and file_name in available_files and file_name not in existing_pair_files
        ]
        pair_usable = bool(row.is_pair) and bool(existing_pair_files) and bool(baseline_candidates)
        pair_type = "heterodimer" if row.left_tf != row.right_tf else "homodimer"
        support_type = ""
        if pair_usable:
            if left_baseline_exists and right_baseline_exists:
                support_type = "paired_plus_both_baselines"
            elif left_baseline_exists:
                support_type = "paired_plus_left_baseline"
            elif right_baseline_exists:
                support_type = "paired_plus_right_baseline"
            else:
                support_type = "paired_plus_treated_proxy"
        elif monomer_usable:
            support_type = "monomer_treated_plus_baseline"
        records.append(
            {
                "pair_name": row.hgnc_pair,
                "left_tf": row.left_tf,
                "right_tf": row.right_tf,
                "is_pair": bool(row.is_pair),
                "pair_type": pair_type,
                "pair_file": ";".join(existing_pair_files),
                "left_baseline_file": left_baseline_file,
                "right_baseline_file": right_baseline_file,
                "pair_file_exists": bool(existing_pair_files),
                "left_baseline_exists": left_baseline_exists,
                "right_baseline_exists": right_baseline_exists,
                "monomer_usable": monomer_usable,
                "pair_usable": pair_usable,
                "support_type": support_type,
                "num_pair_files": len(existing_pair_files),
            }
        )

        if monomer_usable:
            monomer_positive = row.left_seq_file if left_treated_exists else row.right_seq_file
            monomer_negative = left_baseline_file if left_baseline_exists else right_baseline_file
            for label, role, file_name in [(1, "treated", monomer_positive), (0, "baseline", monomer_negative)]:
                for seq_index, sequence in enumerate(_read_seq_lines(available_files[file_name], sample_limit_per_file)):
                    sequence_records.append(
                        {
                            "seq_id": f"phase1::{row.left_tf}::{role}::{seq_index}",
                            "sequence": sequence,
                            "label": label,
                            "phase": "phase1",
                            "source_dataset": "CAP_SELEX",
                            "tf_label": row.left_tf,
                            "tf_pair_label": f"{row.left_tf}::{row.left_tf}",
                            "state_label": "in_vitro",
                            "element_type": "monomer",
                            "synthetic_flag": False,
                            "pair_name": row.hgnc_pair,
                            "role": role,
                            "sequence_file": file_name,
                        }
                    )

        if pair_usable:
            negative_file = baseline_candidates[0]
            reference_edges.append({"left_tf": row.left_tf, "right_tf": row.right_tf})
            positive_records = [(1, f"paired::{Path(file_name).stem}", file_name) for file_name in existing_pair_files]
            negative_records = [(0, "baseline", negative_file)]
            for label, role, file_name in positive_records + negative_records:
                for seq_index, sequence in enumerate(_read_seq_lines(available_files[file_name], sample_limit_per_file)):
                    sequence_records.append(
                        {
                            "seq_id": f"phase2::{row.left_tf}::{row.right_tf}::{role}::{seq_index}",
                            "sequence": sequence,
                            "label": label,
                            "phase": "phase2",
                            "source_dataset": "CAP_SELEX",
                            "tf_label": "",
                            "tf_pair_label": f"{row.left_tf}::{row.right_tf}",
                            "state_label": "in_vitro",
                            "element_type": "pair",
                            "synthetic_flag": False,
                            "pair_name": row.hgnc_pair,
                            "role": role,
                            "sequence_file": file_name,
                        }
                    )

    existing_pair_names = {str(record["pair_name"]) for record in records if record["pair_type"] in {"heterodimer", "homodimer"}}
    existing_pair_keys = {
        "::".join(sorted((str(record["left_tf"]), str(record["right_tf"]))))
        for record in records
        if record["pair_type"] in {"heterodimer", "homodimer"}
    }
    if not secondary_pairs.empty:
        for row in secondary_pairs.itertuples(index=False):
            if str(row.hgnc_pair) in existing_pair_names:
                continue
            pair_file_exists = str(row.pair_seq_file) in available_files
            left_info = tf_registry.get(str(row.left_tf), {"baseline_files": [], "treated_files": []})
            right_info = tf_registry.get(str(row.right_tf), {"baseline_files": [], "treated_files": []})
            left_baseline_file = left_info["baseline_files"][0] if left_info["baseline_files"] else ""
            right_baseline_file = right_info["baseline_files"][0] if right_info["baseline_files"] else ""
            baseline_candidates = [file_name for file_name in [left_baseline_file, right_baseline_file] if file_name]
            pair_usable = bool(pair_file_exists and baseline_candidates)
            support_type = ""
            if pair_usable:
                if left_baseline_file and right_baseline_file:
                    support_type = "secondary_pair_plus_both_baselines"
                elif left_baseline_file:
                    support_type = "secondary_pair_plus_left_baseline"
                else:
                    support_type = "secondary_pair_plus_right_baseline"
            records.append(
                {
                    "pair_name": row.hgnc_pair,
                    "left_tf": row.left_tf,
                    "right_tf": row.right_tf,
                    "is_pair": True,
                    "pair_type": "heterodimer" if row.left_tf != row.right_tf else "homodimer",
                    "pair_file": row.pair_seq_file if pair_file_exists else "",
                    "left_baseline_file": left_baseline_file,
                    "right_baseline_file": right_baseline_file,
                    "pair_file_exists": pair_file_exists,
                    "left_baseline_exists": bool(left_baseline_file),
                    "right_baseline_exists": bool(right_baseline_file),
                    "monomer_usable": False,
                    "pair_usable": pair_usable,
                    "support_type": support_type,
                    "num_pair_files": int(pair_file_exists),
                }
            )
            if pair_usable:
                negative_file = baseline_candidates[0]
                reference_edges.append({"left_tf": row.left_tf, "right_tf": row.right_tf})
                for label, role, file_name in [
                    (1, f"paired::{Path(row.pair_seq_file).stem}", row.pair_seq_file),
                    (0, "baseline", negative_file),
                ]:
                    for seq_index, sequence in enumerate(_read_seq_lines(available_files[file_name], sample_limit_per_file)):
                        sequence_records.append(
                            {
                                "seq_id": f"phase2::{row.left_tf}::{row.right_tf}::{role}::{seq_index}",
                                "sequence": sequence,
                                "label": label,
                                "phase": "phase2",
                                "source_dataset": "CAP_SELEX",
                                "tf_label": "",
                                "tf_pair_label": f"{row.left_tf}::{row.right_tf}",
                                "state_label": "in_vitro",
                                "element_type": "pair",
                                "synthetic_flag": False,
                                "pair_name": row.hgnc_pair,
                                "role": role,
                                "sequence_file": file_name,
                            }
                        )

    if ena_downloaded_inventory_path.exists():
        ena_downloaded = pd.read_csv(ena_downloaded_inventory_path, sep="\t")
        for row in ena_downloaded.itertuples(index=False):
            pair_name = str(getattr(row, "pair_name", ""))
            pair_key = "::".join(sorted((str(row.left_tf), str(row.right_tf))))
            if pair_name in existing_pair_names or pair_key in existing_pair_keys:
                continue
            pair_file = str(getattr(row, "sig_file_name", ""))
            pair_file_exists = pair_file in available_files
            left_info = tf_registry.get(str(row.left_tf), {"baseline_files": [], "treated_files": []})
            right_info = tf_registry.get(str(row.right_tf), {"baseline_files": [], "treated_files": []})
            left_baseline_file = left_info["baseline_files"][0] if left_info["baseline_files"] else ""
            right_baseline_file = right_info["baseline_files"][0] if right_info["baseline_files"] else ""
            baseline_candidates = [
                file_name
                for file_name in [
                    left_baseline_file,
                    right_baseline_file,
                    *(left_info["treated_files"][:1]),
                    *(right_info["treated_files"][:1]),
                ]
                if file_name and file_name in available_files and file_name != pair_file
            ]
            pair_usable = bool(pair_file_exists and baseline_candidates)
            if left_baseline_file and right_baseline_file:
                support_type = "ena_pair_plus_both_baselines"
            elif left_baseline_file or right_baseline_file:
                support_type = "ena_pair_plus_one_baseline"
            elif baseline_candidates:
                support_type = "ena_pair_plus_treated_proxy"
            else:
                support_type = ""
            records.append(
                {
                    "pair_name": pair_name,
                    "left_tf": row.left_tf,
                    "right_tf": row.right_tf,
                    "is_pair": True,
                    "pair_type": "heterodimer" if row.left_tf != row.right_tf else "homodimer",
                    "pair_file": pair_file if pair_file_exists else "",
                    "left_baseline_file": left_baseline_file,
                    "right_baseline_file": right_baseline_file,
                    "pair_file_exists": pair_file_exists,
                    "left_baseline_exists": bool(left_baseline_file),
                    "right_baseline_exists": bool(right_baseline_file),
                    "monomer_usable": False,
                    "pair_usable": pair_usable,
                    "support_type": support_type,
                    "num_pair_files": int(pair_file_exists),
                }
            )
            if pair_usable:
                negative_file = baseline_candidates[0]
                reference_edges.append({"left_tf": row.left_tf, "right_tf": row.right_tf})
                for label, role, file_name in [
                    (1, f"paired::{Path(pair_file).stem}", pair_file),
                    (0, "baseline", negative_file),
                ]:
                    for seq_index, sequence in enumerate(_read_seq_lines(available_files[file_name], sample_limit_per_file)):
                        sequence_records.append(
                            {
                                "seq_id": f"phase2::{row.left_tf}::{row.right_tf}::{role}::{seq_index}",
                                "sequence": sequence,
                                "label": label,
                                "phase": "phase2",
                                "source_dataset": "CAP_SELEX_ENA",
                                "tf_label": "",
                                "tf_pair_label": f"{row.left_tf}::{row.right_tf}",
                                "state_label": "in_vitro",
                                "element_type": "pair",
                                "synthetic_flag": False,
                                "pair_name": pair_name,
                                "role": role,
                                "sequence_file": file_name,
                            }
                        )

    inventory = pd.DataFrame.from_records(records)
    sequence_frame = pd.DataFrame.from_records(sequence_records)
    usable_inventory = inventory[(inventory["monomer_usable"]) | (inventory["pair_usable"])].copy()
    reference_graph = pd.DataFrame.from_records(reference_edges).drop_duplicates()

    write_table(inventory, intermediate_root / "cap_selex_pair_inventory.tsv")
    write_table(usable_inventory, intermediate_root / "cap_selex_pair_inventory_usable.tsv")
    write_table(sequence_frame, intermediate_root / "cap_selex_sequences.parquet")
    write_table(reference_graph, intermediate_root / "reference_graph.parquet")

    reports_root = resolve_path(project_root / "reports")
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "cap_selex_harmonization.md"
    report_path.write_text(
        "\n".join(
            [
                "# CAP-SELEX Harmonization",
                "",
                f"- Total `*_sig.seq` files discovered: {len(available_files)}",
                f"- Total curated rows: {len(curated)}",
                f"- Secondary pair rows: {len(secondary_pairs)}",
                f"- Usable monomer rows: {int(inventory['monomer_usable'].sum())}",
                f"- Usable pair rows: {int(inventory['pair_usable'].sum())}",
                f"- Sampled phase1 sequences: {int((sequence_frame['phase'] == 'phase1').sum()) if not sequence_frame.empty else 0}",
                f"- Sampled phase2 sequences: {int((sequence_frame['phase'] == 'phase2').sum()) if not sequence_frame.empty else 0}",
                f"- Sample limit per file: {'all' if sample_limit_per_file is None else sample_limit_per_file}",
                "",
                "Usability rule: keep pair rows with a paired CAP-SELEX file plus at least one recoverable monomer baseline; keep monomer rows with treated and 0u baseline files.",
            ]
        ),
        encoding="utf-8",
    )
    acceptance_root = resolve_path(project_root / "reports" / "data_acceptance")
    acceptance_root.mkdir(parents=True, exist_ok=True)
    usable_pair_rows = int(inventory["pair_usable"].sum())
    reference_edge_count = int(len(reference_graph))
    acceptance_status = "ready" if usable_pair_rows >= minimum_pairs and reference_edge_count >= minimum_pair_edges else "insufficient"
    (acceptance_root / "cap_selex_acceptance.json").write_text(
        json.dumps(
            {
                "dataset": "cap_selex",
                "status": acceptance_status,
                "seq_file_count": len(available_files),
                "usable_pair_rows": usable_pair_rows,
                "usable_monomer_rows": int(inventory["monomer_usable"].sum()),
                "reference_edges": reference_edge_count,
                "secondary_pair_rows": int(len(secondary_pairs)),
                "sample_limit_per_file": sample_limit_per_file,
                "minimum_pair_edges": int(minimum_pair_edges),
                "minimum_pairs": int(minimum_pairs),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return usable_inventory, sequence_frame
