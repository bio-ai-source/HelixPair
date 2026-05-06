from __future__ import annotations

import json
import re
from pathlib import Path

from helixpair.io_utils import read_table, resolve_path, write_table


def parse_cap_selex_curated_prey(prey_path: str | Path):
    import pandas as pd

    prey_path = resolve_path(prey_path)
    frame = pd.read_csv(prey_path, sep="\t")
    frame.columns = [
        "hgnc_pair",
        "tf1_seq_prefix",
        "tf1_plate",
        "tf1_round",
        "tf2_seq_prefix",
        "tf2_plate",
        "tf2_round",
    ]
    frame["hgnc_pair"] = frame["hgnc_pair"].astype(str)
    frame["is_pair"] = frame["hgnc_pair"].str.contains("_")
    frame["left_tf"] = frame["hgnc_pair"].str.split("_").str[0]
    frame["right_tf"] = frame["hgnc_pair"].str.split("_").str[1].fillna(frame["left_tf"])
    frame["left_seq_file"] = frame["tf1_seq_prefix"].fillna("") + frame["tf1_plate"].fillna("") + frame["tf1_round"].fillna("").astype(str) + "_sig.seq"
    has_right = frame["tf2_seq_prefix"].fillna("").astype(str).str.len() > 0
    frame["right_seq_file"] = ""
    frame.loc[has_right, "right_seq_file"] = (
        frame.loc[has_right, "tf2_seq_prefix"].fillna("")
        + frame.loc[has_right, "tf2_plate"].fillna("")
        + frame.loc[has_right, "tf2_round"].fillna("").astype(str)
        + "_sig.seq"
    )
    return frame


def parse_cap_selex_batch_pairs(batch_path: str | Path):
    import pandas as pd

    batch_path = resolve_path(batch_path)
    frame = pd.read_csv(batch_path, sep="\t")
    if frame.empty:
        return frame
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={"HGNC": "hgnc_pair", "Barcode": "pair_seq_prefix", "Batch": "pair_plate"})
    frame["hgnc_pair"] = frame["hgnc_pair"].astype(str)
    frame["is_pair"] = frame["hgnc_pair"].str.contains("_")
    frame = frame[frame["is_pair"]].copy()
    frame["left_tf"] = frame["hgnc_pair"].str.split("_").str[0]
    frame["right_tf"] = frame["hgnc_pair"].str.split("_").str[1]
    frame["pair_seq_file"] = (
        frame["pair_seq_prefix"].fillna("").astype(str)
        + frame["pair_plate"].fillna("").astype(str)
        + "3u_sig.seq"
    )
    return frame[["hgnc_pair", "left_tf", "right_tf", "pair_seq_prefix", "pair_plate", "pair_seq_file"]]


def build_cap_selex_pair_inventory(curated_frame, selex_root: str | Path):
    import pandas as pd

    selex_root = resolve_path(selex_root)
    available_files = {path.name: path for path in selex_root.glob("*_sig.seq")}
    records = []
    for row in curated_frame.itertuples(index=False):
        records.append(
            {
                "pair_name": row.hgnc_pair,
                "left_tf": row.left_tf,
                "right_tf": row.right_tf,
                "is_pair": bool(row.is_pair),
                "left_seq_file": row.left_seq_file,
                "right_seq_file": row.right_seq_file,
                "left_seq_exists": row.left_seq_file in available_files,
                "right_seq_exists": row.right_seq_file in available_files if row.right_seq_file else False,
                "left_seq_bytes": int(available_files[row.left_seq_file].stat().st_size) if row.left_seq_file in available_files else 0,
                "right_seq_bytes": int(available_files[row.right_seq_file].stat().st_size) if row.right_seq_file in available_files else 0,
            }
        )
    return pd.DataFrame.from_records(records)


def build_cap_selex_seq_inventory(selex_root: str | Path):
    import pandas as pd

    selex_root = resolve_path(selex_root)
    records = []
    for path in sorted(selex_root.glob("*_sig.seq")):
        line_count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _line_count, _ in enumerate(handle, start=1):
                line_count = _line_count
        records.append(
            {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "line_count": line_count,
                "stem": path.stem,
            }
        )
    return pd.DataFrame.from_records(records)


def build_get_resource_inventory(get_root: str | Path):
    import pandas as pd

    get_root = resolve_path(get_root)
    records = []
    for path in sorted(get_root.rglob("*")):
        if path.is_dir():
            continue
        row = {
            "relative_path": str(path.relative_to(get_root)),
            "suffix": path.suffix,
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".feather":
            try:
                frame = pd.read_feather(path)
                row["columns"] = ",".join(map(str, frame.columns))
                row["rows"] = len(frame)
            except Exception:
                row["columns"] = ""
                row["rows"] = -1
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_portal_inventory(root: str | Path):
    import pandas as pd

    root = resolve_path(root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "suffix": path.suffix,
                "bytes": path.stat().st_size,
                "is_landing_page": path.name.lower() == "landing.html",
            }
        )
    return pd.DataFrame.from_records(records)


def _normalize_ena_download_url(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if value.startswith("https://") or value.startswith("http://"):
        return value
    if value.startswith("ftp://"):
        return "https://" + value[len("ftp://") :]
    return "https://" + value


def _parse_cap_selex_ena_name(name: str) -> dict[str, object]:
    stem = str(name).strip()
    if stem.endswith(".fastq.gz"):
        stem = stem[: -len(".fastq.gz")]
    tokens = stem.split("_")
    if len(tokens) != 5 or not tokens[2].isdigit():
        raise ValueError(f"Unexpected CAP-SELEX ENA run name: {name}")
    round_index = int(tokens[2])
    return {
        "pair_name": f"{tokens[0]}_{tokens[1]}",
        "left_tf": tokens[0],
        "right_tf": tokens[1],
        "round_index": round_index,
        "round_label": f"{round_index}u",
        "plate": tokens[3],
        "seq_prefix": tokens[4],
        "sig_file_name": f"{tokens[4]}{tokens[3]}{round_index}u_sig.seq",
    }


def build_cap_selex_ena_inventory(run_report_path: str | Path):
    import pandas as pd

    run_report_path = resolve_path(run_report_path)
    frame = pd.read_csv(run_report_path, sep="\t")
    records = []
    for row in frame.to_dict(orient="records"):
        submitted_ftp = str(row.get("submitted_ftp", "") or "")
        fastq_ftp = str(row.get("fastq_ftp", "") or "")
        file_name = Path(submitted_ftp or fastq_ftp).name
        if not file_name:
            experiment_title = str(row.get("experiment_title", "") or "")
            title_match = re.search(r"Raw reads:\s*([A-Za-z0-9._-]+)$", experiment_title)
            file_name = f"{title_match.group(1)}.fastq.gz" if title_match else ""
        if not file_name:
            continue
        parsed = _parse_cap_selex_ena_name(file_name)
        key = tuple(sorted((str(parsed["left_tf"]), str(parsed["right_tf"]))))
        records.append(
            {
                **parsed,
                "pair_key": "::".join(key),
                "canonical_left_tf": key[0],
                "canonical_right_tf": key[1],
                "run_accession": str(row.get("run_accession", "")),
                "study_accession": str(row.get("study_accession", "")),
                "experiment_accession": str(row.get("experiment_accession", "")),
                "sample_accession": str(row.get("sample_accession", "")),
                "submitted_url": _normalize_ena_download_url(submitted_ftp),
                "fastq_url": _normalize_ena_download_url(fastq_ftp),
                "submitted_bytes": int(0 if pd.isna(row.get("submitted_bytes")) else row.get("submitted_bytes", 0)),
                "fastq_bytes": int(0 if pd.isna(row.get("fastq_bytes")) else row.get("fastq_bytes", 0)),
                "instrument_platform": str(row.get("instrument_platform", "")),
                "instrument_model": str(row.get("instrument_model", "")),
                "experiment_title": str(row.get("experiment_title", "")),
            }
        )
    inventory = pd.DataFrame.from_records(records)
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "pair_name",
                "left_tf",
                "right_tf",
                "round_index",
                "round_label",
                "plate",
                "seq_prefix",
                "sig_file_name",
                "pair_key",
                "canonical_left_tf",
                "canonical_right_tf",
                "run_accession",
                "submitted_url",
                "fastq_url",
            ]
        )
    return inventory.sort_values(["canonical_left_tf", "canonical_right_tf", "run_accession"]).reset_index(drop=True)


def build_cap_selex_ena_pair_targets(ena_inventory, supplementary_edges):
    import pandas as pd

    if isinstance(ena_inventory, (str, Path)):
        ena_inventory_path = ena_inventory
        ena_inventory = read_table(ena_inventory_path)
        if "pair_key" not in ena_inventory.columns or "sig_file_name" not in ena_inventory.columns:
            ena_inventory = build_cap_selex_ena_inventory(ena_inventory_path)
    if isinstance(supplementary_edges, (str, Path)):
        supplementary_edges = read_table(supplementary_edges)
    ena_inventory = ena_inventory.copy()
    supplementary_edges = supplementary_edges.copy()
    if ena_inventory.empty or supplementary_edges.empty:
        return pd.DataFrame(
            columns=[
                "pair_key",
                "left_tf",
                "right_tf",
                "interaction_code",
                "interaction_scope",
                "interaction_type",
                "ena_run_count",
                "run_accession",
                "sig_file_name",
                "submitted_url",
            ]
        )
    supplementary_edges["pair_key"] = supplementary_edges.apply(
        lambda row: "::".join(sorted((str(row["left_tf"]), str(row["right_tf"])))), axis=1
    )
    ena_counts = ena_inventory.groupby("pair_key").size().rename("ena_run_count").reset_index()
    best_runs = (
        ena_inventory.sort_values(["pair_key", "submitted_bytes", "fastq_bytes", "run_accession"], ascending=[True, False, False, True])
        .groupby("pair_key", as_index=False)
        .first()
    )
    targets = supplementary_edges.merge(ena_counts, on="pair_key", how="left").merge(
        best_runs[
            [
                "pair_key",
                "pair_name",
                "run_accession",
                "sig_file_name",
                "submitted_url",
                "fastq_url",
                "submitted_bytes",
                "fastq_bytes",
                "round_label",
                "plate",
                "seq_prefix",
            ]
        ],
        on="pair_key",
        how="left",
    )
    targets["ena_run_count"] = targets["ena_run_count"].fillna(0).astype(int)
    return targets.sort_values(
        ["interaction_code", "ena_run_count", "left_tf", "right_tf"], ascending=[False, False, True, True]
    ).reset_index(drop=True)


def _extract_hca_next_data(landing_path: str | Path) -> dict:
    landing_path = resolve_path(landing_path)
    text = landing_path.read_text(encoding="utf-8")
    match = re.search(r'__NEXT_DATA__" type="application/json">(.*?)</script>', text)
    if match is None:
        raise ValueError(f"Unable to locate __NEXT_DATA__ payload in {landing_path}")
    return json.loads(match.group(1))


def build_hca_fragment_manifest(landing_path: str | Path):
    import pandas as pd

    payload = _extract_hca_next_data(landing_path)
    project = payload["props"]["pageProps"]["data"]["projects"][0]
    records = []

    def visit(node, breadcrumbs: list[tuple[str, str]]) -> None:
        if isinstance(node, dict):
            if {"name", "format", "uuid", "azul_url"}.issubset(node.keys()):
                context = {key: value for key, value in breadcrumbs}
                records.append(
                    {
                        **context,
                        "name": str(node.get("name", "")),
                        "format": str(node.get("format", "")),
                        "size": int(node.get("size", 0) or 0),
                        "uuid": str(node.get("uuid", "")),
                        "version": str(node.get("version", "")),
                        "file_source": str(node.get("fileSource", "")),
                        "content_description": ";".join(map(str, node.get("contentDescription", []) or [])),
                        "azul_url": str(node.get("azul_url", "")),
                        "drs_uri": str(node.get("drs_uri", "")),
                        "azul_mirror_uri": str(node.get("azul_mirror_uri", "")),
                    }
                )
                return
            for key, value in node.items():
                if key == "genusSpecies" and isinstance(value, dict):
                    for label, child in value.items():
                        visit(child, breadcrumbs + [("genus_species", str(label))])
                    continue
                if key == "developmentStage" and isinstance(value, dict):
                    for label, child in value.items():
                        visit(child, breadcrumbs + [("development_stage", str(label))])
                    continue
                if key == "libraryConstructionApproach" and isinstance(value, dict):
                    for label, child in value.items():
                        visit(child, breadcrumbs + [("library_construction_approach", str(label))])
                    continue
                if key == "organ" and isinstance(value, dict):
                    for label, child in value.items():
                        visit(child, breadcrumbs + [("organ", str(label))])
                    continue
                visit(value, breadcrumbs)
            return
        if isinstance(node, list):
            for value in node:
                visit(value, breadcrumbs)

    visit(project.get("contributedAnalyses", {}), [])
    frame = pd.DataFrame.from_records(records).drop_duplicates(subset=["uuid"])
    if not frame.empty:
        preferred = [
            "genus_species",
            "development_stage",
            "library_construction_approach",
            "organ",
            "name",
            "format",
            "size",
            "uuid",
            "version",
            "file_source",
            "content_description",
            "azul_url",
            "drs_uri",
            "azul_mirror_uri",
        ]
        ordered = [column for column in preferred if column in frame.columns] + [column for column in frame.columns if column not in preferred]
        frame = frame[ordered]
    return frame


def build_cap_selex_supplementary_edge_inventory(supplementary_root: str | Path):
    import pandas as pd

    supplementary_root = resolve_path(supplementary_root)
    candidates = [
        supplementary_root / "supplementary" / "41586_2025_8844_MOESM4_ESM.xlsx",
        supplementary_root / "unzipped" / "41586_2025_8844_MOESM4_ESM.xlsx",
        supplementary_root / "41586_2025_8844_MOESM4_ESM.xlsx",
    ]
    matrix_path = next((path for path in candidates if path.exists()), None)
    if matrix_path is None:
        return pd.DataFrame(columns=["left_tf", "right_tf", "interaction_code", "interaction_scope", "interaction_type"])

    sheet = pd.read_excel(matrix_path, header=None)
    header_rows = sheet.index[sheet.iloc[:, 0].astype(str).str.strip() == "Prey\\Bait"].tolist()
    if not header_rows:
        raise ValueError(f"Unable to locate interaction matrix header in {matrix_path}")
    header_row = int(header_rows[0])
    columns = [str(value).strip() for value in sheet.iloc[header_row, 1:].tolist() if pd.notna(value)]
    pair_records: dict[tuple[str, str], dict[str, object]] = {}
    for row_index in range(header_row + 1, len(sheet)):
        left_tf = sheet.iat[row_index, 0]
        if pd.isna(left_tf):
            continue
        left_tf = str(left_tf).strip()
        if not left_tf:
            continue
        if left_tf.lower().startswith("section "):
            break
        for col_index, right_tf in enumerate(columns, start=1):
            value = sheet.iat[row_index, col_index]
            if pd.isna(value):
                continue
            interaction_type = str(value).strip()
            if interaction_type in {"", "0", "0.0", "self"}:
                continue
            if interaction_type.lower().startswith("section "):
                break
            if "-" in interaction_type:
                code_token, scope = interaction_type.split("-", 1)
            else:
                code_token, scope = interaction_type, ""
            code_match = re.match(r'^"?(\d+)', str(code_token).strip())
            if code_match is None:
                continue
            interaction_code = int(code_match.group(1))
            key = tuple(sorted((left_tf, right_tf)))
            current = pair_records.get(key)
            payload = {
                "left_tf": key[0],
                "right_tf": key[1],
                "interaction_code": interaction_code,
                "interaction_scope": scope.strip(),
                "interaction_type": interaction_type,
            }
            if current is None or int(payload["interaction_code"]) > int(current["interaction_code"]):
                pair_records[key] = payload
    frame = pd.DataFrame.from_records(list(pair_records.values()))
    if frame.empty:
        return pd.DataFrame(columns=["left_tf", "right_tf", "interaction_code", "interaction_scope", "interaction_type"])
    return frame.sort_values(["left_tf", "right_tf"]).reset_index(drop=True)


def materialize_download_inventories(project_root: str | Path) -> None:
    project_root = resolve_path(project_root)
    cap_prey = project_root / "data_raw" / "cap_selex" / "relative_affinity" / "data" / "Curated_Prey_Final.txt"
    cap_selex_root = project_root / "data_raw" / "cap_selex" / "relative_affinity" / "SELEX"
    get_root = project_root / "data_raw" / "get_model" / "data"
    output_root = project_root / "data_intermediate"
    output_root.mkdir(parents=True, exist_ok=True)

    if cap_prey.exists():
        curated = parse_cap_selex_curated_prey(cap_prey)
        write_table(curated, output_root / "cap_selex_curated_pairs.tsv")
        write_table(build_cap_selex_pair_inventory(curated, cap_selex_root), output_root / "cap_selex_pair_inventory.tsv")
    if cap_selex_root.exists():
        write_table(build_cap_selex_seq_inventory(cap_selex_root), output_root / "cap_selex_seq_inventory.tsv")
    if get_root.exists():
        write_table(build_get_resource_inventory(get_root), output_root / "get_resource_inventory.tsv")
    hca_landing = project_root / "data_raw" / "hca" / "landing.html"
    if hca_landing.exists():
        write_table(build_hca_fragment_manifest(hca_landing), output_root / "hca_fragment_manifest.tsv")
    cap_ena_root = project_root / "data_raw" / "cap_selex" / "ena"
    ena_reports = sorted(cap_ena_root.glob("PRJ*_runs.tsv"))
    if ena_reports:
        ena_inventory = build_cap_selex_ena_inventory(ena_reports[0])
        write_table(ena_inventory, output_root / "cap_selex_ena_inventory.tsv")
    cap_repository_root = project_root / "data_raw" / "cap_selex" / "repository"
    if cap_repository_root.exists():
        supplementary_edges = build_cap_selex_supplementary_edge_inventory(cap_repository_root)
        if not supplementary_edges.empty:
            write_table(supplementary_edges, output_root / "cap_selex_supplementary_edges.tsv")
            ena_inventory_path = output_root / "cap_selex_ena_inventory.tsv"
            if ena_inventory_path.exists():
                write_table(
                    build_cap_selex_ena_pair_targets(ena_inventory_path, supplementary_edges),
                    output_root / "cap_selex_ena_pair_targets.tsv",
                )
    for portal_name in ["catlas", "hca", "encode"]:
        portal_root = project_root / "data_raw" / portal_name
        if portal_root.exists():
            write_table(build_portal_inventory(portal_root), output_root / f"{portal_name}_resource_inventory.tsv")
