from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_table

JASPAR_CORE_URL = "https://jaspar2026.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_non-redundant_pfms_jaspar.zip"
HOCOMOCO_PCM_URL = "https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pcms_HUMAN_mono.txt"
HOCOMOCO_ANNOTATION_URL = "https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_annotation_HUMAN_mono.tsv"
HUMAN_TF_DATABASE_URL = "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"
HUMAN_TF_MOTIF_LIST_URL = "https://humantfs.ccbr.utoronto.ca/download/v_1.01/Human_TF_MotifList_v_1.01.csv"


def download_file(url: str, output_path: str | Path, chunk_size: int = 1 << 16) -> Path:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return output_path


def _parse_matrix_lines(lines: Iterable[str]) -> list[dict[str, float]]:
    stripped = [line.strip() for line in lines if line.strip()]
    if stripped and stripped[0][0] not in {"A", "C", "G", "T"}:
        positions = []
        for position, line in enumerate(stripped):
            values = [float(token) for token in re.findall(r"[-+]?\d*\.?\d+", line)]
            if len(values) < 4:
                continue
            positions.append({"position": position, "A": values[0], "C": values[1], "G": values[2], "T": values[3]})
        return positions
    positions: list[dict[str, float]] = []
    current = {"A": [], "C": [], "G": [], "T": []}
    for line in stripped:
        base = line[0]
        if base not in current:
            continue
        values = [float(token) for token in re.findall(r"[-+]?\d*\.?\d+", line)]
        current[base] = values
    width = max((len(values) for values in current.values()), default=0)
    for position in range(width):
        positions.append(
            {
                "position": position,
                "A": current["A"][position] if position < len(current["A"]) else 0.0,
                "C": current["C"][position] if position < len(current["C"]) else 0.0,
                "G": current["G"][position] if position < len(current["G"]) else 0.0,
                "T": current["T"][position] if position < len(current["T"]) else 0.0,
            }
        )
    return positions


def parse_jaspar_bundle(zip_path: str | Path):
    import pandas as pd

    zip_path = resolve_path(zip_path)
    profiles = []
    positions = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".jaspar"):
                continue
            lines = archive.read(member).decode("utf-8").splitlines()
            if not lines:
                continue
            header = lines[0].lstrip(">").split("\t")
            motif_id = header[0].strip()
            gene_symbol = header[1].strip() if len(header) > 1 else motif_id
            matrix_rows = _parse_matrix_lines(lines[1:])
            profiles.append(
                {
                    "motif_id": motif_id,
                    "gene_symbol": gene_symbol.upper(),
                    "motif_source": "JASPAR2026",
                    "collection": "CORE",
                    "matrix_format": "jaspar",
                    "matrix_file": member,
                    "model_length": len(matrix_rows),
                }
            )
            for row in matrix_rows:
                positions.append({"motif_id": motif_id, "motif_source": "JASPAR2026", **row})
    return pd.DataFrame.from_records(profiles), pd.DataFrame.from_records(positions)


def parse_hocomoco_bundle(pcm_path: str | Path, annotation_path: str | Path):
    import pandas as pd

    pcm_path = resolve_path(pcm_path)
    annotation_path = resolve_path(annotation_path)
    annotation = pd.read_csv(annotation_path, sep="\t")
    annotation = annotation.rename(
        columns={
            "Model": "motif_id",
            "Transcription factor": "gene_symbol",
            "TF family": "tf_family",
            "TF subfamily": "tf_subfamily",
            "UniProt AC": "uniprot_id",
            "Quality": "quality",
            "Data source": "data_source",
        }
    )
    profiles = []
    positions = []
    current_id: str | None = None
    current_lines: list[str] = []
    with pcm_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    matrix_rows = _parse_matrix_lines(current_lines)
                    positions.extend({"motif_id": current_id, "motif_source": "HOCOMOCOv11", **row} for row in matrix_rows)
                    current_lines = []
                current_id = line[1:].strip()
            else:
                current_lines.append(line)
    if current_id is not None:
        matrix_rows = _parse_matrix_lines(current_lines)
        positions.extend({"motif_id": current_id, "motif_source": "HOCOMOCOv11", **row} for row in matrix_rows)
    position_frame = pd.DataFrame.from_records(positions)
    length_lookup = position_frame.groupby("motif_id").size().to_dict()
    annotation["gene_symbol"] = annotation["gene_symbol"].astype(str).str.upper()
    annotation["model_length"] = annotation["motif_id"].map(length_lookup).fillna(0).astype(int)
    annotation["motif_source"] = "HOCOMOCOv11"
    annotation["collection"] = "CORE"
    annotation["matrix_format"] = "pcm"
    profile_cols = [
        "motif_id",
        "gene_symbol",
        "motif_source",
        "collection",
        "matrix_format",
        "model_length",
        "quality",
        "data_source",
        "tf_family",
        "tf_subfamily",
        "uniprot_id",
    ]
    return annotation[profile_cols].copy(), position_frame


def materialize_motif_sources(project_root: str | Path) -> dict[str, str]:
    project_root = resolve_path(project_root)
    raw_root = ensure_dir(project_root / "data_raw")
    intermediate_root = ensure_dir(project_root / "data_intermediate")

    jaspar_zip = download_file(JASPAR_CORE_URL, raw_root / "jaspar2026" / "JASPAR2026_CORE_non-redundant_pfms_jaspar.zip")
    hocomoco_pcm = download_file(HOCOMOCO_PCM_URL, raw_root / "hocomoco" / "HOCOMOCOv11_core_pcms_HUMAN_mono.txt")
    hocomoco_annotation = download_file(HOCOMOCO_ANNOTATION_URL, raw_root / "hocomoco" / "HOCOMOCOv11_core_annotation_HUMAN_mono.tsv")
    human_tf_database = download_file(HUMAN_TF_DATABASE_URL, raw_root / "human_tf_catalog" / "DatabaseExtract_v_1.01.csv")
    human_tf_motif_list = download_file(HUMAN_TF_MOTIF_LIST_URL, raw_root / "human_tf_catalog" / "Human_TF_MotifList_v_1.01.csv")

    jaspar_profiles, jaspar_positions = parse_jaspar_bundle(jaspar_zip)
    hocomoco_profiles, hocomoco_positions = parse_hocomoco_bundle(hocomoco_pcm, hocomoco_annotation)

    write_table(jaspar_profiles, intermediate_root / "jaspar2026_profiles.tsv")
    write_table(jaspar_positions, intermediate_root / "jaspar2026_matrix_positions.tsv")
    write_table(hocomoco_profiles, intermediate_root / "hocomoco_profiles.tsv")
    write_table(hocomoco_positions, intermediate_root / "hocomoco_matrix_positions.tsv")
    write_table(pd.concat([jaspar_profiles, hocomoco_profiles], ignore_index=True, sort=False), intermediate_root / "motif_catalog.tsv")
    write_table(pd.concat([jaspar_positions, hocomoco_positions], ignore_index=True, sort=False), intermediate_root / "motif_matrix_positions.tsv")
    return {
        "jaspar_zip": str(jaspar_zip),
        "hocomoco_pcm": str(hocomoco_pcm),
        "hocomoco_annotation": str(hocomoco_annotation),
        "human_tf_database": str(human_tf_database),
        "human_tf_motif_list": str(human_tf_motif_list),
    }


def _load_optional_table(*paths: str | Path):
    for path in paths:
        candidate = resolve_path(path)
        if candidate.exists():
            return read_table(candidate)
    return pd.DataFrame()


def build_fallback_tf_catalog(project_root: str | Path):
    import pandas as pd

    project_root = resolve_path(project_root)
    intermediate_root = resolve_path(project_root / "data_intermediate")
    raw_root = resolve_path(project_root / "data_raw")

    human_tf = pd.read_csv(raw_root / "human_tf_catalog" / "DatabaseExtract_v_1.01.csv")
    human_tf["HGNC symbol"] = human_tf["HGNC symbol"].astype(str).str.upper()
    human_tf = human_tf[human_tf["Is TF?"].astype(str).str.upper() == "YES"].copy()
    human_tf = human_tf.rename(
        columns={
            "Ensembl ID": "ensembl_id",
            "HGNC symbol": "gene_symbol",
            "DBD": "dbd_family",
            "Binding mode": "binding_mode",
        }
    )

    motif_list = pd.read_csv(raw_root / "human_tf_catalog" / "Human_TF_MotifList_v_1.01.csv")
    motif_list["HGNC symbol"] = motif_list["HGNC symbol"].astype(str).str.upper()
    motif_list = motif_list.rename(columns={"HGNC symbol": "gene_symbol", "Motif ID": "human_tf_motif_id", "Motif source": "human_tf_motif_source"})

    hocomoco = read_table(intermediate_root / "hocomoco_profiles.tsv")
    jaspar = read_table(intermediate_root / "jaspar2026_profiles.tsv")
    tfclass = _load_optional_table(
        raw_root / "tfclass" / "tfclass_catalog.tsv",
        raw_root / "tfclass" / "tfclass_catalog.csv",
        intermediate_root / "tfclass_catalog.tsv",
    )
    cisbp = _load_optional_table(
        raw_root / "cisbp" / "cisbp_tf_catalog.tsv",
        raw_root / "cisbp" / "cisbp_tf_catalog.csv",
    )

    hocomoco = hocomoco.sort_values(["gene_symbol", "quality", "model_length"], ascending=[True, True, False]).drop_duplicates("gene_symbol")
    jaspar = jaspar.sort_values(["gene_symbol", "model_length"], ascending=[True, False]).drop_duplicates("gene_symbol")
    motif_list = motif_list.sort_values(["gene_symbol"]).drop_duplicates("gene_symbol")
    if not tfclass.empty:
        rename_map = {
            "gene_symbol": "gene_symbol",
            "symbol": "gene_symbol",
            "family": "tfclass_family",
            "subfamily": "tfclass_subfamily",
            "paralog_group": "tfclass_paralog_group",
        }
        tfclass = tfclass.rename(columns={key: value for key, value in rename_map.items() if key in tfclass.columns})
        if "gene_symbol" in tfclass.columns:
            tfclass["gene_symbol"] = tfclass["gene_symbol"].astype(str).str.upper()
            keep_cols = [column for column in ["gene_symbol", "tfclass_family", "tfclass_subfamily", "tfclass_paralog_group"] if column in tfclass.columns]
            tfclass = tfclass[keep_cols].drop_duplicates("gene_symbol")
    if not cisbp.empty:
        rename_map = {
            "TF_Name": "gene_symbol",
            "GeneSymbol": "gene_symbol",
            "MSource_Type": "cisbp_source_type",
            "DBID.1": "cisbp_motif_id",
            "TF_Family": "cisbp_family",
        }
        cisbp = cisbp.rename(columns={key: value for key, value in rename_map.items() if key in cisbp.columns})
        if "gene_symbol" in cisbp.columns:
            cisbp["gene_symbol"] = cisbp["gene_symbol"].astype(str).str.upper()
            keep_cols = [column for column in ["gene_symbol", "cisbp_source_type", "cisbp_motif_id", "cisbp_family"] if column in cisbp.columns]
            cisbp = cisbp[keep_cols].drop_duplicates("gene_symbol")

    cap_pairs = read_table(intermediate_root / "cap_selex_curated_pairs.tsv") if (intermediate_root / "cap_selex_curated_pairs.tsv").exists() else pd.DataFrame(columns=["left_tf", "right_tf"])
    cap_tfs = set(cap_pairs.get("left_tf", pd.Series(dtype=str)).astype(str).str.upper()) | set(cap_pairs.get("right_tf", pd.Series(dtype=str)).astype(str).str.upper())

    merged = human_tf.merge(motif_list[["gene_symbol", "human_tf_motif_id", "human_tf_motif_source"]], on="gene_symbol", how="left")
    merged = merged.merge(hocomoco.add_suffix("_hocomoco"), left_on="gene_symbol", right_on="gene_symbol_hocomoco", how="left")
    merged = merged.merge(jaspar.add_suffix("_jaspar"), left_on="gene_symbol", right_on="gene_symbol_jaspar", how="left")
    if not tfclass.empty:
        merged = merged.merge(tfclass, on="gene_symbol", how="left")
    if not cisbp.empty:
        merged = merged.merge(cisbp, on="gene_symbol", how="left")

    def _series(name: str):
        if name in merged.columns:
            return merged[name]
        return pd.Series([""] * len(merged), index=merged.index, dtype=object)

    merged["family"] = (
        _series("tfclass_family")
        .replace("", pd.NA)
        .fillna(_series("tf_family_hocomoco").replace("", pd.NA))
        .fillna(_series("cisbp_family").replace("", pd.NA))
        .fillna(_series("dbd_family").replace("", pd.NA))
        .fillna("")
    )
    merged["subfamily"] = (
        _series("tfclass_subfamily")
        .replace("", pd.NA)
        .fillna(_series("tf_subfamily_hocomoco").replace("", pd.NA))
        .fillna(_series("binding_mode").replace("", pd.NA))
        .fillna(merged["family"])
    )
    merged["paralog_group"] = (
        _series("tfclass_paralog_group")
        .replace("", pd.NA)
        .fillna(_series("tf_subfamily_hocomoco").replace("", pd.NA))
        .fillna(_series("dbd_family").replace("", pd.NA))
        .fillna(_series("gene_symbol"))
    )
    merged["motif_source"] = (
        _series("motif_source_hocomoco").replace("", pd.NA)
        .fillna(_series("motif_source_jaspar").replace("", pd.NA))
        .fillna(_series("cisbp_source_type").replace("", pd.NA))
        .fillna("")
    )
    merged["motif_id"] = (
        _series("motif_id_hocomoco").replace("", pd.NA)
        .fillna(_series("motif_id_jaspar").replace("", pd.NA))
        .fillna(_series("cisbp_motif_id").replace("", pd.NA))
        .fillna("")
    )
    merged["availability_proxy_sources"] = "RNA,accessibility"
    merged["is_in_CAP_SELEX"] = merged["gene_symbol"].isin(cap_tfs)
    merged["is_in_state_layer"] = False
    merged["is_in_orthogonal_validation"] = merged["gene_symbol"].isin({"TEAD4", "CLOCK", "GLI3", "RFX3", "PROX1", "HOX2"})
    merged["uniprot_id"] = merged["uniprot_id_hocomoco"].fillna("")
    merged = merged.sort_values("gene_symbol").drop_duplicates("gene_symbol").reset_index(drop=True)
    merged["tf_id"] = [f"TF_{index:05d}" for index in range(1, len(merged) + 1)]

    tf_catalog = merged[
        [
            "tf_id",
            "gene_symbol",
            "ensembl_id",
            "uniprot_id",
            "family",
            "subfamily",
            "paralog_group",
            "motif_source",
            "motif_id",
            "availability_proxy_sources",
            "is_in_CAP_SELEX",
            "is_in_state_layer",
            "is_in_orthogonal_validation",
        ]
    ].copy()
    write_table(tf_catalog, intermediate_root / "tfclass_catalog.tsv")
    write_table(tf_catalog, intermediate_root / "tf_master_table.tsv")
    reports_root = ensure_dir(project_root / "reports")
    (reports_root / "tf_catalog_fallback.md").write_text(
        "\n".join(
            [
                "# TF Catalog Fallback",
                "",
                f"- TFClass rows merged: {0 if tfclass.empty else len(tfclass)}",
                f"- CIS-BP rows merged: {0 if cisbp.empty else len(cisbp)}",
                "- Hierarchy preference order: TFClass > HOCOMOCO > CIS-BP > Human TF database DBD/binding-mode.",
                f"- Total TF rows in fallback catalog: {len(tf_catalog)}",
            ]
        ),
        encoding="utf-8",
    )
    (reports_root / "split_change_log.md").write_text(
        "\n".join(
            [
                "# Split Change Log",
                "",
                "- Initial formal TF master table materialized from public motif and TF hierarchy resources.",
                "- No split manifests should be altered without recording the reason in this file.",
            ]
        ),
        encoding="utf-8",
    )
    return tf_catalog
