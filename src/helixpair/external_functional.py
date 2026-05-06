from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from helixpair.io_utils import ensure_dir, resolve_path, sha256_file, write_json, write_table
from helixpair.public_state import _centered_window, _fetch_ucsc_sequence


ENCODE_PORTAL = "https://www.encodeproject.org"
ENCODE_HEADERS = {"accept": "application/json"}
ENCODE_MPRA_SPECS = [
    {
        "dataset_id": "encode_tre_mpra_k562",
        "accession": "ENCSR382BVV",
        "collection": "functional-characterization-experiments",
        "biosample": "K562",
        "assay_family": "lentiMPRA",
        "citation_url": "https://www.nature.com/articles/s41586-024-08430-9",
    },
    {
        "dataset_id": "encode_tre_mpra_hepg2",
        "accession": "ENCSR022GQD",
        "collection": "functional-characterization-experiments",
        "biosample": "HepG2",
        "assay_family": "lentiMPRA",
        "citation_url": "https://www.nature.com/articles/s41586-024-08430-9",
    },
]
EPERTURBDB_SPECS = [
    {
        "dataset_id": "eperturbdb_enhancer_catalog",
        "url": "http://reggen.iiitd.edu.in:1207/ePerturbDB-html/downloads/enhancers_details_simple.bed",
        "local_name": "enhancers_details_simple.bed",
        "assay_family": "endogenous_perturbation",
        "citation_url": "https://academic.oup.com/database/article/doi/10.1093/database/baaf084/8426098",
    }
]
FLOWFISH_SPECS = [
    {
        "dataset_id": "hcr_flowfish_pubdata",
        "accession": "ENCSR455UGU",
        "collection": "publication-data",
        "assay_family": "HCR-FlowFISH",
        "citation_url": "https://www.nature.com/articles/s41588-021-00900-4",
    }
]
MPRA_BED_COLUMNS = [
    "chromosome",
    "region_start",
    "region_end",
    "element_id",
    "bed_score",
    "strand",
    "activity_score",
    "activity_aux_1",
    "activity_aux_2",
    "activity_aux_3",
    "activity_aux_4",
]
EPERTURBDB_COLUMNS = [
    "chromosome",
    "region_start",
    "region_end",
    "locus_descriptor",
    "perturbation_descriptor",
    "evidence_descriptor",
]


def _request_json(url: str) -> dict[str, Any]:
    response = requests.get(url, headers=ENCODE_HEADERS, timeout=120)
    response.raise_for_status()
    return response.json()


def _download(url: str, destination: Path, chunk_size: int = 1 << 20) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.raw.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
        temp_path.replace(destination)
        return destination
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _encode_record_url(accession: str, collection: str) -> str:
    return f"{ENCODE_PORTAL}/{collection}/{accession}/?format=json"


def fetch_encode_record(accession: str, collection: str) -> dict[str, Any]:
    return _request_json(_encode_record_url(accession, collection))


def _file_record_from_ref(ref: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ref, dict):
        return ref
    ref_path = str(ref).rstrip("/")
    return _request_json(f"{ENCODE_PORTAL}{ref_path}/?format=json")


def resolve_encode_files(
    record: dict[str, Any],
    *,
    allowed_formats: set[str],
    include_fastq: bool = False,
    file_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_file in record.get("files", []):
        file_record = _file_record_from_ref(raw_file)
        if str(file_record.get("status", "")) != "released":
            continue
        file_format = str(file_record.get("file_format", ""))
        if not include_fastq and file_format == "fastq":
            continue
        if allowed_formats and file_format not in allowed_formats:
            continue
        href = str(file_record.get("href", ""))
        if not href:
            continue
        if file_filter is not None and not file_filter(file_record):
            continue
        rows.append(
            {
                "accession": str(file_record.get("accession", "")),
                "file_format": file_format,
                "output_type": str(file_record.get("output_type", "")),
                "submitted_file_name": str(file_record.get("submitted_file_name", "")),
                "assembly": str(file_record.get("assembly", "")),
                "href": href,
                "download_url": f"{ENCODE_PORTAL}{href}",
            }
        )
    return rows


def _download_encode_spec(
    spec: dict[str, str],
    raw_root: Path,
    *,
    allowed_formats: set[str],
    resolve_files: bool,
    file_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    dataset_root = ensure_dir(raw_root / spec["dataset_id"])
    record = fetch_encode_record(spec["accession"], spec["collection"])
    metadata_path = dataset_root / "metadata.json"
    write_json(metadata_path, record)
    inventory_rows: list[dict[str, Any]] = [
        {
            "dataset_id": spec["dataset_id"],
            "kind": spec["collection"],
            "accession": spec["accession"],
            "local_path": str(metadata_path),
            "download_url": _encode_record_url(spec["accession"], spec["collection"]),
            "file_format": "json",
            "output_type": "metadata",
            "size_bytes": int(metadata_path.stat().st_size),
            "sha256": sha256_file(metadata_path),
        }
    ]
    if not resolve_files:
        return inventory_rows
    for file_meta in resolve_encode_files(record, allowed_formats=allowed_formats, file_filter=file_filter):
        extension = file_meta["submitted_file_name"].split("/")[-1] or f"{file_meta['accession']}.{file_meta['file_format']}"
        destination = dataset_root / extension
        _download(file_meta["download_url"], destination)
        inventory_rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "kind": spec["collection"],
                "accession": spec["accession"],
                "file_accession": file_meta["accession"],
                "local_path": str(destination),
                "download_url": file_meta["download_url"],
                "file_format": file_meta["file_format"],
                "output_type": file_meta["output_type"],
                "submitted_file_name": file_meta["submitted_file_name"],
                "size_bytes": int(destination.stat().st_size),
                "sha256": sha256_file(destination),
            }
        )
    return inventory_rows


def download_external_functional_sources(
    project_root: str | Path,
    *,
    download_flowfish_files: bool = False,
    include_bigbed: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    project_root = resolve_path(project_root)
    raw_root = ensure_dir(project_root / "data_raw" / "external_functional")
    report_root = ensure_dir(project_root / "reports" / "external_functional")
    inventory_rows: list[dict[str, Any]] = []

    def _aggregate_mpra_only(file_record: dict[str, Any]) -> bool:
        submitted_name = str(file_record.get("submitted_file_name", ""))
        file_format = str(file_record.get("file_format", ""))
        if file_format == "bed":
            return submitted_name.endswith(".bed.gz")
        if file_format == "bigBed":
            return submitted_name.endswith(".bb")
        if file_format == "tsv":
            return "_rep" not in submitted_name
        return False

    mpra_formats = {"bed", "tsv"}
    if include_bigbed:
        mpra_formats.add("bigBed")
    for spec in ENCODE_MPRA_SPECS:
        inventory_rows.extend(
            _download_encode_spec(
                spec,
                raw_root,
                allowed_formats=mpra_formats,
                resolve_files=True,
                file_filter=_aggregate_mpra_only,
            )
        )

    for spec in FLOWFISH_SPECS:
        inventory_rows.extend(
            _download_encode_spec(
                spec,
                raw_root,
                allowed_formats={"bed", "tsv", "bigBed"} if include_bigbed else {"bed", "tsv"},
                resolve_files=download_flowfish_files,
                file_filter=None,
            )
        )

    for spec in EPERTURBDB_SPECS:
        destination = raw_root / spec["dataset_id"] / spec["local_name"]
        _download(spec["url"], destination)
        inventory_rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "kind": "bed_track",
                "accession": "",
                "local_path": str(destination),
                "download_url": spec["url"],
                "file_format": "bed",
                "output_type": "catalog",
                "size_bytes": int(destination.stat().st_size),
                "sha256": sha256_file(destination),
            }
        )

    inventory = pd.DataFrame.from_records(inventory_rows).sort_values(
        ["dataset_id", "file_format", "output_type", "local_path"],
        kind="stable",
    )
    write_table(inventory, report_root / "source_inventory.csv")
    manifest = {
        "builder": "download_external_functional_sources",
        "raw_root": str(raw_root),
        "report_root": str(report_root),
        "datasets": sorted(inventory["dataset_id"].dropna().astype(str).unique().tolist()),
        "row_count": int(len(inventory)),
        "download_flowfish_files": bool(download_flowfish_files),
        "include_bigbed": bool(include_bigbed),
    }
    write_json(report_root / "source_inventory.json", manifest)
    return inventory, manifest


def parse_encode_mpra_bed(path: str | Path, *, dataset_id: str, biosample: str) -> pd.DataFrame:
    frame = pd.read_csv(
        resolve_path(path),
        sep="\t",
        header=None,
        names=MPRA_BED_COLUMNS,
        compression="gzip",
    )
    frame["source_dataset"] = dataset_id
    frame["assay_family"] = "lentiMPRA"
    frame["biosample_label"] = biosample
    frame["state_label"] = biosample
    frame["element_group"] = frame["element_id"].astype(str).str.replace("_Reversed:$", "", regex=True)
    frame["orientation_mode"] = frame["element_id"].astype(str).map(
        lambda value: "reverse_construct" if str(value).endswith("_Reversed:") else "forward_construct"
    )
    frame["region_start"] = frame["region_start"].astype(int)
    frame["region_end"] = frame["region_end"].astype(int)
    frame["activity_score"] = pd.to_numeric(frame["activity_score"], errors="coerce")
    frame["element_width"] = frame["region_end"] - frame["region_start"]
    frame["local_source_path"] = str(resolve_path(path))
    return frame


def _parse_eperturbdb_locus(value: str) -> tuple[str, str, str]:
    tokens = [token.strip() for token in str(value).split("_") if token.strip()]
    if not tokens:
        return "", "", ""
    enhancer_id = tokens[0]
    target_gene = tokens[-1] if len(tokens) >= 2 else ""
    biosample = " ".join(tokens[1:-1]) if len(tokens) >= 3 else ""
    return enhancer_id, biosample, target_gene


def _parse_eperturbdb_evidence(value: str) -> tuple[str, str]:
    match = re.match(r"^(?P<platform>[^_]+)_(?P<reference>.+)$", str(value).strip())
    if not match:
        return str(value).strip(), ""
    return match.group("platform"), match.group("reference")


def parse_eperturbdb_bed(path: str | Path) -> pd.DataFrame:
    source_path = resolve_path(path)
    lines = [
        line
        for line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line and not line.startswith("track ")
    ]
    frame = pd.DataFrame([line.split("\t") for line in lines], columns=EPERTURBDB_COLUMNS)
    enhancer_meta = frame["locus_descriptor"].astype(str).map(_parse_eperturbdb_locus)
    evidence_meta = frame["evidence_descriptor"].astype(str).map(_parse_eperturbdb_evidence)
    frame["enhancer_id"] = enhancer_meta.map(lambda item: item[0])
    frame["biosample_label"] = enhancer_meta.map(lambda item: item[1])
    frame["putative_target_gene"] = enhancer_meta.map(lambda item: item[2])
    frame["target_gene"] = frame["putative_target_gene"]
    frame["perturbation_platform"] = evidence_meta.map(lambda item: item[0])
    frame["reference_token"] = evidence_meta.map(lambda item: item[1])
    frame["source_dataset"] = "eperturbdb_enhancer_catalog"
    frame["assay_family"] = "endogenous_perturbation"
    frame["state_label"] = frame["biosample_label"].astype(str)
    frame["element_id"] = frame["enhancer_id"].astype(str)
    frame["element_width"] = frame["region_end"].astype(int) - frame["region_start"].astype(int)
    frame["local_source_path"] = str(resolve_path(path))
    return frame


def _discover_mpra_bed_paths(raw_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for spec in ENCODE_MPRA_SPECS:
        dataset_root = raw_root / spec["dataset_id"]
        for path in sorted(dataset_root.glob("ActivityRatios.*_full.bed.gz")):
            rows.append((spec["dataset_id"], spec["biosample"], path))
    return rows


def _discover_eperturbdb_path(raw_root: Path) -> Path | None:
    spec = EPERTURBDB_SPECS[0]
    path = raw_root / spec["dataset_id"] / spec["local_name"]
    return path if path.exists() else None


def _build_flowfish_dataset_manifest(raw_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in FLOWFISH_SPECS:
        dataset_root = raw_root / spec["dataset_id"]
        metadata_path = dataset_root / "metadata.json"
        if not metadata_path.exists():
            continue
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference_identifiers = []
        for reference in record.get("references", []):
            reference_identifiers.extend([str(identifier) for identifier in reference.get("identifiers", [])])
        records.append(
            {
                "dataset_id": spec["dataset_id"],
                "assay_family": spec["assay_family"],
                "accession": str(record.get("accession", "")),
                "file_reference_count": int(len(record.get("files", []))),
                "reference_identifiers": "; ".join(reference_identifiers),
                "status": str(record.get("status", "")),
                "metadata_path": str(metadata_path),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_flowfish_file_manifest(raw_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in FLOWFISH_SPECS:
        dataset_root = raw_root / spec["dataset_id"]
        metadata_path = dataset_root / "metadata.json"
        if not metadata_path.exists():
            continue
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        file_refs = record.get("files", [])
        for raw_file in file_refs:
            try:
                file_record = _file_record_from_ref(raw_file)
            except Exception:
                continue
            if str(file_record.get("status", "")) != "released":
                continue
            href = str(file_record.get("href", ""))
            if not href:
                continue
            records.append(
                {
                    "dataset_id": spec["dataset_id"],
                    "assay_family": spec["assay_family"],
                    "file_accession": str(file_record.get("accession", "")),
                    "file_format": str(file_record.get("file_format", "")),
                    "output_type": str(file_record.get("output_type", "")),
                    "submitted_file_name": str(file_record.get("submitted_file_name", "")),
                    "download_url": f"{ENCODE_PORTAL}{href}",
                }
            )
    return pd.DataFrame.from_records(records)


def hydrate_region_sequences(
    frame: pd.DataFrame,
    *,
    window_length: int,
    max_rows: int | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    hydrated = frame.head(max_rows).copy() if max_rows is not None else frame.copy()

    def _resolve_row(row: Any) -> tuple[int, int, str]:
        window_start, window_end = _centered_window(
            str(row.chromosome),
            int(row.region_start),
            int(row.region_end),
            int(window_length),
        )
        sequence = _fetch_ucsc_sequence(str(row.chromosome), int(window_start), int(window_end))
        return int(window_start), int(window_end), str(sequence)

    rows = list(hydrated.itertuples(index=False))
    workers = max(1, min(int(max_workers), len(rows)))
    if workers == 1:
        resolved = [_resolve_row(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resolved = list(pool.map(_resolve_row, rows))
    window_starts = [item[0] for item in resolved]
    window_ends = [item[1] for item in resolved]
    sequences = [item[2] for item in resolved]
    hydrated["window_start"] = window_starts
    hydrated["window_end"] = window_ends
    hydrated["sequence"] = sequences
    return hydrated


def build_external_functional_dataset(
    project_root: str | Path,
    *,
    window_length: int = 96,
    hydrate_sequences_for: str = "none",
    sequence_limit: int | None = None,
    resolve_flowfish_files: bool = False,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    raw_root = ensure_dir(project_root / "data_raw" / "external_functional")
    scenario_root = ensure_dir(project_root / "data_intermediate" / "external_functional")
    report_root = ensure_dir(project_root / "reports" / "external_functional")

    mpra_frames = [
        parse_encode_mpra_bed(path, dataset_id=dataset_id, biosample=biosample)
        for dataset_id, biosample, path in _discover_mpra_bed_paths(raw_root)
    ]
    mpra_frame = pd.concat(mpra_frames, ignore_index=True, sort=False) if mpra_frames else pd.DataFrame()
    if not mpra_frame.empty:
        write_table(mpra_frame, scenario_root / "mpra_observations.parquet")
        write_table(mpra_frame, report_root / "mpra_observations.csv")

    eperturb_path = _discover_eperturbdb_path(raw_root)
    endogenous_frame = parse_eperturbdb_bed(eperturb_path) if eperturb_path is not None else pd.DataFrame()
    if not endogenous_frame.empty:
        write_table(endogenous_frame, scenario_root / "endogenous_perturbation_observations.parquet")
        write_table(endogenous_frame, report_root / "endogenous_perturbation_observations.csv")

    flowfish_dataset_manifest = _build_flowfish_dataset_manifest(raw_root)
    if not flowfish_dataset_manifest.empty:
        write_table(flowfish_dataset_manifest, report_root / "flowfish_dataset_manifest.csv")
    flowfish_file_manifest = pd.DataFrame()
    flowfish_file_manifest_path = report_root / "flowfish_file_manifest.csv"
    if resolve_flowfish_files:
        flowfish_file_manifest = _build_flowfish_file_manifest(raw_root)
        if not flowfish_file_manifest.empty:
            write_table(flowfish_file_manifest, flowfish_file_manifest_path)
    elif flowfish_file_manifest_path.exists():
        flowfish_file_manifest_path.unlink()

    hydrated_outputs: dict[str, str] = {}
    hydrate_token = str(hydrate_sequences_for).lower().strip()
    if hydrate_token in {"mpra", "all"} and not mpra_frame.empty:
        hydrated_mpra = hydrate_region_sequences(mpra_frame, window_length=window_length, max_rows=sequence_limit)
        output_path = scenario_root / "mpra_sequence_windows.parquet"
        write_table(hydrated_mpra, output_path)
        hydrated_outputs["mpra_sequence_windows"] = str(output_path)
    if hydrate_token in {"endogenous", "all"} and not endogenous_frame.empty:
        hydrated_endogenous = hydrate_region_sequences(endogenous_frame, window_length=window_length, max_rows=sequence_limit)
        output_path = scenario_root / "endogenous_sequence_windows.parquet"
        write_table(hydrated_endogenous, output_path)
        hydrated_outputs["endogenous_sequence_windows"] = str(output_path)

    summary_rows: list[dict[str, Any]] = []
    if not mpra_frame.empty:
        for row in (
            mpra_frame.groupby(["source_dataset", "assay_family", "biosample_label"], as_index=False)
            .size()
            .rename(columns={"size": "rows"})
            .itertuples(index=False)
        ):
            summary_rows.append(
                {
                    "source_dataset": str(row.source_dataset),
                    "assay_family": str(row.assay_family),
                    "biosample_label": str(row.biosample_label),
                    "rows": int(row.rows),
                }
            )
    if not endogenous_frame.empty:
        for row in (
            endogenous_frame.groupby(["source_dataset", "assay_family"], as_index=False)
            .agg(
                rows=("element_id", "size"),
                biosamples=("biosample_label", "nunique"),
                putative_target_genes=("putative_target_gene", lambda values: int(pd.Series(values).astype(str).replace("", pd.NA).dropna().nunique())),
            )
            .itertuples(index=False)
        ):
            summary_rows.append(
                {
                    "source_dataset": str(row.source_dataset),
                    "assay_family": str(row.assay_family),
                    "biosample_label": "ALL",
                    "rows": int(row.rows),
                    "biosamples": int(row.biosamples),
                    "putative_target_genes": int(row.putative_target_genes),
                }
            )
    if summary_rows:
        write_table(pd.DataFrame.from_records(summary_rows), report_root / "dataset_rollup.csv")

    manifest = {
        "builder": "build_external_functional_dataset",
        "window_length": int(window_length),
        "hydrate_sequences_for": hydrate_token,
        "sequence_limit": None if sequence_limit is None else int(sequence_limit),
        "resolve_flowfish_files": bool(resolve_flowfish_files),
        "mpra_rows": int(len(mpra_frame)),
        "mpra_biosamples": sorted(mpra_frame["biosample_label"].dropna().astype(str).unique().tolist()) if not mpra_frame.empty else [],
        "endogenous_rows": int(len(endogenous_frame)),
        "endogenous_biosamples": sorted(endogenous_frame["biosample_label"].dropna().astype(str).unique().tolist()) if not endogenous_frame.empty else [],
        "endogenous_putative_target_genes": int(endogenous_frame["putative_target_gene"].astype(str).replace("", pd.NA).dropna().nunique()) if not endogenous_frame.empty else 0,
        "flowfish_dataset_manifest_rows": int(len(flowfish_dataset_manifest)),
        "flowfish_file_manifest_rows": int(len(flowfish_file_manifest)),
        "outputs": {
            "mpra_observations": str(scenario_root / "mpra_observations.parquet") if not mpra_frame.empty else "",
            "endogenous_perturbation_observations": str(scenario_root / "endogenous_perturbation_observations.parquet") if not endogenous_frame.empty else "",
            "dataset_rollup": str(report_root / "dataset_rollup.csv") if summary_rows else "",
            "flowfish_dataset_manifest": str(report_root / "flowfish_dataset_manifest.csv") if not flowfish_dataset_manifest.empty else "",
            "flowfish_file_manifest": str(flowfish_file_manifest_path) if not flowfish_file_manifest.empty else "",
            **hydrated_outputs,
        },
    }
    write_json(report_root / "external_functional_manifest.json", manifest)
    return manifest
