#!/usr/bin/env python3
"""ACTA2 removal and overlap-removed CML sensitivity analysis."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


INPUT_DIR = Path(os.environ.get("CORNEA_INPUT_DIR", "inputs")).resolve()
OUT = Path(os.environ.get("CORNEA_OUTPUT_DIR", "results")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
OBJECT = INPUT_DIR / "GSE302936_processed.h5ad"

MODULES = {
    "contractile": ["ACTA2", "TAGLN", "MYL9", "CNN1", "TPM1", "TPM2", "CALD1"],
    "matricellular": ["POSTN", "SPARC", "TGFBI", "FN1", "TNC", "THBS1", "CTGF", "CYR61"],
    "LOX": ["LOX", "LOXL1", "LOXL2", "PLOD2"],
    "collagen_ECM": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "LOX", "LOXL1", "LOXL2"],
    "ECM_degradation": ["MMP2", "MMP9", "MMP14", "ADAMTS2", "ADAMTS4", "TIMP1", "TIMP2", "PLAU", "PLAUR"],
    "corneal_identity": ["KERA", "LUM", "DCN", "ALDH3A1", "KRT12", "AQP1", "PTGDS"],
    "inflammatory_fibrosis": ["IL1B", "TNF", "CCL2", "CXCL8", "PTGS2", "S100A8", "S100A9", "IL6", "OSM", "SOCS3"],
    "pericyte_proxy": ["PDGFRB", "RGS5", "ACTA2", "MCAM", "NOTCH3"],
    "vascular_proxy": ["PECAM1", "VWF", "KDR", "FLT1", "EMCN"],
    "stress_AP1": ["JUN", "JUNB", "FOS", "FOSB", "ATF3", "DUSP1", "DDIT3", "HSPA1A"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "UBE2C", "HMGB2"],
    "immune_proxy": ["PTPRC", "LST1", "TYROBP", "CD68", "CD14", "S100A8", "S100A9"],
}
ARTIFACT_MODULES = ["pericyte_proxy", "vascular_proxy", "stress_AP1", "proliferation", "immune_proxy"]
CORE_MODULES = ["contractile", "matricellular", "LOX"]


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0)
    sd[sd == 0] = np.nan
    return np.nan_to_num((matrix - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def build_shared_gene_matrix() -> tuple[pd.DataFrame, set[str]]:
    rows = []
    all_names = list(MODULES)
    for left, right in combinations(all_names, 2):
        overlap = sorted(set(MODULES[left]) & set(MODULES[right]))
        rows.append(
            {
                "module_a": left,
                "module_b": right,
                "shared_gene_count": len(overlap),
                "shared_genes": ";".join(overlap) if overlap else "none",
                "core_artifact_overlap": "yes" if ((left in CORE_MODULES and right in ARTIFACT_MODULES) or (right in CORE_MODULES and left in ARTIFACT_MODULES)) and overlap else "no",
            }
        )
    artifact_union = set().union(*(set(MODULES[name]) for name in ARTIFACT_MODULES))
    core_union = set().union(*(set(MODULES[name]) for name in CORE_MODULES))
    core_artifact_overlap = core_union & artifact_union
    frame = pd.DataFrame(rows)
    write_tsv(frame, OUT / "shared_gene_matrix.tsv")
    return frame, core_artifact_overlap


def score_sensitivity(core_artifact_overlap: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    backed = ad.read_h5ad(OBJECT, backed="r")
    obs = backed.obs.copy()
    gene_index = {str(g).upper(): i for i, g in enumerate(backed.var_names)}
    if "gene_symbol" in backed.var.columns:
        for i, gene in enumerate(backed.var["gene_symbol"].astype(str)):
            gene_index[str(gene).upper()] = i

    definitions = {
        "A_original_contractile": MODULES["contractile"],
        "B_contractile_minus_ACTA2": [g for g in MODULES["contractile"] if g != "ACTA2"],
        "C_contractile_minus_all_artifact_overlap": [g for g in MODULES["contractile"] if g not in core_artifact_overlap],
        "matricellular": MODULES["matricellular"],
        "LOX": MODULES["LOX"],
    }
    requested = sorted(set().union(*(set(genes) for genes in definitions.values())))
    present = [gene for gene in requested if gene in gene_index]
    memory = backed[:, [gene_index[gene] for gene in present]].to_memory()
    backed.file.close()
    matrix = memory.X.toarray() if sparse.issparse(memory.X) else np.asarray(memory.X)
    matrix = np.asarray(matrix, dtype=float)
    totals = pd.to_numeric(obs["nCount_RNA"], errors="coerce").to_numpy(dtype=float)
    replacement = np.nanmedian(totals[totals > 0])
    totals = np.where(np.isfinite(totals) & (totals > 0), totals, replacement)
    standardized = zscore_columns(np.log1p(matrix / totals[:, None] * 1e4))
    position = {gene: i for i, gene in enumerate(present)}

    cell_scores = pd.DataFrame(index=obs.index)
    for name, genes in definitions.items():
        matched = [gene for gene in genes if gene in position]
        if not matched:
            raise RuntimeError(f"No genes available for {name}")
        cell_scores[name] = standardized[:, [position[gene] for gene in matched]].mean(axis=1)
    cell_scores["D_CML_minus_all_artifact_overlap"] = cell_scores[
        ["C_contractile_minus_all_artifact_overlap", "matricellular", "LOX"]
    ].mean(axis=1)
    cell_scores["sample"] = obs["sample"].astype(str).to_numpy()
    cell_scores["condition"] = obs["condition"].astype(str).to_numpy()
    cell_scores["rough_cell_type"] = obs["rough_cell_type"].astype(str).to_numpy()
    stromal = cell_scores[cell_scores["rough_cell_type"].str.contains("stromal|keratocyte|myofibroblast|fibroblast", case=False, regex=True)].copy()
    measures = ["A_original_contractile", "B_contractile_minus_ACTA2", "C_contractile_minus_all_artifact_overlap", "D_CML_minus_all_artifact_overlap"]
    sample_scores = stromal.groupby(["sample", "condition"], as_index=False)[measures].mean()
    long = sample_scores.melt(id_vars=["sample", "condition"], var_name="sensitivity_definition", value_name="module_value")
    long["biological_unit"] = "independent animal"
    long["normalization"] = "log1p(count/cell_total*10000); gene-wise z-score within GSE302936; stromal/myofibroblast cell mean within animal"
    write_tsv(long, OUT / "overlap_sample_scores.tsv")

    effects = []
    original = None
    for measure in measures:
        endpoint = sample_scores.loc[sample_scores.condition == "alkali_burn_endpoint", measure].to_numpy(float)
        naive = sample_scores.loc[sample_scores.condition == "naive_control", measure].to_numpy(float)
        delta = endpoint.mean() - naive.mean()
        if measure == "A_original_contractile":
            original = delta
        attenuation = np.nan if measure == "A_original_contractile" else (original - delta) / abs(original) if original else np.nan
        all_endpoint_above_all_naive = bool(endpoint.min() > naive.max())
        direction_retained = bool(np.sign(delta) == np.sign(original)) if original is not None else True
        if not direction_retained or not all_endpoint_above_all_naive:
            decision = "UNSTABLE"
        elif measure != "A_original_contractile" and attenuation > 0.50:
            decision = "SUBSTANTIALLY_ATTENUATED"
        else:
            decision = "ROBUST_DIRECTION"
        effects.append(
            {
                "sensitivity_definition": measure,
                "genes_used": ";".join(
                    definitions.get(measure, definitions["C_contractile_minus_all_artifact_overlap"] + definitions["matricellular"] + definitions["LOX"])
                ),
                "n_endpoint_animals": len(endpoint),
                "n_naive_animals": len(naive),
                "endpoint_values": ";".join(f"{v:.9f}" for v in endpoint),
                "naive_values": ";".join(f"{v:.9f}" for v in naive),
                "endpoint_mean": endpoint.mean(),
                "naive_mean": naive.mean(),
                "endpoint_minus_naive_mean_difference": delta,
                "attenuation_fraction_vs_original_contractile": attenuation,
                "direction_retained": direction_retained,
                "all_endpoint_above_all_naive": all_endpoint_above_all_naive,
                "decision": decision,
                "inferential_test": "none",
            }
        )
    effect_frame = pd.DataFrame(effects)
    write_tsv(effect_frame, OUT / "overlap_effects.tsv")
    return long, effect_frame


def main() -> None:
    _, overlap = build_shared_gene_matrix()
    _, effects = score_sensitivity(overlap)
    expected = {
        "A_original_contractile": 0.7288742862107256,
        "B_contractile_minus_ACTA2": 0.750930,
        "D_CML_minus_all_artifact_overlap": 0.423646,
    }
    observed = effects.set_index("sensitivity_definition")[
        "endpoint_minus_naive_mean_difference"
    ].to_dict()
    for definition, target in expected.items():
        if abs(float(observed[definition]) - target) > 5e-6:
            raise AssertionError(
                f"Reported-value reproduction failed for {definition}: "
                f"{observed[definition]} vs {target}"
            )
    summary = {
        "shared_genes": sorted(overlap),
        "reported_value_reproduction": {
            key: float(observed[key]) for key in expected
        },
        "status": "PASS",
    }
    (OUT / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

