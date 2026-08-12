#!/usr/bin/env python3
"""Outcome-independent stromal-compartment sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


INPUT_DIR = Path(os.environ.get("CORNEA_INPUT_DIR", "inputs")).resolve()
OUT = Path(os.environ.get("CORNEA_OUTPUT_DIR", "results")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
GSE302 = INPUT_DIR / "GSE302936_processed.h5ad"
GSE328 = INPUT_DIR / "GSE328779_processed.h5ad"

MODULES = {
    "Contractile": ["ACTA2", "TAGLN", "MYL9", "CNN1", "TPM1", "TPM2", "CALD1"],
    "Matricellular": ["POSTN", "SPARC", "TGFBI", "FN1", "TNC", "THBS1", "CTGF", "CYR61"],
    "LOX/LOXL2": ["LOX", "LOXL1", "LOXL2", "PLOD2"],
    "ECM remodeling": ["MMP2", "MMP9", "MMP14", "ADAMTS2", "ADAMTS4", "TIMP1", "TIMP2", "PLAU", "PLAUR"],
    "Keratocyte-associated identity": ["KERA", "LUM", "DCN", "ALDH3A1", "KRT12", "AQP1", "PTGDS"],
    "strict KERA/LUM/DCN/PTGDS core": ["KERA", "LUM", "DCN", "PTGDS"],
    "Collagen": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "LOX", "LOXL1", "LOXL2"],
    "Inflammation-associated response": ["IL1B", "TNF", "CCL2", "CXCL8", "PTGS2", "S100A8", "S100A9", "IL6", "OSM", "SOCS3"],
}

EXCLUSIONS = {
    "epithelial_exclusion": ["KRT3", "KRT15", "EPCAM"],
    "basal_limbal_exclusion": ["KRT15", "KRT5", "TP63", "ITGA6"],
    "immune_exclusion": ["PTPRC", "LST1", "TYROBP", "CD68", "CD14"],
    "endothelial_vascular_exclusion": ["PECAM1", "VWF", "KDR", "FLT1", "EMCN", "RGS5"],
}

FORBIDDEN_GATE_GENES = {
    "KERA", "LUM", "DCN", "AQP1", "PTGDS", "ACTA2", "TAGLN", "MYL9", "CNN1",
    "POSTN", "COL1A1", "FN1", "LOX", "LOXL1", "LOXL2",
}

RABBIT_LIBRARY_MAP = {
    "GSM9689606": ("594E", "Day7", "FI"), "GSM9689608": ("594E", "Day7", "Control"),
    "GSM9689607": ("616E", "Day7", "FI"), "GSM9689609": ("616E", "Day7", "Control"),
    "GSM9689610": ("595E", "Day28", "FI"), "GSM9689612": ("595E", "Day28", "Control"),
    "GSM9689611": ("596E", "Day28", "FI"), "GSM9689613": ("596E", "Day28", "Control"),
}

ANCHOR_302 = {
    "Contractile": 0.7288742862107256, "Matricellular": 0.3507657356263531,
    "LOX/LOXL2": 0.1692412252028231, "Collagen": 0.0738313094326886,
    "ECM remodeling": 0.1577917927117815, "Keratocyte-associated identity": -0.3984401197997662,
}
ANCHOR_328 = {
    "Contractile": 0.4048380445322568, "Matricellular": 0.1783258318741007,
    "LOX/LOXL2": 0.1814645537343694, "Collagen": 0.1191988986241643,
    "ECM remodeling": 0.1525735768327857, "Keratocyte-associated identity": -0.054121461980339,
    "Inflammation-associated response": 0.1392497292976025,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gene_index(a: ad.AnnData) -> dict[str, int]:
    idx = {str(g).upper(): i for i, g in enumerate(a.var_names)}
    for col in ("gene_symbol", "gene_ids"):
        if col in a.var.columns:
            for i, gene in enumerate(a.var[col].astype(str)):
                idx[str(gene).upper()] = i
    return idx


def load_standardized(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    idx = gene_index(a)
    requested = sorted({g for genes in list(MODULES.values()) + list(EXCLUSIONS.values()) for g in genes})
    matched = [g for g in requested if g in idx]
    sub = a[:, [idx[g] for g in matched]].to_memory()
    a.file.close()
    x = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    x = np.asarray(x, dtype=float)
    total_col = "nCount_RNA" if "nCount_RNA" in obs else "n_counts"
    totals = pd.to_numeric(obs[total_col], errors="coerce").to_numpy(float)
    replacement = np.nanmedian(totals[totals > 0]) if np.any(totals > 0) else 1.0
    totals = np.where(np.isfinite(totals) & (totals > 0), totals, replacement)
    lognorm = np.log1p(x / totals[:, None] * 10000.0)
    means = np.nanmean(lognorm, axis=0)
    sds = np.nanstd(lognorm, axis=0, ddof=0)
    sds[~np.isfinite(sds) | (sds == 0)] = np.nan
    z = np.nan_to_num((lognorm - means) / sds, nan=0.0, posinf=0.0, neginf=0.0)
    zdf = pd.DataFrame(z, index=obs.index, columns=matched)
    module_coverage = {name: [g for g in genes if g in matched] for name, genes in MODULES.items()}
    exclusion_coverage = {name: [g for g in genes if g in matched] for name, genes in EXCLUSIONS.items()}
    scores = pd.DataFrame(index=obs.index)
    for name, genes in module_coverage.items():
        scores[name] = zdf[genes].mean(axis=1) if genes else np.nan
    for name, genes in exclusion_coverage.items():
        scores[name] = zdf[genes].mean(axis=1) if genes else np.nan
    scores["exclusion_max"] = scores[list(EXCLUSIONS)].max(axis=1, skipna=True)
    return obs, scores, module_coverage, exclusion_coverage


def primary_mask(obs: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(obs), dtype=bool)
    for col in ("rough_cell_type", "rough_marker_cell_type"):
        if col in obs:
            mask |= obs[col].astype(str).str.lower().str.contains("stromal|keratocyte|myofibroblast|fibroblast", regex=True, na=False).to_numpy()
    return mask


def gate_masks(obs: pd.DataFrame, scores: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "PRIMARY_ROUGH_CELL_TYPE": primary_mask(obs),
        "NEGSEL_STRICT": scores.exclusion_max.to_numpy(float) <= 0.0,
        "NEGSEL_PERMISSIVE": scores.exclusion_max.to_numpy(float) <= 0.5,
    }


def summarize_302(obs: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    modules = ["Contractile", "Matricellular", "LOX/LOXL2", "ECM remodeling", "Keratocyte-associated identity", "strict KERA/LUM/DCN/PTGDS core", "Collagen"]
    rows = []
    deltas: dict[str, float] = {}
    for gate, mask in gate_masks(obs, scores).items():
        selected = scores.loc[mask, modules].copy()
        selected["sample"] = obs.loc[mask, "sample"].astype(str).values
        selected["condition"] = obs.loc[mask, "condition"].astype(str).values
        animal = selected.groupby(["sample", "condition"], observed=True, as_index=False)[modules].mean()
        counts = selected.groupby(["sample", "condition"], observed=True).size().rename("selected_cell_count").reset_index()
        animal = animal.merge(counts, on=["sample", "condition"], validate="one_to_one")
        if animal["sample"].nunique() != 6:
            raise AssertionError(f"{gate} does not retain all six GSE302936 animals")
        for module in modules:
            endpoint = float(animal.loc[animal.condition.eq("alkali_burn_endpoint"), module].mean())
            naive = float(animal.loc[animal.condition.eq("naive_control"), module].mean())
            delta = endpoint - naive
            deltas[f"{gate}|{module}"] = delta
            for r in animal.itertuples(index=False):
                rows.append({
                    "dataset": "GSE302936", "record_type": "animal_score", "gate": gate,
                    "biological_unit": r.sample, "condition_or_day": r.condition, "module": module,
                    "selected_cell_count": int(r.selected_cell_count), "module_score": getattr(r, module.replace("/", "_").replace(" ", "_"), np.nan),
                    "endpoint_mean": endpoint, "naive_control_mean": naive, "endpoint_minus_naive_delta": delta,
                    "direction": "positive" if delta > 0 else "negative" if delta < 0 else "zero",
                })
    out = pd.DataFrame(rows)
    # itertuples sanitizes column names unpredictably; fill module scores by indexed lookup.
    for idx, row in out.iterrows():
        mask = (out.gate.eq(row.gate) & out.biological_unit.eq(row.biological_unit) & out.module.eq(row.module))
        if pd.isna(out.at[idx, "module_score"]):
            gate_mask = gate_masks(obs, scores)[row.gate]
            values = scores.loc[gate_mask & obs["sample"].astype(str).eq(row.biological_unit).to_numpy(), row.module]
            out.at[idx, "module_score"] = float(values.mean())
    return out, deltas


def summarize_328(obs: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    modules = ["Contractile", "Matricellular", "LOX/LOXL2", "ECM remodeling", "Keratocyte-associated identity", "Inflammation-associated response", "Collagen"]
    rows = []
    deltas: dict[str, float] = {}
    for gate, mask in gate_masks(obs, scores).items():
        selected = scores.loc[mask, modules].copy()
        selected["sample"] = obs.loc[mask, "sample"].astype(str).values
        eye = selected.groupby("sample", observed=True)[modules].mean()
        counts = selected.groupby("sample", observed=True).size()
        if eye.index.nunique() != 8:
            raise AssertionError(f"{gate} does not retain all eight GSE328779 eye libraries")
        for sample in eye.index:
            rabbit, day, condition = RABBIT_LIBRARY_MAP[sample]
            for module in modules:
                rows.append({
                    "dataset": "GSE328779", "record_type": "eye_score", "gate": gate,
                    "biological_unit": sample, "rabbit_id": rabbit, "condition_or_day": f"{day}_{condition}",
                    "module": module, "selected_cell_count": int(counts.loc[sample]), "module_score": float(eye.loc[sample, module]),
                })
        for module in modules:
            residuals = []
            for rabbit in sorted({v[0] for v in RABBIT_LIBRARY_MAP.values()}):
                members = [(s, meta) for s, meta in RABBIT_LIBRARY_MAP.items() if meta[0] == rabbit]
                day = members[0][1][1]
                fi = next(s for s, meta in members if meta[2] == "FI")
                control = next(s for s, meta in members if meta[2] == "Control")
                residual = float(eye.loc[fi, module] - eye.loc[control, module])
                residuals.append((rabbit, day, residual))
                rows.append({
                    "dataset": "GSE328779", "record_type": "rabbit_residual", "gate": gate,
                    "biological_unit": rabbit, "rabbit_id": rabbit, "condition_or_day": day,
                    "module": module, "injured_minus_control_residual": residual,
                    "selected_cell_count": int(counts.loc[fi] + counts.loc[control]),
                })
            day7 = float(np.mean([x[2] for x in residuals if x[1] == "Day7"]))
            day28 = float(np.mean([x[2] for x in residuals if x[1] == "Day28"]))
            delta = day7 - day28
            deltas[f"{gate}|{module}"] = delta
            rows.append({
                "dataset": "GSE328779", "record_type": "gate_summary", "gate": gate,
                "biological_unit": "day_group_mean", "module": module,
                "day7_mean_residual": day7, "day28_mean_residual": day28,
                "cross_sectional_day7_minus_day28": delta,
                "direction": "positive" if delta > 0 else "negative" if delta < 0 else "zero",
            })
    return pd.DataFrame(rows), deltas


def verdicts(d302: dict[str, float], d328: dict[str, float]) -> dict[str, object]:
    expected_302 = {"Contractile": 1, "Matricellular": 1, "strict KERA/LUM/DCN/PTGDS core": -1}
    detail302 = {}
    for module, sign in expected_302.items():
        vals = [d302[f"NEGSEL_STRICT|{module}"], d302[f"NEGSEL_PERMISSIVE|{module}"]]
        detail302[module] = [bool(np.sign(v) == sign) for v in vals]
    retained_counts = [sum(v) for v in detail302.values()]
    verdict302 = "PASS" if min(retained_counts) == 2 else "QUALIFIED_PASS" if min(retained_counts) == 1 else "FAIL"
    repair = [d328["NEGSEL_STRICT|Contractile"] > 0, d328["NEGSEL_PERMISSIVE|Contractile"] > 0]
    verdict328 = "PASS" if all(repair) else "QUALIFIED_PASS" if any(repair) else "FAIL"
    return {
        "CORE_GATE_SENSITIVITY": verdict302,
        "GSE302936_core_direction_retention": detail302,
        "GSE328779_GATE_SENSITIVITY": verdict328,
        "GSE328779_contractile_repair_direction_retained": repair,
        "REPAIR_CONTEXT_GATE_DEPENDENT": not any(repair),
        "SCIENTIFIC_REVIEW_REQUIRED": verdict302 == "FAIL" or verdict328 == "FAIL",
    }


def assert_primary(deltas: dict[str, float], anchors: dict[str, float]) -> None:
    for module, target in anchors.items():
        observed = deltas[f"PRIMARY_ROUGH_CELL_TYPE|{module}"]
        if abs(observed - target) > 5e-9:
            raise AssertionError(f"Primary regression failed for {module}: {observed} vs {target}")


def main() -> None:
    if FORBIDDEN_GATE_GENES & {g for genes in EXCLUSIONS.values() for g in genes}:
        raise AssertionError("Outcome-defining gene leaked into the exclusion gate")
    obs302, scores302, covm302, cove302 = load_standardized(GSE302)
    obs328, scores328, covm328, cove328 = load_standardized(GSE328)
    out302, d302 = summarize_302(obs302, scores302)
    out328, d328 = summarize_328(obs328, scores328)
    assert_primary(d302, ANCHOR_302)
    assert_primary(d328, ANCHOR_328)
    out302.to_csv(OUT / "GSE302936_GATE_SENSITIVITY.tsv", sep="\t", index=False)
    out328.to_csv(OUT / "GSE328779_GATE_SENSITIVITY.tsv", sep="\t", index=False)
    verdict = verdicts(d302, d328)
    verdict.update({
        "source_independent_annotation": {"GSE302936": "not_available", "GSE328779": "not_available"},
        "source_annotation_reason": "GSE302936 active_ident/seurat_clusters are cluster identifiers rather than an outcome-independent source broad stromal annotation; GSE328779 was built from raw 10x matrices and has no source annotation.",
        "normalization": "log1p(raw count / cell total * 10000)",
        "standardization_population": "all cells within each dataset, gene-wise population SD (ddof=0), identical to the primary cell-standardized scoring scale",
        "gate_definitions": {"NEGSEL_STRICT": "exclusion_max <= 0", "NEGSEL_PERMISSIVE": "exclusion_max <= 0.5"},
        "exclusion_gene_sets": EXCLUSIONS,
        "module_coverage": {"GSE302936": covm302, "GSE328779": covm328},
        "exclusion_coverage": {"GSE302936": cove302, "GSE328779": cove328},
        "input_sha256": {"GSE302936": sha256(GSE302), "GSE328779": sha256(GSE328)},
        "deltas": {"GSE302936": d302, "GSE328779": d328},
    })
    (OUT / "GATE_SENSITIVITY_VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ["CORE_GATE_SENSITIVITY", "GSE328779_GATE_SENSITIVITY", "SCIENTIFIC_REVIEW_REQUIRED"]}))


if __name__ == "__main__":
    main()
