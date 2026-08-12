#!/usr/bin/env python3
"""Reproduce primary module effects and prespecified sensitivity analyses."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


INPUT_DIR = Path(os.environ.get("CORNEA_INPUT_DIR", "inputs")).resolve()
OUT = Path(os.environ.get("CORNEA_OUTPUT_DIR", "results")).resolve()
LOG = OUT / "run_records"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

GSE302936 = INPUT_DIR / "GSE302936_processed.h5ad"
GSE328779 = INPUT_DIR / "GSE328779_processed.h5ad"
REFERENCE_302 = INPUT_DIR / "GSE302936_animal_module_scores.tsv"
REFERENCE_328 = INPUT_DIR / "GSE328779_animal_residuals.tsv"

MODULES = {
    "contractile": ["ACTA2", "TAGLN", "MYL9", "CNN1", "TPM1", "TPM2", "CALD1"],
    "matricellular": ["POSTN", "SPARC", "TGFBI", "FN1", "TNC", "THBS1", "CTGF", "CYR61"],
    "collagen": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "LOX", "LOXL1", "LOXL2"],
    "ECM_remodeling": ["MMP2", "MMP9", "MMP14", "ADAMTS2", "ADAMTS4", "TIMP1", "TIMP2", "PLAU", "PLAUR"],
    "LOX_LOXL2": ["LOX", "LOXL1", "LOXL2", "PLOD2"],
    "keratocyte_associated_identity_current": ["KERA", "LUM", "DCN", "ALDH3A1", "KRT12", "AQP1", "PTGDS"],
    "keratocyte_associated_identity_KRT12_excluded": ["KERA", "LUM", "DCN", "ALDH3A1", "AQP1", "PTGDS"],
    "strict_keratocyte_core": ["KERA", "LUM", "DCN", "PTGDS"],
    "inflammation_associated_response": ["IL1B", "TNF", "CCL2", "CXCL8", "PTGS2", "S100A8", "S100A9", "IL6", "OSM", "SOCS3"],
}

EXPECTED = {
    "contractile": 1,
    "matricellular": 1,
    "collagen": 1,
    "ECM_remodeling": 1,
    "LOX_LOXL2": 1,
    "keratocyte_associated_identity_current": -1,
    "keratocyte_associated_identity_KRT12_excluded": -1,
    "strict_keratocyte_core": -1,
    "inflammation_associated_response": 1,
}

RABBIT_LIBRARY_MAP = {
    "GSM9689606": ("594E", "Day7", "FI"),
    "GSM9689608": ("594E", "Day7", "Control"),
    "GSM9689607": ("616E", "Day7", "FI"),
    "GSM9689609": ("616E", "Day7", "Control"),
    "GSM9689610": ("595E", "Day28", "FI"),
    "GSM9689612": ("595E", "Day28", "Control"),
    "GSM9689611": ("596E", "Day28", "FI"),
    "GSM9689613": ("596E", "Day28", "Control"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gene_index(a: ad.AnnData) -> dict[str, int]:
    idx = {str(g).upper(): i for i, g in enumerate(a.var_names)}
    for col in ("gene_symbol", "gene_ids"):
        if col in a.var.columns:
            for i, gene in enumerate(a.var[col].astype(str)):
                idx[str(gene).upper()] = i
    return idx


def normalized_gene_matrix(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, list[str]]]:
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    idx = gene_index(a)
    requested = sorted({gene.upper() for genes in MODULES.values() for gene in genes})
    matched = [gene for gene in requested if gene in idx]
    sub = a[:, [idx[gene] for gene in matched]].to_memory()
    a.file.close()
    x = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    x = np.asarray(x, dtype=float)
    count_col = "nCount_RNA" if "nCount_RNA" in obs else "n_counts" if "n_counts" in obs else None
    totals = pd.to_numeric(obs[count_col], errors="coerce").to_numpy(float) if count_col else x.sum(axis=1)
    replacement = np.nanmedian(totals[totals > 0]) if np.any(totals > 0) else 1.0
    totals = np.where(np.isfinite(totals) & (totals > 0), totals, replacement)
    normalized = np.log1p(x / totals[:, None] * 10_000.0)
    coverage = {
        module: [gene.upper() for gene in genes if gene.upper() in matched]
        for module, genes in MODULES.items()
    }
    return obs, normalized, matched, coverage


def zscale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=0)
    zero_variance = ~np.isfinite(sd) | (sd == 0)
    safe_sd = sd.copy()
    safe_sd[zero_variance] = np.nan
    z = (matrix - mean) / safe_sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, zero_variance


def score_modules(z: np.ndarray, matched: list[str], coverage: dict[str, list[str]]) -> pd.DataFrame:
    col = {gene: i for i, gene in enumerate(matched)}
    result = {}
    for module, present in coverage.items():
        use = [col[gene] for gene in present]
        result[module] = z[:, use].mean(axis=1) if use else np.full(z.shape[0], np.nan)
    return pd.DataFrame(result)


def stromal_mask(obs: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(obs.shape[0], dtype=bool)
    for col in ("rough_cell_type", "rough_marker_cell_type"):
        if col in obs:
            label = obs[col].astype(str).str.lower()
            mask |= label.str.contains("stromal|keratocyte|myofibroblast|fibroblast", regex=True, na=False).to_numpy()
    return mask


def current_cell_standardized_scores(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], int]:
    obs, normalized, matched, coverage = normalized_gene_matrix(path)
    z, zero_variance = zscale(normalized)
    scores = score_modules(z, matched, coverage)
    scores.index = obs.index
    for col in ("sample", "condition", "time_point", "condition_time", "rough_cell_type"):
        if col in obs:
            scores[col] = obs[col].values
    scores = scores.loc[stromal_mask(obs)].copy()
    return scores, coverage, int(zero_variance.sum())


def sample_balanced_scores(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], int]:
    obs, normalized, matched, coverage = normalized_gene_matrix(path)
    mask = stromal_mask(obs)
    samples = obs.loc[mask, "sample"].astype(str)
    gene_frame = pd.DataFrame(normalized[mask], index=samples.to_numpy(), columns=matched)
    sample_gene = gene_frame.groupby(level=0, sort=True).mean()
    z, zero_variance = zscale(sample_gene.to_numpy(float))
    scores = score_modules(z, matched, coverage)
    scores.insert(0, "sample", sample_gene.index)
    sample_meta = obs[["sample", "condition"] + [c for c in ("time_point", "condition_time") if c in obs]].copy()
    sample_meta["sample"] = sample_meta["sample"].astype(str)
    sample_meta = sample_meta.drop_duplicates("sample")
    scores = scores.merge(sample_meta, on="sample", how="left", validate="one_to_one")
    return scores, coverage, int(zero_variance.sum())


def write_identity_sensitivity(cell_scores_302: pd.DataFrame, coverage: dict[str, list[str]]) -> pd.DataFrame:
    modules = ["keratocyte_associated_identity_KRT12_excluded", "strict_keratocyte_core"]
    animal = cell_scores_302.groupby(["sample", "condition"], as_index=False, observed=True)[modules].mean()
    rows = []
    for module in modules:
        endpoint_mean = animal.loc[animal["condition"].eq("alkali_burn_endpoint"), module].mean()
        naive_mean = animal.loc[animal["condition"].eq("naive_control"), module].mean()
        delta = endpoint_mean - naive_mean
        for record in animal[["sample", "condition", module]].itertuples(index=False):
            rows.append({
                "definition": module,
                "requested_genes": ";".join(MODULES[module]),
                "available_genes": ";".join(coverage[module]),
                "missing_genes": ";".join(g for g in MODULES[module] if g not in coverage[module]),
                "animal": record.sample,
                "condition": record.condition,
                "animal_score": getattr(record, module),
                "endpoint_mean": endpoint_mean,
                "naive_control_mean": naive_mean,
                "endpoint_minus_naive": delta,
                "direction": "negative" if delta < 0 else "positive" if delta > 0 else "zero",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "S6_identity_definition_sensitivity.tsv", sep="\t", index=False)
    return out


def reported_original_deltas() -> tuple[dict[str, float], dict[str, float]]:
    p302 = pd.read_csv(REFERENCE_302, sep="\t")
    mapping = {
        "contractile": "mean_contractile_myofibroblast",
        "matricellular": "mean_matricellular_ECM",
        "collagen": "mean_collagen_ECM",
        "ECM_remodeling": "mean_ECM_degradation",
        "LOX_LOXL2": "mean_LOX_LOXL2_matrix_fixation",
        "keratocyte_associated_identity_current": "mean_corneal_identity",
        "inflammation_associated_response": "mean_inflammatory_fibrosis",
    }
    d302 = {}
    for module, col in mapping.items():
        means = p302.groupby("condition", observed=True)[col].mean()
        d302[module] = float(means["alkali_burn_endpoint"] - means["naive_control"])
    p328 = pd.read_csv(REFERENCE_328, sep="\t")
    module_map = {
        "contractile_myofibroblast": "contractile",
        "matricellular_ECM": "matricellular",
        "collagen_ECM": "collagen",
        "ECM_degradation": "ECM_remodeling",
        "LOX_LOXL2_matrix_fixation": "LOX_LOXL2",
        "corneal_identity": "keratocyte_associated_identity_current",
        "inflammatory_fibrosis": "inflammation_associated_response",
    }
    p328["module_std"] = p328["module"].map(module_map)
    means = p328.groupby(["module_std", "time_point"])["freeze_injury_minus_control"].mean().unstack()
    d328 = {module: float(row["Day7"] - row["Day28"]) for module, row in means.iterrows()}
    return d302, d328


def direction(value: float, expected: int) -> str:
    if value == 0 or not np.isfinite(value):
        return "zero_or_missing"
    return "expected" if np.sign(value) == expected else "reversed"


def gse328_residuals(sample_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in sample_scores["sample"]:
        if sample not in RABBIT_LIBRARY_MAP:
            raise AssertionError(f"Unmapped GSE328779 sample: {sample}")
    indexed = sample_scores.set_index("sample")
    for rabbit in sorted({v[0] for v in RABBIT_LIBRARY_MAP.values()}):
        records = [(sample, meta) for sample, meta in RABBIT_LIBRARY_MAP.items() if meta[0] == rabbit]
        day = records[0][1][1]
        fi = next(sample for sample, meta in records if meta[2] == "FI")
        control = next(sample for sample, meta in records if meta[2] == "Control")
        for module in MODULES:
            rows.append({
                "rabbit_id": rabbit,
                "time_point": day,
                "module": module,
                "FI_library": fi,
                "control_library": control,
                "FI_score": indexed.loc[fi, module],
                "control_score": indexed.loc[control, module],
                "injured_minus_control": indexed.loc[fi, module] - indexed.loc[control, module],
            })
    return pd.DataFrame(rows)


def write_sample_balanced(sample302: pd.DataFrame, sample328: pd.DataFrame, s6: pd.DataFrame) -> pd.DataFrame:
    d302_original, d328_original = reported_original_deltas()
    s6_delta = s6.drop_duplicates("definition").set_index("definition")["endpoint_minus_naive"].to_dict()
    d302_original.update(s6_delta)
    balanced302 = {}
    for module in MODULES:
        means = sample302.groupby("condition", observed=True)[module].mean()
        balanced302[module] = float(means["alkali_burn_endpoint"] - means["naive_control"])
    residuals = gse328_residuals(sample328)
    balanced328 = residuals.groupby(["module", "time_point"])["injured_minus_control"].mean().unstack()
    rows = []
    for dataset, original, balanced in [
        ("GSE302936", d302_original, balanced302),
        ("GSE328779", d328_original, {m: float(r["Day7"] - r["Day28"]) for m, r in balanced328.iterrows()}),
    ]:
        for module in MODULES:
            if module not in original:
                continue
            od = float(original[module])
            bd = float(balanced[module])
            exp = EXPECTED[module]
            rows.append({
                "dataset": dataset,
                "module": module,
                "original_direction": direction(od, exp),
                "sample_balanced_direction": direction(bd, exp),
                "original_delta": od,
                "sample_balanced_delta": bd,
                "direction_retained": direction(bd, exp) == direction(od, exp),
                "contrast_definition": "endpoint minus naive/control" if dataset == "GSE302936" else "mean day-7 paired-eye residual minus mean different-day-28 paired-eye residual",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "S7_sample_balanced_sensitivity.tsv", sep="\t", index=False)
    residuals.to_csv(OUT / "S7_GSE328779_sample_balanced_animal_residuals.tsv", sep="\t", index=False)
    sample302.to_csv(OUT / "S7_GSE302936_sample_balanced_animal_scores.tsv", sep="\t", index=False)
    sample328.to_csv(OUT / "S7_GSE328779_sample_balanced_eye_scores.tsv", sep="\t", index=False)
    return out


def write_small_n(cell_scores_302: pd.DataFrame) -> pd.DataFrame:
    modules = ["contractile", "matricellular", "LOX_LOXL2", "ECM_remodeling", "keratocyte_associated_identity_KRT12_excluded", "collagen"]
    animal = cell_scores_302.groupby(["sample", "condition"], as_index=False, observed=True)[modules].mean()
    endpoint = animal[animal["condition"].eq("alkali_burn_endpoint")].set_index("sample")
    control = animal[animal["condition"].eq("naive_control")].set_index("sample")
    summary = []
    pair_details = []
    leave_details = []
    for module in modules:
        exp = EXPECTED[module]
        pair_deltas = []
        for ep, ct in product(endpoint.index, control.index):
            delta = float(endpoint.loc[ep, module] - control.loc[ct, module])
            pair_deltas.append(delta)
            pair_details.append({"module": module, "endpoint_animal": ep, "control_animal": ct, "delta": delta, "expected_direction": bool(np.sign(delta) == exp)})
        leave_deltas = []
        for leave_ep, leave_ct in product(endpoint.index, control.index):
            ep_keep = endpoint.drop(index=leave_ep)[module]
            ct_keep = control.drop(index=leave_ct)[module]
            delta = float(ep_keep.mean() - ct_keep.mean())
            leave_deltas.append(delta)
            leave_details.append({"module": module, "left_out_endpoint": leave_ep, "left_out_control": leave_ct, "delta": delta, "expected_direction": bool(np.sign(delta) == exp)})
        summary.append({
            "dataset": "GSE302936",
            "module": module,
            "expected_direction": "negative" if exp < 0 else "positive",
            "pairwise_expected_direction_count": int(sum(np.sign(x) == exp for x in pair_deltas)),
            "pairwise_total": 9,
            "pairwise_fraction": float(np.mean([np.sign(x) == exp for x in pair_deltas])),
            "leave_one_expected_direction_count": int(sum(np.sign(x) == exp for x in leave_deltas)),
            "leave_one_total": 9,
            "leave_one_fraction": float(np.mean([np.sign(x) == exp for x in leave_deltas])),
            "minimum_leave_one_delta": float(min(leave_deltas)),
            "maximum_leave_one_delta": float(max(leave_deltas)),
            "minimum_absolute_leave_one_delta": float(min(abs(x) for x in leave_deltas)),
        })
    out = pd.DataFrame(summary)
    out.to_csv(OUT / "S8_small_n_directional_robustness.tsv", sep="\t", index=False)
    pd.DataFrame(pair_details).to_csv(OUT / "S8_pairwise_contrasts.tsv", sep="\t", index=False)
    pd.DataFrame(leave_details).to_csv(OUT / "S8_leave_one_endpoint_control_contrasts.tsv", sep="\t", index=False)
    return out


def validate_reported_reproduction(cell_scores_302: pd.DataFrame) -> dict[str, float]:
    current = cell_scores_302.groupby(["sample", "condition"], as_index=False, observed=True)[
        ["contractile", "matricellular", "collagen", "ECM_remodeling", "LOX_LOXL2", "keratocyte_associated_identity_current"]
    ].mean()
    observed = {}
    for module in ["contractile", "matricellular", "collagen", "ECM_remodeling", "LOX_LOXL2", "keratocyte_associated_identity_current"]:
        means = current.groupby("condition", observed=True)[module].mean()
        observed[module] = float(means["alkali_burn_endpoint"] - means["naive_control"])
    expected = {
        "contractile": 0.728874286,
        "matricellular": 0.350765736,
        "LOX_LOXL2": 0.169241225,
        "collagen": 0.073831309,
        "ECM_remodeling": 0.157791793,
        "keratocyte_associated_identity_current": -0.398440120,
    }
    for module, target in expected.items():
        if abs(observed[module] - target) > 5e-9:
            raise AssertionError(f"Reported-value reproduction failed for {module}: {observed[module]} vs {target}")
    return observed


def main() -> None:
    cell302, coverage302, zero_cell302 = current_cell_standardized_scores(GSE302936)
    observed = validate_reported_reproduction(cell302)
    s6 = write_identity_sensitivity(cell302, coverage302)
    balanced302, cov_b302, zero_b302 = sample_balanced_scores(GSE302936)
    balanced328, cov_b328, zero_b328 = sample_balanced_scores(GSE328779)
    s7 = write_sample_balanced(balanced302, balanced328, s6)
    s8 = write_small_n(cell302)

    science_review = s7[
        s7["module"].isin(["contractile", "matricellular", "keratocyte_associated_identity_KRT12_excluded"])
        & ~s7["direction_retained"]
    ]
    run_record = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "anndata": ad.__version__,
        "inputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in [GSE302936, GSE328779, REFERENCE_302, REFERENCE_328]},
        "normalization": "log1p(raw count / cell total * 10000)",
        "current_standardization_population": "all cells within each dataset, gene-wise population SD (ddof=0)",
        "sample_balanced_standardization_population": "stromal/myofibroblast sample-level normalized-gene profiles within each dataset, gene-wise population SD (ddof=0)",
        "zero_variance_gene_columns": {"current_GSE302936": zero_cell302, "sample_balanced_GSE302936": zero_b302, "sample_balanced_GSE328779": zero_b328},
        "coverage": {"GSE302936": cov_b302, "GSE328779": cov_b328},
        "reported_value_reproduction": observed,
        "scientific_review_required": not science_review.empty,
        "scientific_review_rows": science_review.to_dict(orient="records"),
        "outputs": ["S6_identity_definition_sensitivity.tsv", "S7_sample_balanced_sensitivity.tsv", "S8_small_n_directional_robustness.tsv"],
    }
    (LOG / "sensitivity_run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"PASS S6={len(s6)} rows S7={len(s7)} rows S8={len(s8)} rows")
    print(f"SCIENTIFIC_REVIEW_REQUIRED={not science_review.empty}")


if __name__ == "__main__":
    main()
