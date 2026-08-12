# Outputs

`analysis/core_sensitivity.py` writes the identity-definition, sample-level aggregation, and small-sample robustness tables, together with animal- and eye-level calculation tables.

`analysis/overlap_sensitivity.py` writes the shared-gene matrix, animal-level overlap sensitivity scores, effect summaries, and a numerical check record for the original Contractile score, Contractile minus ACTA2, and overlap-removed CML score.

`analysis/compartment_sensitivity.py` writes dataset-specific results for the primary stromal definition and two outcome-independent exclusion definitions, plus a machine-readable verdict.

All calculations are descriptive at the biological-unit level. No cell-level inferential test is performed.

