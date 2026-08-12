# Inputs

The scripts expect the following local files. Public expression data are not redistributed in this repository.

| Local filename | Public source | Purpose |
|---|---|---|
| `GSE302936_processed.h5ad` | GEO accession GSE302936 | Cell-level expression and annotations for the independent-animal endpoint comparison |
| `GSE328779_processed.h5ad` | GEO accession GSE328779 | Cell-level expression and annotations for paired-eye residual calculations |
| `GSE302936_animal_module_scores.tsv` | Derived from GSE302936 | Reported animal-level module-score reference table |
| `GSE328779_animal_residuals.tsv` | Derived from GSE328779 | Reported within-rabbit residual reference table |

The processed objects retain count matrices and the dataset-specific sample, condition, time-point, and broad cell-type fields required by the scripts. The TSV files are included in the article's supporting source data.

