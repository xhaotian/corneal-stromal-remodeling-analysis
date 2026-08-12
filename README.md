# Corneal stromal remodeling analysis

This repository contains the analysis code used to calculate the reported module-level results from two public rabbit single-cell RNA-sequencing datasets: GSE302936 and GSE328779.

GSE302936 comprises three day-14 alkali-burn endpoint animals and three independent naive/control animals. Its biological unit is the independent animal, and the reported contrast is the endpoint group mean minus the naive/control group mean.

GSE328779 comprises four rabbits: two at day 7 and two different rabbits at day 28. Each rabbit contributes an injured eye and a contralateral control eye. The biological calculation is the within-rabbit injured-eye minus control-eye residual; day 7 and day 28 are a cross-sectional comparison between different rabbits, not a longitudinal trajectory.

The code performs module scoring, animal-level aggregation, identity-definition sensitivity, sample-level aggregation sensitivity, small-sample directional checks, ACTA2 removal, overlap-removed contractile–matricellular–LOX sensitivity, and outcome-independent stromal-compartment sensitivity. It does not contain plotting, figure rendering, manuscript production, data-download, or source-data workbook code.

## Quick start

1. Create the environment described in `environment/environment.yml`.
2. Place the four documented inputs in a local `inputs/` directory; see `documentation/INPUTS.md`.
3. Run `bash run_analysis.sh`.

The default output directory is `results/`. Both directories can be changed with `CORNEA_INPUT_DIR` and `CORNEA_OUTPUT_DIR`.

The expected numerical summaries are provided in `expected_results/`. Standardized values are specific to each dataset and are not interpreted on a common absolute scale.

## License

The code is available under the MIT License.

