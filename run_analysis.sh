#!/usr/bin/env bash
set -euo pipefail

python analysis/core_sensitivity.py
python analysis/overlap_sensitivity.py
python analysis/compartment_sensitivity.py

