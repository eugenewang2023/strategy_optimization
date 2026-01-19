#!/bin/bash
set -euo pipefail

method="kmeans"
#method="gmm"

python3 split_data.py \
  --data_dir data \
  --split-method $method
