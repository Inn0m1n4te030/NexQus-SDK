#!/usr/bin/env bash
# Run on the maintainer's CUDA machine from a checkout of this exact candidate.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e '.[tutorial,release]'
python -m ipykernel install --user --name nexqus --display-name "Python (NexQus)"
python scripts/check_release.py --require-gpu --public --report docs/release_cuda_validation.json
python scripts/run_notebooks.py --engine jupyter --kernel nexqus --require-gpu --report docs/notebook_cuda_execution.json
echo "CUDA and real Jupyter release checks passed with GPU required."
