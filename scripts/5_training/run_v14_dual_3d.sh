#!/usr/bin/env bash
# Original V14 3D runner retired to prevent accidentally mixing incompatible
# 3D checkpoints/CSVs with the new 5D model.
set -euo pipefail
echo "The V14 3D experiment is preserved in results/metrics/v14_dual_3d_refiner_pilot." >&2
echo "V14 now uses 5D bottlenecks. Run: bash scripts/5_training/run_v14_dual_5d.sh" >&2
exit 2
