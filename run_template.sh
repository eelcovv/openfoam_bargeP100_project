#!/usr/bin/env bash
set -euo pipefail
TEMPLATE="$1"        # bv. templates/template_draft1270_12mps
OUT="$2"             # bv. cases/draft1270_12mps
PREFIX="${3:-r}"     # default r
ANGLES="${4:-0,45,90,135,180}"
NP="${5:-16}"
MPIRUN="${6:-mpirun --bind-to none --map-by slot}"

python batch_process_barges.py \
  --template "$TEMPLATE" \
  --out-root "$OUT" \
  --case-prefix "$PREFIX" \
  --angles "$ANGLES" \
  --np "$NP" \
  --mpirun "$MPIRUN" \
  --start-solver


# use --mesh-only to only generate the mesh and skip the solver run
#  --mesh-only

# use --start-exist to run the solver on existing cases
#  --start-exist
