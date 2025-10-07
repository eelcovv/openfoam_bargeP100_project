#!/usr/bin/env bash
# snappy.sh — single-node parallel snappyHexMesh runner with auto hostfile
# Usage:
#   ./snappy.sh                 # gebruikt alle cores (nproc), verwacht dat je al decomposed hebt
#   NP=10 ./snappy.sh           # forceer np=10
#   DECOMPOSE=1 ./snappy.sh     # laat script zelf decomposePar doen
#   RECONSTRUCT=1 ./snappy.sh   # reconstructParMesh na afloop
#   CHECK=1 ./snappy.sh         # checkMesh na afloop
#   HOSTFILE=hf ./snappy.sh     # eigen hostfile-naam
#
# Env-vars:
#   NP:            aantal processen (default: nproc)
#   DECOMPOSE:     1 om decomposePar -force te draaien (default: 0)
#   RECONSTRUCT:   1 om reconstructParMesh -constant te draaien (default: 0)
#   CHECK:         1 om checkMesh te draaien (default: 0)
#   HOSTFILE:      pad/naam van hostfile (default: hostfile)
#   BIND_FLAGS:    extra mpirun bind/map flags (default: "--bind-to none --map-by slot")

set -euo pipefail

NP="${NP:-$(nproc)}"
HOSTFILE="${HOSTFILE:-hostfile}"
BIND_FLAGS="${BIND_FLAGS:---bind-to none --map-by slot}"
MPI_BIN="${MPI_BIN:-mpirun}"

# 1) Basischecks
[[ -f system/snappyHexMeshDict ]] || { echo "ERROR: system/snappyHexMeshDict ontbreekt"; exit 1; }

# 2) Hostfile op localhost (robuust in containers)
echo "localhost slots=${NP}" > "${HOSTFILE}"
echo "[snappy.sh] hostfile -> ${HOSTFILE}:"
cat "${HOSTFILE}"

# 3) Optioneel decomposen
if [[ "${DECOMPOSE:-0}" == "1" ]]; then
  echo "[snappy.sh] decomposePar -force"
  decomposePar -force
else
  if ! ls -d processor* >/dev/null 2>&1; then
    echo "[snappy.sh] Waarschuwing: geen 'processor*' dirs gevonden. "
    echo "            Draai zelf 'decomposePar -force' of run met DECOMPOSE=1."
  fi
fi

# 4) Snappy parallel
echo "[snappy.sh] ${MPI_BIN} --hostfile ${HOSTFILE} -np ${NP} ${BIND_FLAGS} snappyHexMesh -parallel"
${MPI_BIN} --hostfile "${HOSTFILE}" -np "${NP}" ${BIND_FLAGS} snappyHexMesh -parallel

# 5) Optioneel reconstruct + check
if [[ "${RECONSTRUCT:-0}" == "1" ]]; then
  echo "[snappy.sh] reconstructParMesh -constant -mergeTol 1e-6"
  reconstructParMesh -constant -mergeTol 1e-6
fi

if [[ "${CHECK:-0}" == "1" ]]; then
  echo "[snappy.sh] checkMesh -allGeometry -allTopology"
  checkMesh -allGeometry -allTopology
fi

echo "[snappy.sh] Klaar."

