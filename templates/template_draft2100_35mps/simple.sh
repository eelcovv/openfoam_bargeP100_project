#!/usr/bin/env bash
# simple.sh — single-node parallel simpleFoam runner with auto hostfile
# Usage:
#   ./simple.sh                 # gebruikt alle cores (nproc), verwacht dat je al decomposed hebt
#   NP=10 ./simple.sh           # forceer np=10
#   DECOMPOSE=1 ./simple.sh     # laat script zelf decomposePar doen
#   RECONSTRUCT=1 ./simple.sh   # reconstructParMesh na afloop
#   CHECK=1 ./simple.sh         # checkMesh na afloop
#   HOSTFILE=hf ./simple.sh     # eigen hostfile-naam
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

echo "localhost slots=${NP}" > "${HOSTFILE}"
echo "[simple.sh] hostfile -> ${HOSTFILE}:"
cat "${HOSTFILE}"

# 3) Optioneel decomposen
if [[ "${DECOMPOSE:-0}" == "1" ]]; then
  echo "[simple.sh] decomposePar -force"
  decomposePar -force
else
  if ! ls -d processor* >/dev/null 2>&1; then
    echo "[simple.sh] Waarschuwing: geen 'processor*' dirs gevonden. "
    echo "            Draai zelf 'decomposePar -force' of run met DECOMPOSE=1."
  fi
fi

echo "[simple.sh] ${MPI_BIN} --hostfile ${HOSTFILE} -np ${NP} ${BIND_FLAGS} simpleFoam -parallel"
${MPI_BIN} --hostfile "${HOSTFILE}" -np "${NP}" ${BIND_FLAGS} simpleFoam -parallel

echo "[simple.sh] Done."

