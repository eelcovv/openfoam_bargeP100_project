
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Settings
# -----------------------------
TEMPLATE="cases/baseYaw"      # Your base case
TRISRC="triSurface/hull.stl"  # STL name inside each case
ANGLES=(0 15 30 45 60 75 90 105 120 135 150 165 180)

# Choose how to define the rotation origin:
# Option A: Fixed known centre (recommended if you designed to these extents)
ORIGIN_X=50
ORIGIN_Y=0
ORIGIN_Z=0   # rotate about Z through z=0 (yaw about waterline)

# Option B: Auto-detect STL bbox centre (uncomment to enable)
detect_origin() {
  # Requires OpenFOAM 'surfaceCheck'; parses min/max and computes midpoint
  local stl="$1"
  # Extract bbox lines and compute midpoint with awk
  surfaceCheck "$stl" > .bbox.txt
  # Example lines include "Bounding box : (xmin ymin zmin) (xmax ymax zmax)"
  local line
  line=$(grep -m1 -i "Bounding box" .bbox.txt)
  # Get numbers
  local xmin ymin zmin xmax ymax zmax
  read -r xmin ymin zmin xmax ymax zmax < <(
    echo "$line" | sed -E 's/.*\(([^)]*)\)\s*\(([^)]*)\).*/\1 \2/' \
      | awk '{print $1,$2,$3,$4,$5,$6}'
  )
  ORIGIN_X=$(awk -v a="$xmin" -v b="$xmax" 'BEGIN{print (a+b)/2}')
  ORIGIN_Y=$(awk -v a="$ymin" -v b="$ymax" 'BEGIN{print (a+b)/2}')
  ORIGIN_Z=$(awk -v a="$zmin" -v b="$zmax" 'BEGIN{print (a+b)/2}')
  echo "Auto origin = ($ORIGIN_X $ORIGIN_Y $ORIGIN_Z)"
}

# -----------------------------
# Main loop
# -----------------------------
for ang in "${ANGLES[@]}"; do
  CASE="runs/yaw_${ang}"
  echo ">>> Preparing $CASE"
  rsync -a --delete "$TEMPLATE/" "$CASE/"

  pushd "$CASE" >/dev/null

  # Option B: enable this to auto-detect the centre from STL
  # detect_origin "$TRISRC"

  # Rotate STL about Z by 'ang' degrees around chosen origin (degrees!)
  # 1) shift origin to (0,0,0), 2) yaw rotate, 3) shift back
  surfaceTransformPoints \
      "-origin ($ORIGIN_X $ORIGIN_Y $ORIGIN_Z) -yawPitchRoll ( $ang 0 0 )" \
      "$TRISRC" "$TRISRC.rot"

  # Overwrite the STL expected by snappy (keep filename stable)
  mv -f "$TRISRC.rot" "$TRISRC"

  # Fresh mesh + snap
  foamCleanPolyMesh || true
  blockMesh
  snappyHexMesh -overwrite

  # (Optional) layers later:
  # snappyHexMesh -overwrite -dict system/snappyHexMeshLayersDict

  # Solver run (choose yours)
  # potentialFoam -initialiseUBCs
  # simpleFoam

  # Forces (if configured in controlDict) will be written during run
  popd >/dev/null
done

echo "All headings done."

