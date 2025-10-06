"""
Prepare and (optionally) run a yaw sweep with parallel snappyHexMesh and solver.

Per angle:
  1) Clone template case
  2) Rotate constant/triSurface/hull.stl around z-axis about (x0,y0,z0)
  3) Ensure surfaceFeatureExtractDict, then run surfaceFeatureExtract (serial)
  4) Run blockMesh (serial)
  5) Ensure/update decomposeParDict with numberOfSubdomains = --np
  6) decomposePar
  7) mpirun: snappyHexMesh -parallel -overwrite
  8) reconstructParMesh -constant
  9) decomposePar -force
 10) mpirun: <application from controlDict> -parallel   (optional via --start-solver)

Run from outside the OpenFOAM shell; the script tries to source common bashrcs or
uses 'of-run' if present.
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------- OpenFOAM environment runner ----------------------------


def guess_of_bashrc_paths() -> List[Path]:
    """Return likely OpenFOAM bashrc locations to try in order."""
    candidates = [
        "/usr/lib/openfoam/openfoam2406/etc/bashrc",
        "/opt/openfoam2406/etc/bashrc",
        "/usr/lib/openfoam/openfoam2312/etc/bashrc",
        "/usr/lib/openfoam/openfoam2306/etc/bashrc",
    ]
    return [Path(p) for p in candidates if Path(p).exists()]


def run_of(cmd: str, case_dir: Path, extra_env: Optional[dict] = None) -> None:
    """
    Run a command with the OpenFOAM environment loaded.
    Prefers 'of-run' if available, otherwise sources a bashrc and runs the command.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    bashrcs = guess_of_bashrc_paths()
    sourced = " || ".join([f"source '{p}'" for p in bashrcs]) if bashrcs else ""
    of_run = shutil.which("of-run")

    if of_run:
        full_cmd = f"{of_run} {cmd}"
    elif sourced:
        full_cmd = f'bash -lc "({sourced}) >/dev/null 2>&1; {cmd}"'
    else:
        full_cmd = cmd  # last resort; may fail if OF env not present

    print(f"[RUN] (in {case_dir}) {cmd}")
    res = subprocess.run(full_cmd, shell=True, cwd=str(case_dir), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}): {cmd}")


def run_mpi(
    cmd: str,
    case_dir: Path,
    mpirun_cmd: str,
    np: int,
    hostfile: Optional[Path],
    extra: str = "",
) -> None:
    """
    Launch an MPI-enabled OpenFOAM command in parallel.
    Example: mpirun -np 8 [--hostfile X] snappyHexMesh -parallel -overwrite
    """
    pieces = [mpirun_cmd, f"-np {np}"]
    if hostfile:
        pieces.append(f"--hostfile {hostfile}")
    if extra:
        pieces.append(extra)
    pieces.append(cmd)
    run_of(" ".join(pieces), case_dir)


# ---------------------------- STL rotation helpers ----------------------------


def rotate_stl_with_surfaceTransformPoints(
    case_dir: Path,
    stl_rel: Path,
    yaw_deg: float,
    pivot: Tuple[float, float, float] = (50.0, 0.0, 0.0),
) -> None:
    """Rotate STL around z-axis by yaw_deg degrees about pivot using surfaceTransformPoints."""
    stl_path = case_dir / stl_rel
    tri_dir = stl_path.parent
    tmp1 = tri_dir / f".tmp_rot1_{yaw_deg}.stl"
    tmp2 = tri_dir / f".tmp_rot2_{yaw_deg}.stl"
    tmp3 = tri_dir / f".tmp_rot3_{yaw_deg}.stl"

    x0, y0, z0 = pivot

    # translate (-pivot)
    run_of(
        f"surfaceTransformPoints -translate '({-x0} {-y0} {-z0})' {stl_path.name} {tmp1.name}",
        case_dir,
    )
    # rotate (rollPitchYaw expects radians)
    import math

    yaw_rad = math.radians(yaw_deg)
    run_of(
        f"surfaceTransformPoints -rollPitchYaw '(0 0 {yaw_rad})' {tmp1.name} {tmp2.name}",
        case_dir,
    )
    # translate back (+pivot)
    run_of(
        f"surfaceTransformPoints -translate '({x0} {y0} {z0})' {tmp2.name} {tmp3.name}",
        case_dir,
    )

    tmp3.replace(stl_path)
    for t in (tmp1, tmp2):
        try:
            t.unlink()
        except FileNotFoundError:
            pass


# ---------------------------- Case + dict helpers ----------------------------


def clone_template(template: Path, destination: Path) -> None:
    """Clone the template case directory into destination."""
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    def _ignore(dirpath, names):
        ignore_list = []
        for n in names:
            if n.startswith("processor"):
                ignore_list.append(n)
            if n in {".git", ".venv", "__pycache__"}:
                ignore_list.append(n)
        return set(ignore_list)

    shutil.copytree(template, destination, ignore=_ignore)


def ensure_surface_feature_dict(case_dir: Path, stl_rel: Path) -> None:
    """Ensure system/surfaceFeatureExtractDict exists and references the STL; create minimal one if missing."""
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)
    dict_path = sys_dir / "surfaceFeatureExtractDict"
    if dict_path.exists():
        return
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| Minimal surfaceFeatureExtractDict generated by prepare_yaw_sweep.py          |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}}

surfaces
(
    {{
        file    "{stl_rel.as_posix()}";
        level   0;
        extractFromSurfaceFeatureEdges yes;
        writeObj yes;
        includedAngle 150;
    }}
);
"""
    dict_path.write_text(content)


def ensure_decompose_dict(case_dir: Path, np: int) -> None:
    """
    Ensure system/decomposeParDict exists and set numberOfSubdomains to np.
    If it exists, only update numberOfSubdomains; leave method and coefficients intact.
    """
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)
    dpath = sys_dir / "decomposeParDict"

    template = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| decomposeParDict generated/updated by prepare_yaw_sweep.py                   |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {np};

method          scotch;

distributed     no;

roots           ();

"""
    if not dpath.exists():
        dpath.write_text(template)
        return

    # Update numberOfSubdomains in-place
    txt = dpath.read_text()
    if "numberOfSubdomains" in txt:
        txt = re.sub(r"numberOfSubdomains\s+\d+\s*;", f"numberOfSubdomains {np};", txt)
    else:
        # Prepend at top after FoamFile block
        m = re.search(r"(FoamFile.*?\}\s*)", txt, flags=re.DOTALL)
        if m:
            idx = m.end()
            txt = txt[:idx] + f"\nnumberOfSubdomains {np};\n" + txt[idx:]
        else:
            txt = f"numberOfSubdomains {np};\n" + txt
    dpath.write_text(txt)


def parse_application(case_dir: Path) -> str:
    """Read system/controlDict and extract the application name."""
    ctrl = (case_dir / "system" / "controlDict").read_text()
    m = re.search(
        r"^\s*application\s+([A-Za-z0-9_./+-]+)\s*;", ctrl, flags=re.MULTILINE
    )
    if not m:
        raise RuntimeError("Could not find 'application' in system/controlDict.")
    return m.group(1)


# ---------------------------- Main sweep logic ----------------------------


def prepare_and_run_angle(
    template: Path,
    out_root: Path,
    angle: float,
    pivot: Tuple[float, float, float],
    run_solver: bool,
    np: int,
    mpirun_cmd: str,
    hostfile: Optional[Path],
    mpirun_extra: str,
) -> Path:
    """Prepare a single angle case and run the full parallel pipeline."""
    case_name = f"yaw_{int(angle) if float(angle).is_integer() else angle}deg"
    case_dir = out_root / case_name
    print(f"\n=== Preparing angle {angle}° → {case_dir} ===")

    clone_template(template, case_dir)

    # STL must exist
    stl_rel = Path("constant/triSurface/hull.stl")
    if not (case_dir / stl_rel).exists():
        raise FileNotFoundError(f"Missing STL: {case_dir / stl_rel}")

    # Rotate STL
    rotate_stl_with_surfaceTransformPoints(case_dir, stl_rel, angle, pivot=pivot)

    # Features (serial)
    ensure_surface_feature_dict(case_dir, stl_rel)
    run_of("surfaceFeatureExtract", case_dir)

    # blockMesh (serial)
    run_of("blockMesh", case_dir)

    # Prepare decomposeParDict and decompose
    ensure_decompose_dict(case_dir, np)
    run_of("decomposePar", case_dir)

    # Parallel snappyHexMesh
    run_mpi(
        "snappyHexMesh -parallel -overwrite",
        case_dir,
        mpirun_cmd,
        np,
        hostfile,
        mpirun_extra,
    )

    # Reconstruct mesh into constant/ (keeps fields in processors)
    run_of("reconstructParMesh -constant", case_dir)

    # Re-decompose for solver (ensures consistent partitions after mesh change)
    run_of("decomposePar -force", case_dir)

    # Solver (parallel)
    if run_solver:
        app = parse_application(case_dir)
        run_mpi(f"{app} -parallel", case_dir, mpirun_cmd, np, hostfile, mpirun_extra)

    return case_dir


def parse_angles(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and run a yaw sweep (parallel snappy + solver)."
    )
    parser.add_argument(
        "--template", required=True, type=Path, help="Path to the template case."
    )
    parser.add_argument(
        "--out-root", required=True, type=Path, help="Output root for generated cases."
    )
    parser.add_argument(
        "--x0", type=float, default=50.0, help="Pivot x (default: 50.0)."
    )
    parser.add_argument("--y0", type=float, default=0.0, help="Pivot y (default: 0.0).")
    parser.add_argument("--z0", type=float, default=0.0, help="Pivot z (default: 0.0).")
    parser.add_argument(
        "--angles",
        type=str,
        default="90,45,135,180,15,30,60,75,105,120,150",
        help="Comma-separated list of angles to process, in order.",
    )
    parser.add_argument(
        "--start-solver", action="store_true", help="Run the solver after meshing."
    )
    parser.add_argument(
        "--no-start-solver", action="store_true", help="Do not run the solver."
    )
    parser.add_argument(
        "--np",
        type=int,
        default=8,
        help="MPI ranks for decompose/snappy/solver (default: 8).",
    )
    parser.add_argument(
        "--mpirun",
        type=str,
        default="mpirun",
        help="MPI launcher command (default: 'mpirun').",
    )
    parser.add_argument(
        "--hostfile", type=Path, default=None, help="Optional path to an MPI hostfile."
    )
    parser.add_argument(
        "--mpirun-extra",
        type=str,
        default="",
        help="Extra args appended after -np (e.g., '--bind-to none --map-by slot').",
    )

    args = parser.parse_args()
    template = args.template.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_solver = args.start_solver and not args.no_start_solver
    pivot = (args.x0, args.y0, args.z0)
    angles = parse_angles(args.angles)

    for ang in angles:
        prepare_and_run_angle(
            template=template,
            out_root=out_root,
            angle=ang,
            pivot=pivot,
            run_solver=run_solver,
            np=args.np,
            mpirun_cmd=args.mpirun,
            hostfile=args.hostfile.resolve() if args.hostfile else None,
            mpirun_extra=args.mpirun_extra.strip(),
        )

    print(
        "\nAll requested angles prepared." + (" Solvers started." if run_solver else "")
    )


if __name__ == "__main__":
    main()
