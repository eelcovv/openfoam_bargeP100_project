#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal yaw-sweep script with logging and two-phase runs.

Phase A (meshing only):
  clone → rotate STL → surfaceFeatureExtract → blockMesh →
  decomposePar → mpirun snappyHexMesh -parallel -overwrite →
  reconstructParMesh -constant → decomposePar -force
(Use --mesh-only)

Phase B (start existing solvers later):
  For each prepared case dir, run: mpirun <application> -parallel
(Use --start-existing)

Works with:
- Docker via `of-run` (auto-detected)
- Native OpenFOAM (no container), if OF binaries are on PATH
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------- Environment detection ----------------------------


def has_of_run() -> bool:
    """Return True if of-run is available."""
    return shutil.which("of-run") is not None


USE_OF_RUN = has_of_run()


def build_cmd(base: str) -> List[str]:
    """Return the command list, prefixed with 'of-run' when available."""
    if USE_OF_RUN:
        return ["of-run", *base.split()]
    # Native: run as-is via shell if we used pipes; here we pass a pure argv list.
    return base.split()


# ---------------------------- Command runners with logging ----------------------------


def _run_and_log(argv: List[str], case_dir: Path, log_file: Optional[Path]) -> None:
    """
    Run command (argv) in case_dir. If log_file is given, capture both stdout/stderr to it.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        res = subprocess.run(argv, cwd=str(case_dir))
    else:
        with open(log_file, "ab", buffering=0) as f:
            res = subprocess.run(
                argv, cwd=str(case_dir), stdout=f, stderr=subprocess.STDOUT
            )
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(argv)}")


def run_of(cmd: str, case_dir: Path, log_name: Optional[str] = None) -> None:
    """
    Run a (serial) OpenFOAM command in case_dir, logging to log_name if provided.
    """
    argv = build_cmd(cmd)
    log_file = (case_dir / log_name) if log_name else None
    print(
        f"[RUN] (in {case_dir}) {' '.join(argv)}"
        + (f"  -> {log_name}" if log_name else "")
    )
    _run_and_log(argv, case_dir, log_file)


def run_mpi(
    cmd: str,
    case_dir: Path,
    mpirun_cmd: str,
    np: int,
    hostfile: Optional[Path],
    extra: str,
    log_name: Optional[str],
) -> None:
    """
    Run an MPI command inside the OF environment (via of-run or native).
    """
    parts = [mpirun_cmd, f"-np {np}"]
    if hostfile:
        parts += ["--hostfile", str(hostfile)]
    if extra:
        parts += extra.split()
    parts += cmd.split()
    argv = build_cmd(" ".join(parts))
    log_file = (case_dir / log_name) if log_name else None
    print(
        f"[RUN] (in {case_dir}) {' '.join(argv)}"
        + (f"  -> {log_name}" if log_name else "")
    )
    _run_and_log(argv, case_dir, log_file)


# ---------------------------- STL rotation ----------------------------


def rotate_stl_with_surfaceTransformPoints(
    case_dir: Path,
    stl_rel: Path,
    yaw_deg: float,
    pivot: Tuple[float, float, float] = (50.0, 0.0, 0.0),
) -> None:
    """Rotate STL around z-axis by yaw_deg degrees about pivot using surfaceTransformPoints, logging to log.rotate."""
    stl_path = case_dir / stl_rel
    tri_dir = stl_path.parent
    tri_dir.mkdir(parents=True, exist_ok=True)

    src_name = stl_path.name
    tmp1 = f".tmp_rot1_{yaw_deg}.stl"
    tmp2 = f".tmp_rot2_{yaw_deg}.stl"
    tmp3 = f".tmp_rot3_{yaw_deg}.stl"

    x0, y0, z0 = pivot
    import math

    yaw_rad = math.radians(yaw_deg)

    # We run in tri_dir, but still resolve logs in the case root for simplicity.
    def rof(local_cmd: str):
        argv = build_cmd(local_cmd)
        logf = case_dir / "log.rotate"
        print(f"[RUN] (in {tri_dir}) {' '.join(argv)}  -> log.rotate")
        _run_and_log(argv, tri_dir, logf)

    rof(f"surfaceTransformPoints -translate '({-x0} {-y0} {-z0})' {src_name} {tmp1}")
    rof(f"surfaceTransformPoints -rollPitchYaw '(0 0 {yaw_rad})' {tmp1} {tmp2}")
    rof(f"surfaceTransformPoints -translate '({x0} {y0} {z0})' {tmp2} {tmp3}")

    (tri_dir / tmp3).replace(stl_path)
    for t in (tmp1, tmp2):
        try:
            (tri_dir / t).unlink()
        except FileNotFoundError:
            pass


# ---------------------------- Case helpers ----------------------------


def clone_template(template: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    def _ignore(dirpath, names):
        ignore = []
        for n in names:
            if n.startswith("processor") or n in {".git", ".venv", "__pycache__"}:
                ignore.append(n)
        return set(ignore)

    shutil.copytree(template, destination, ignore=_ignore)


def ensure_surface_feature_dict(case_dir: Path, stl_rel: Path) -> None:
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)
    dpath = sys_dir / "surfaceFeatureExtractDict"
    if dpath.exists():
        return
    dpath.write_text(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| Minimal surfaceFeatureExtractDict (auto-generated)                           |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object surfaceFeatureExtractDict;
}}
surfaces
(
    {{
        file "{stl_rel.as_posix()}";
        level 0;
        extractFromSurfaceFeatureEdges yes;
        writeObj yes;
        includedAngle 150;
    }}
);
""")


def ensure_decompose_dict(case_dir: Path, np: int) -> None:
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)
    dpath = sys_dir / "decomposeParDict"
    if not dpath.exists():
        dpath.write_text(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| decomposeParDict (auto-generated)                                            |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object decomposeParDict;
}}
numberOfSubdomains {np};
method scotch;
distributed no;
roots ();
""")
        return
    txt = dpath.read_text()
    if "numberOfSubdomains" in txt:
        txt = re.sub(
            r"numberOfSubdomains\\s+\\d+\\s*;", f"numberOfSubdomains {np};", txt
        )
    else:
        txt = f"numberOfSubdomains {np};\n" + txt
    dpath.write_text(txt)


def parse_application(case_dir: Path) -> str:
    ctrl = (case_dir / "system" / "controlDict").read_text()
    m = re.search(
        r"^\\s*application\\s+([A-Za-z0-9_./+-]+)\\s*;", ctrl, flags=re.MULTILINE
    )
    if not m:
        raise RuntimeError("Could not find application in controlDict.")
    return m.group(1)


def make_case_name(prefix: str, angle: float) -> str:
    return f"{prefix}{int(round(angle)):03d}"


# ---------------------------- Meshing pipeline per angle ----------------------------


def prepare_and_mesh_angle(
    template: Path,
    out_root: Path,
    angle: float,
    pivot: Tuple[float, float, float],
    np: int,
    mpirun_cmd: str,
    hostfile: Optional[Path],
    mpirun_extra: str,
    case_prefix: str,
) -> Path:
    case_name = make_case_name(case_prefix, angle)
    case_dir = out_root / case_name
    print(f"\n=== Prepare/Mesh angle {angle}° → {case_dir} ===")

    clone_template(template, case_dir)
    stl_rel = Path("constant/triSurface/hull.stl")
    if not (case_dir / stl_rel).exists():
        raise FileNotFoundError(f"Missing STL: {case_dir / stl_rel}")

    rotate_stl_with_surfaceTransformPoints(case_dir, stl_rel, angle, pivot)

    ensure_surface_feature_dict(case_dir, stl_rel)
    run_of("surfaceFeatureExtract", case_dir, log_name="log.surfaceFeatureExtract")

    run_of("blockMesh", case_dir, log_name="log.blockMesh")

    ensure_decompose_dict(case_dir, np)
    run_of("decomposePar", case_dir, log_name="log.decomposePar")

    run_mpi(
        "snappyHexMesh -parallel -overwrite",
        case_dir,
        mpirun_cmd,
        np,
        hostfile,
        mpirun_extra,
        log_name="log.snappyHexMesh",
    )

    run_of("reconstructParMesh -constant", case_dir, log_name="log.reconstructParMesh")

    # Re-decompose for solver so the case is solver-ready in phase B
    run_of("decomposePar -force", case_dir, log_name="log.decomposePar.solver")

    return case_dir


# ---------------------------- Start solvers for existing cases ----------------------------


def start_solver_for_existing_cases(
    out_root: Path,
    case_prefix: str,
    np: int,
    mpirun_cmd: str,
    hostfile: Optional[Path],
    mpirun_extra: str,
) -> None:
    """
    Iterate all subdirs in out_root starting with case_prefix and launch solver in parallel.
    """
    for sub in sorted(out_root.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith(case_prefix):
            continue
        # Make sure it looks like a prepared case
        if not (sub / "system" / "controlDict").exists():
            continue
        app = parse_application(sub)
        print(f"\n=== Start solver in {sub} (app={app}) ===")
        run_mpi(
            f"{app} -parallel",
            sub,
            mpirun_cmd,
            np,
            hostfile,
            mpirun_extra,
            log_name="log.simpleFoam",
        )


# ---------------------------- CLI ----------------------------


def parse_angles(spec: str) -> List[float]:
    return [float(tok.strip()) for tok in spec.split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Yaw sweep with logging (Docker via of-run or native)."
    )
    parser.add_argument("--template", type=Path, help="Template case path (phase A).")
    parser.add_argument(
        "--out-root", required=True, type=Path, help="Output root for cases."
    )
    parser.add_argument(
        "--case-prefix", default="yaw_", help="Case folder prefix, e.g., 'barge_'."
    )
    parser.add_argument("--x0", type=float, default=50.0)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument(
        "--angles", type=str, default="90,45,135,180,15,30,60,75,105,120,150"
    )
    parser.add_argument("--np", type=int, default=8)
    parser.add_argument("--mpirun", type=str, default="mpirun")
    parser.add_argument("--mpirun-extra", type=str, default="")
    parser.add_argument("--hostfile", type=Path, default=None)
    # Phase control:
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Phase A: prepare/mesh cases, do not start solver.",
    )
    parser.add_argument(
        "--start-existing",
        action="store_true",
        help="Phase B: start solvers for existing cases under --out-root.",
    )
    # Legacy flag (mutually exclusive with mesh-only/start-existing)
    parser.add_argument(
        "--start-solver",
        action="store_true",
        help="(Phase A) Start solver immediately after meshing.",
    )
    parser.add_argument("--no-start-solver", action="store_true")

    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pivot = (args.x0, args.y0, args.z0)

    mode = "Docker (of-run)" if USE_OF_RUN else "native OpenFOAM"
    print(f"[INFO] Mode: {mode}")

    # Phase B only
    if args.start_existing:
        start_solver_for_existing_cases(
            out_root=out_root,
            case_prefix=args.case_prefix,
            np=args.np,
            mpirun_cmd=args.mpirun,
            hostfile=args.hostfile.resolve() if args.hostfile else None,
            mpirun_extra=args.mpirun_extra.strip(),
        )
        print("\nAll existing cases started.")
        return

    # Phase A (prepare/mesh) – requires template and angles
    if args.template is None:
        raise SystemExit(
            "--template is required for meshing phase (omit it when using --start-existing)."
        )
    template = args.template.resolve()
    angles = parse_angles(args.angles)

    # Decide solver start for Phase A
    start_solver = False
    if args.mesh_only:
        start_solver = False
    else:
        start_solver = args.start_solver and not args.no_start_solver

    for ang in angles:
        case_dir = prepare_and_mesh_angle(
            template=template,
            out_root=out_root,
            angle=ang,
            pivot=pivot,
            np=args.np,
            mpirun_cmd=args.mpirun,
            hostfile=args.hostfile.resolve() if args.hostfile else None,
            mpirun_extra=args.mpirun_extra.strip(),
            case_prefix=args.case_prefix,
        )
        if start_solver:
            app = parse_application(case_dir)
            print(f"\n=== Start solver immediately in {case_dir} (app={app}) ===")
            run_mpi(
                f"{app} -parallel",
                case_dir,
                args.mpirun,
                args.np,
                args.hostfile,
                args.mpirun_extra.strip(),
                log_name="log.simpleFoam",
            )

    print(
        "\nDone. Cases prepared."
        + (" Solvers started." if start_solver else " (mesh-only)")
    )


if __name__ == "__main__":
    main()
