#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor OpenFOAM case time traces (solverInfo + forces) using a YAML recipe.

Usage:
    python monitor_case.py <case_folder> [--config monitor.yaml] [--out-dir plots] [--show]

YAML schema (minimal):
figures:
  - title: <figure title>
    yscale: log | linear           # optional (default: linear)
    series:
      - source: solver | forces    # required
        field:  <column or key>    # e.g. 'p_final', 'Ux_final', 'total_x'
        label:  <legend label>     # optional (defaults to field)

Dependencies:
    pyyaml, pandas, matplotlib

Notes:
- solverInfo: parses header line to get exact column names.
- forces: parses standard OpenFOAM forces function output (pressure & viscous
  forces/moments). Exposes:
    fpx,fpy,fpz, fvx,fvy,fvz, mpx,mpy,mpz, mvx,mvy,mvz
    ftx,fty,ftz (totals = pressure + viscous)
    mtx,mty,mtz (totals = pressure + viscous)
  And convenience aliases: total_x,total_y,total_z; moment_total_x, etc.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


# --------------------------- Filesystem helpers -----------------------------


def find_single_file(root: Path, pattern_parts: List[str]) -> Optional[Path]:
    """
    Walk under `root` and return the first file whose relative path components
    match the sequence in `pattern_parts`, where '*' means 'any single directory'.

    Example:
        find_single_file(case, ["postProcessing", "solverInfo", "*", "solverInfo.dat"])
    """
    # Build a glob pattern like 'postProcessing/solverInfo/*/solverInfo.dat'
    pattern = str(Path(*pattern_parts))
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


# --------------------------- solverInfo parsing -----------------------------


def load_solverinfo(case_dir: Path) -> Optional[pd.DataFrame]:
    """
    Read solverInfo.dat into a DataFrame. Returns None if not found.
    """
    fpath = find_single_file(
        case_dir, ["postProcessing", "solverInfo", "*", "solverInfo.dat"]
    )
    if fpath is None or not fpath.is_file():
        return None

    times: List[float] = []
    rows: List[List[float]] = []
    header_cols: List[str] = []

    with fpath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Time"):
                # Extract column names from header line (split on whitespace).
                header_cols = [c for c in line[1:].split() if c]  # remove leading '#'
                continue
            if line.startswith("#"):
                continue
            # Data line: split by whitespace; columns must match header
            parts = line.split()
            if not header_cols:
                raise ValueError(f"solverInfo: header not found before data in {fpath}")
            if len(parts) < len(header_cols):
                # Sometimes fields like 'true/false' appear; keep as strings but we need numerics
                # Convert booleans to 0/1 and non-numerics to NaN.
                # Pad to header length if needed.
                parts += [""] * (len(header_cols) - len(parts))

            # Convert fields to float when possible, booleans to float, else NaN
            def to_num(x: str) -> float:
                if x.lower() == "true":
                    return 1.0
                if x.lower() == "false":
                    return 0.0
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            row = [to_num(p) for p in parts[: len(header_cols)]]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header_cols)
    # Standardize time column name to 'time' (lowercase) for plotting
    if "Time" in df.columns:
        df.rename(columns={"Time": "time"}, inplace=True)
    return df


# --------------------------- forces.dat parsing -----------------------------

_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")


def _extract_floats(line: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(line)]


def load_forces(case_dir: Path) -> Optional[pd.DataFrame]:
    """
    Parse OpenFOAM forces function output into a tidy DataFrame with columns:
        time,
        fpx,fpy,fpz, fvx,fvy,fvz,
        mpx,mpy,mpz, mvx,mvy,mvz,
        ftx,fty,ftz, mtx,mty,mtz
    Returns None if not found.
    """
    fpath = find_single_file(case_dir, ["postProcessing", "forces", "*", "forces.dat"])
    if fpath is None or not fpath.is_file():
        return None

    rows: List[List[float]] = []
    with fpath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = _extract_floats(s)
            # Expected sequence (classic forces function):
            # t,
            # (fpx fpy fpz) (fvx fvy fvz) (mpx mpy mpz) (mvx mvy mvz)
            if len(vals) < 13:
                # Some variants include additional columns; we accept minimal.
                # If malformed, skip line.
                continue
            t = vals[0]
            fpx, fpy, fpz = vals[1:4]
            fvx, fvy, fvz = vals[4:7]
            mpx, mpy, mpz = vals[7:10]
            mvx, mvy, mvz = vals[10:13]
            ftx, fty, ftz = fpx + fvx, fpy + fvy, fpz + fvz
            mtx, mty, mtz = mpx + mvx, mpy + mvy, mpz + mvz
            rows.append(
                [
                    t,
                    fpx,
                    fpy,
                    fpz,
                    fvx,
                    fvy,
                    fvz,
                    mpx,
                    mpy,
                    mpz,
                    mvx,
                    mvy,
                    mvz,
                    ftx,
                    fty,
                    ftz,
                    mtx,
                    mty,
                    mtz,
                ]
            )

    cols = [
        "time",
        "fpx",
        "fpy",
        "fpz",
        "fvx",
        "fvy",
        "fvz",
        "mpx",
        "mpy",
        "mpz",
        "mvx",
        "mvy",
        "mvz",
        "ftx",
        "fty",
        "ftz",
        "mtx",
        "mty",
        "mtz",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Convenience aliases for YAML 'field' names:
    # total_x/y/z and moment_total_x/y/z
    df["total_x"], df["total_y"], df["total_z"] = df["ftx"], df["fty"], df["ftz"]
    df["moment_total_x"], df["moment_total_y"], df["moment_total_z"] = (
        df["mtx"],
        df["mty"],
        df["mtz"],
    )
    return df


# --------------------------- Plotting engine -------------------------------


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^\w\-+.]+", "_", name.strip())
    return safe[:120] if len(safe) > 120 else safe


def plot_figures(case_dir: Path, cfg: Dict, out_dir: Path, show: bool) -> int:
    """
    Render all figures from YAML config. Returns number of figures plotted.
    """
    solver_df = load_solverinfo(case_dir)
    forces_df = load_forces(case_dir)

    figures = cfg.get("figures", [])
    if not isinstance(figures, list) or not figures:
        print("No figures defined in YAML (expected key 'figures').", file=sys.stderr)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0

    for fig in figures:
        title = str(fig.get("title", "figure"))
        yscale = str(fig.get("yscale", "linear")).lower()
        series = fig.get("series", [])
        if not isinstance(series, list) or not series:
            print(f"Skipping figure '{title}': no 'series' list.", file=sys.stderr)
            continue

        plt.figure()
        ax = plt.gca()
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_yscale(yscale if yscale in ("linear", "log", "symlog") else "linear")

        any_plotted = False

        for s in series:
            source = str(s.get("source", "")).lower()
            field = str(s.get("field", ""))
            label = s.get("label", field)

            if source == "solver":
                if solver_df is None:
                    print(
                        f"  [warn] solverInfo not found for case; cannot plot '{field}'.",
                        file=sys.stderr,
                    )
                    continue
                if field not in solver_df.columns:
                    # Try common variants like 'U_final' to per-component fields if needed.
                    if field == "U_final":
                        # aggregate magnitude from components if available
                        comps = [
                            c
                            for c in ("Ux_final", "Uy_final", "Uz_final")
                            if c in solver_df.columns
                        ]
                        if comps:
                            y = solver_df[comps[0]] ** 2
                            for c in comps[1:]:
                                y = y + solver_df[c] ** 2
                            y = y.pow(0.5)
                            ax.plot(solver_df["time"], y, label=label)
                            any_plotted = True
                            continue
                    print(
                        f"  [warn] solver field '{field}' not in columns.",
                        file=sys.stderr,
                    )
                    continue
                ax.plot(solver_df["time"], solver_df[field], label=label)
                any_plotted = True

            elif source == "forces":
                if forces_df is None:
                    print(
                        f"  [warn] forces.dat not found for case; cannot plot '{field}'.",
                        file=sys.stderr,
                    )
                    continue
                if field not in forces_df.columns:
                    print(
                        f"  [warn] forces field '{field}' not available.",
                        file=sys.stderr,
                    )
                    continue
                ax.plot(forces_df["time"], forces_df[field], label=label)
                any_plotted = True

            else:
                print(
                    f"  [warn] unknown source '{source}' (use 'solver' or 'forces').",
                    file=sys.stderr,
                )
                continue

        if not any_plotted:
            plt.close()
            print(f"Skipping figure '{title}': nothing plotted.", file=sys.stderr)
            continue

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fname = sanitize_filename(title) + ".png"
        out_path = out_dir / fname
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Wrote {out_path}")
        plotted += 1

    return plotted


# --------------------------- CLI -------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot OpenFOAM case time traces from YAML recipe."
    )
    ap.add_argument(
        "case_folder",
        type=Path,
        help="Path to case folder (contains postProcessing/...)",
    )
    ap.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("monitor.yaml"),
        help="YAML config path (default: monitor.yaml in repo root)",
    )
    ap.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to write PNGs (default: <case>/postProcessing/monitor)",
    )
    ap.add_argument(
        "--show", action="store_true", help="Show figures interactively after saving"
    )
    args = ap.parse_args()

    case_dir: Path = args.case_folder.resolve()
    if not case_dir.is_dir():
        print(f"Case folder not found: {case_dir}", file=sys.stderr)
        return 2

    cfg_path: Path = args.config.resolve()
    if not cfg_path.is_file():
        print(f"YAML config not found: {cfg_path}", file=sys.stderr)
        return 2

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    out_dir = args.out_dir or (case_dir / "postProcessing" / "monitor")
    out_dir = out_dir.resolve()

    n = plot_figures(case_dir, cfg, out_dir, show=args.show)
    if n == 0:
        print("No figures produced.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
