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
            xlim: [min,max]        # optional (default: None)
            ylim: [min,max]        # optional (default: None)
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
import math
import numpy as np
import re
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

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
    Load forces (force[s].dat) and, if present, moments (moment[s].dat).
    Returns a merged DataFrame on 'time' with columns:
      # forces (tabular or classic parsed):
      time, total_x,total_y,total_z,
            pressure_x,pressure_y,pressure_z,
            viscous_x, viscous_y, viscous_z,
      # moments (tabular):
      moment_total_x, moment_total_y, moment_total_z,
      moment_pressure_x, moment_pressure_y, moment_pressure_z,
      moment_viscous_x,  moment_viscous_y,  moment_viscous_z
    """

    def _read_tabular(fpath: Path) -> Optional[pd.DataFrame]:
        header_cols: List[str] = []
        data_rows: List[List[float]] = []
        with fpath.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("# Time"):
                    header_cols = [c for c in s.lstrip("#").strip().split() if c]
                    continue
                if s.startswith("#"):
                    continue
                if header_cols:
                    parts = s.split()
                    vals: List[float] = []
                    for p in parts[: len(header_cols)]:
                        try:
                            vals.append(float(p))
                        except Exception:
                            vals.append(float("nan"))
                    if vals:
                        data_rows.append(vals)
        if header_cols and data_rows:
            df = pd.DataFrame(data_rows, columns=header_cols)
            if "Time" in df.columns:
                df.rename(columns={"Time": "time"}, inplace=True)
            return df
        return None

    def _read_classic_forces(fpath: Path) -> Optional[pd.DataFrame]:
        rows: List[List[float]] = []
        with fpath.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                vals = _extract_floats(s)
                if len(vals) < 13:
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
        if not rows:
            return None
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
            "total_x",
            "total_y",
            "total_z",
            "moment_total_x",
            "moment_total_y",
            "moment_total_z",
        ]
        df = pd.DataFrame(rows, columns=cols)
        # provide pressure/viscous split aliases
        df["pressure_x"], df["pressure_y"], df["pressure_z"] = (
            df["fpx"],
            df["fpy"],
            df["fpz"],
        )
        df["viscous_x"], df["viscous_y"], df["viscous_z"] = (
            df["fvx"],
            df["fvy"],
            df["fvz"],
        )
        # also expose moment pressure/viscous if useful
        df["moment_pressure_x"], df["moment_pressure_y"], df["moment_pressure_z"] = (
            df["mpx"],
            df["mpy"],
            df["mpz"],
        )
        df["moment_viscous_x"], df["moment_viscous_y"], df["moment_viscous_z"] = (
            df["mvx"],
            df["mvy"],
            df["mvz"],
        )
        return df

    def _find_first(patterns: List[List[str]]) -> Optional[Path]:
        for parts in patterns:
            p = find_single_file(case_dir, parts)
            if p and p.is_file():
                return p
        return None

    # --------- FORCES ----------
    forces_path = _find_first(
        [
            ["postProcessing", "forces", "*", "force.dat"],
            ["postProcessing", "forces", "*", "forces.dat"],
        ]
    )
    forces_df: Optional[pd.DataFrame] = None
    if forces_path:
        # try tabular first
        forces_df = _read_tabular(forces_path)
        if forces_df is not None:
            # normalize/aliases expected by YAML
            # ensure required columns exist
            for a, b in [
                ("total_x", "total_x"),
                ("total_y", "total_y"),
                ("total_z", "total_z"),
                ("pressure_x", "pressure_x"),
                ("pressure_y", "pressure_y"),
                ("pressure_z", "pressure_z"),
                ("viscous_x", "viscous_x"),
                ("viscous_y", "viscous_y"),
                ("viscous_z", "viscous_z"),
            ]:
                if b in forces_df.columns and a not in forces_df.columns:
                    forces_df[a] = forces_df[b]
        else:
            # classic fallback
            forces_df = _read_classic_forces(forces_path)

    # --------- MOMENTS ----------
    moments_path = _find_first(
        [
            ["postProcessing", "moment", "*", "moment.dat"],
            ["postProcessing", "moment", "*", "moments.dat"],
        ]
    )
    moments_df: Optional[pd.DataFrame] = None
    if moments_path:
        moments_df = _read_tabular(moments_path)
        if moments_df is not None:
            # Expect columns: time, total_x,total_y,total_z, pressure_x.., viscous_x..
            # add explicit 'moment_*' aliases
            rename_map = {}
            if "Time" in moments_df.columns:  # safety; _read_tabular already fixes this
                rename_map["Time"] = "time"
            moments_df.rename(columns=rename_map, inplace=True)
            for a, b in [
                ("moment_total_x", "total_x"),
                ("moment_total_y", "total_y"),
                ("moment_total_z", "total_z"),
                ("moment_pressure_x", "pressure_x"),
                ("moment_pressure_y", "pressure_y"),
                ("moment_pressure_z", "pressure_z"),
                ("moment_viscous_x", "viscous_x"),
                ("moment_viscous_y", "viscous_y"),
                ("moment_viscous_z", "viscous_z"),
            ]:
                if b in moments_df.columns and a not in moments_df.columns:
                    moments_df[a] = moments_df[b]

            # reduce to the columns we care about
            keep_cols = [
                "time",
                "moment_total_x",
                "moment_total_y",
                "moment_total_z",
                "moment_pressure_x",
                "moment_pressure_y",
                "moment_pressure_z",
                "moment_viscous_x",
                "moment_viscous_y",
                "moment_viscous_z",
            ]
            existing = [c for c in keep_cols if c in moments_df.columns]
            moments_df = moments_df[existing]

    # --------- MERGE ----------
    if forces_df is None and moments_df is None:
        return None
    if forces_df is None:
        return moments_df
    if moments_df is None:
        return forces_df

    merged = pd.merge(forces_df, moments_df, on="time", how="outer").sort_values("time")
    merged.reset_index(drop=True, inplace=True)
    return merged


# --------------------------- Plotting engine -------------------------------


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^\w\-+.]+", "_", name.strip())
    return safe[:120] if len(safe) > 120 else safe


def resolve_limits_from_data(
    limits: Optional[list[float]],
    data: Optional[pd.DataFrame],
    xcol: str = "time",
) -> Optional[tuple[float | None, float | None]]:
    """Convert [min,max] where None means use data min/max."""
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("Limits must be a list of two numbers [min, max]")
    lo, hi = limits
    if data is not None and xcol in data.columns:
        xmin, xmax = float(data[xcol].min()), float(data[xcol].max())
        if lo in (None, "None"):
            lo = xmin
        if hi in (None, "None"):
            hi = xmax
    return (
        float(lo) if lo is not None else None,
        float(hi) if hi is not None else None,
    )


def _series_xy(
    s: dict,
    solver_df: Optional[pd.DataFrame],
    forces_df: Optional[pd.DataFrame],
) -> Optional[tuple[pd.Series, pd.Series]]:
    source = str(s.get("source", "")).lower()
    field = str(s.get("field", ""))
    if source == "solver":
        if solver_df is None:
            return None
        if field == "U_final":
            comps = [
                c
                for c in ("Ux_final", "Uy_final", "Uz_final")
                if c in solver_df.columns
            ]
            if not comps:
                return None
            y = solver_df[comps[0]] ** 2
            for c in comps[1:]:
                y = y + solver_df[c] ** 2
            y = y.pow(0.5)
            return solver_df["time"], y
        if field not in solver_df.columns:
            return None
        return solver_df["time"], solver_df[field]

    if source == "forces":
        if forces_df is None or field not in forces_df.columns:
            return None
        return forces_df["time"], forces_df[field]

    return None


def compute_ylim_from_series(
    fig_cfg: dict,
    solver_df: Optional[pd.DataFrame],
    forces_df: Optional[pd.DataFrame],
    xlim_tuple: Optional[tuple[float | None, float | None]],
    yscale: str = "linear",
) -> Optional[tuple[float, float]]:
    ymins: list[float] = []
    ymaxs: list[float] = []
    series = fig_cfg.get("series", []) or []
    if not series:
        return None

    xlo, xhi = (None, None) if not xlim_tuple else xlim_tuple

    for s in series:
        xy = _series_xy(s, solver_df, forces_df)
        if xy is None:
            continue
        t, y = xy

        mask = pd.Series(True, index=t.index)
        if xlo is not None:
            mask &= t >= xlo
        if xhi is not None:
            mask &= t <= xhi

        ysel = y[mask].replace([np.inf, -np.inf], np.nan).dropna()
        if yscale == "log":
            ysel = ysel[ysel > 0]

        if not ysel.empty:
            ymins.append(float(ysel.min()))
            ymaxs.append(float(ysel.max()))

    if not ymins:
        return None

    ymin, ymax = min(ymins), max(ymaxs)

    if not math.isfinite(ymin) or not math.isfinite(ymax):
        return None
    if ymin == ymax:
        if yscale == "log":
            ymin, ymax = ymin / 10.0, ymax * 10.0
        else:
            eps = 1e-12 if ymin == 0 else abs(ymin) * 0.05
            ymin, ymax = ymin - eps, ymax + eps
    else:
        if yscale == "log":
            ymin, ymax = ymin / 1.2, ymax * 1.2
            if ymin <= 0:
                ymin = min(v for v in ymins if v > 0) / 1.2
        else:
            span = ymax - ymin
            ymin, ymax = ymin - 0.05 * span, ymax + 0.05 * span

    return ymin, ymax


def plot_figures(
    case_dir: Path, cfg: Dict, out_dir: Path, show: bool, live: bool = False
) -> int:
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

        data_df = forces_df if forces_df is not None else solver_df
        xlim = resolve_limits_from_data(fig.get("xlim"), data_df)

        plt.figure()
        ax = plt.gca()
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_yscale(yscale if yscale in ("linear", "log", "symlog") else "linear")

        if xlim:
            ax.set_xlim(xlim)

        y_auto = compute_ylim_from_series(
            fig, solver_df, forces_df, xlim, yscale=yscale
        )
        ylim_cfg = fig.get("ylim")
        if isinstance(ylim_cfg, list) and len(ylim_cfg) == 2:
            lo, hi = ylim_cfg
            if (lo in (None, "None")) and y_auto:
                lo = y_auto[0]
            if (hi in (None, "None")) and y_auto:
                hi = y_auto[1]
            ylim = (
                (float(lo), float(hi)) if (lo is not None and hi is not None) else None
            )
        else:
            ylim = y_auto

        if ylim:
            ax.set_ylim(ylim)

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
        print("Saving figure:", out_path)
        plt.savefig(out_path, dpi=120)

        if live and show:
            # Interactieve/non-blocking update
            plt.ion()
            try:
                plt.show(block=False)
            except TypeError:
                # Oudere Matplotlibs zonder block-arg
                plt.show()
            plt.pause(0.001)  # event loop laten ademen
            # NIET sluiten bij live-mode
        else:
            if show:
                plt.show()
            plt.close()

        plotted += 1

    return plotted


# --------------------------- Live watch & serve -----------------------------


def collect_targets(case_dir: Path) -> list[Path]:
    """Return the files we should watch for changes (solverInfo/forces/moment)."""
    pats = [
        ["postProcessing", "solverInfo", "*", "solverInfo.dat"],
        ["postProcessing", "forces", "*", "force.dat"],
        ["postProcessing", "forces", "*", "forces.dat"],
        ["postProcessing", "moment", "*", "moment.dat"],
        ["postProcessing", "moment", "*", "moments.dat"],
    ]
    files: list[Path] = []
    for parts in pats:
        p = find_single_file(case_dir, parts)
        if p:
            files.append(p)
    return files


def mtime_signature(paths: list[Path]) -> float:
    """Combine mtimes tot één signatuur; als er iets verandert, verandert deze."""
    sig = 0.0
    for p in paths:
        try:
            sig += p.stat().st_mtime
        except FileNotFoundError:
            sig += 0.0
    return sig


def write_autoindex(out_dir: Path):
    """Maak een simpele index.html met auto-refresh en thumbnails."""
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p.name for p in out_dir.glob("*.png")])
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>OpenFOAM monitor</title>",
        "<meta http-equiv='refresh' content='2'>",
        "<style>body{font-family:sans-serif;margin:20px} img{max-width:95vw; height:auto; display:block; margin:16px 0;}</style>",
        "<h1>OpenFOAM monitor</h1>",
        "<p>Auto-refresh every 2 seconds.</p>",
    ]
    for im in imgs:
        lines.append(f"<h3>{im}</h3>")
        lines.append(f"<img src='{im}?t={int(time.time())}'>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def run_http_server(root: Path, port: int):
    """Start een eenvoudige HTTP-server op aparte thread."""

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):
            pass  # stil houden

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"Serving {root} at http://localhost:{port}")
    return httpd


def live_watch(
    case_dir: Path,
    cfg: dict,
    out_dir: Path,
    cfg_path: Path,
    show: bool,
    interval: float,
    serve_port: int | None,
):
    targets = collect_targets(case_dir)

    def files_sig(paths: list[Path]) -> float:
        return mtime_signature(paths) if paths else 0.0

    prev_data_sig = -1.0
    prev_cfg_mtime = -1.0

    httpd = None
    if serve_port is not None:
        write_autoindex(out_dir)
        httpd = run_http_server(out_dir, serve_port)
        try:
            webbrowser.open_new_tab(f"http://localhost:{serve_port}/")
        except Exception:
            pass

    print(f"Watching {case_dir} every {interval}s. Press Ctrl+C to stop.")
    try:
        while True:
            # 1) check data changes
            data_sig = files_sig(targets)

            # 2) check config changes
            try:
                cfg_mtime = cfg_path.stat().st_mtime
            except FileNotFoundError:
                cfg_mtime = -1.0

            # 3) trigger if either changed
            if (data_sig != prev_data_sig) or (cfg_mtime != prev_cfg_mtime):
                # (re)load config only if it changed
                if cfg_mtime != prev_cfg_mtime and cfg_mtime >= 0:
                    with cfg_path.open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}

                prev_data_sig = data_sig
                prev_cfg_mtime = cfg_mtime

                n = plot_figures(case_dir, cfg, out_dir, show=show)
                if serve_port is not None:
                    write_autoindex(out_dir)
                print(
                    f"[watch] Updated {n} figure(s)."
                    if n
                    else "[watch] No figures produced (yet)."
                )

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if httpd is not None:
            httpd.shutdown()


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
    ap.add_argument(
        "--watch",
        type=float,
        metavar="SECS",
        help="Poll interval in seconds for live updates",
    )
    ap.add_argument(
        "--serve",
        type=int,
        metavar="PORT",
        help="Serve out-dir via simple HTTP server on PORT (auto-refresh index.html)",
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

    if args.watch or args.serve:
        live_watch(
            case_dir,
            cfg,
            out_dir,
            cfg_path=cfg_path,
            show=bool(args.show),
            interval=(args.watch or 2.0),
            serve_port=args.serve,
        )
        return 0
    else:
        n = plot_figures(case_dir, cfg, out_dir, show=args.show, live=False)
        return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
