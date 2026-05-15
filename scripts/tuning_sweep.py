#!/usr/bin/env python3
"""
Hyperparameter sweep driver for PGM tuning presets (``configs/tuning_*.yaml``).

Run from the repository root with the project environment activated::

    python scripts/tuning_sweep.py --phase hvg
    python scripts/tuning_sweep.py --phase all --fast

Results append to ``reports/results/tuning/sweep_results.csv``. The best row is
promoted to ``configs/tuning_best.yaml`` when running ``--phase finalize`` or
``--phase all`` (finalize runs last).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pgm.config.loader import deep_merge_dicts, load_project_config
from pgm.pipelines.full_pipeline import run_full_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configs_dir(root: Path) -> Path:
    return root / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    out = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out if isinstance(out, dict) else {}


def build_merged_dict(
    configs_dir: Path,
    preset: str | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    merged = _load_yaml(configs_dir / "default.yaml")
    if preset:
        merged = deep_merge_dicts(
            merged,
            _load_yaml(configs_dir / f"{preset}.yaml"),
        )
    merged = deep_merge_dicts(merged, dict(extra))
    merged.pop("smoke_mode", None)
    return merged


def score_metrics_series(metrics: pd.Series) -> float:
    """Exploratory composite: stability + STRING overlap + diversity + KG effect."""
    ge = float(metrics.get("global_edge_count", 0) or 0)
    su = float(metrics.get("soft_union_edge_count", 0) or 0)
    if ge + su < 5:
        return -1e9
    boot = float(metrics.get("bootstrap_mean_pos_freq", 0) or 0)
    if math.isnan(boot):
        boot = 0.0
    boot_soft = float(metrics.get("bootstrap_soft_mean_pos_freq", 0) or 0)
    if math.isnan(boot_soft):
        boot_soft = 0.0
    rs = float(metrics.get("string_recall_soft_union", 0) or 0)
    ps = float(metrics.get("string_precision_soft_union", 0) or 0)
    jm = float(metrics.get("jac_soft_pairwise_mean", 0) or 0)
    kg_diff = float(metrics.get("kg_vs_soft_mean_absdiff", 0) or 0)
    return (
        2.0 * boot
        + 1.5 * boot_soft
        + 3.0 * rs
        + 1.0 * ps
        + 1.0 * jm
        + 0.5 * kg_diff
    )


def _metrics_from_result_row(row: pd.Series) -> pd.Series:
    d: dict[str, Any] = {}
    for k in row.index:
        ks = str(k)
        if ks.startswith("m__"):
            d[ks[3:]] = row[k]
    return pd.Series(d)


def run_one(
    *,
    root: Path,
    smoke: bool,
    preset: str | None,
    extra_overrides: dict[str, Any],
    force: bool,
    run_tag: str,
    phase: str,
) -> dict[str, Any]:
    cfg_dir = _configs_dir(root)
    cfg = load_project_config(
        configs_dir=cfg_dir,
        smoke=smoke,
        preset=preset,
        extra_overrides=extra_overrides,
    )
    summary = run_full_pipeline(cfg, force=force)
    metrics_path = cfg.resolve(cfg.reports.results_dir) / cfg.evaluation.metrics_csv
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        mdf = pd.read_csv(metrics_path)
        metrics = mdf.iloc[0].to_dict()
    ser = pd.Series(metrics)
    score = score_metrics_series(ser)
    out_row: dict[str, Any] = {
        "run_tag": run_tag,
        "phase": phase,
        "preset": preset or "",
        "overrides_json": json.dumps(extra_overrides, sort_keys=True),
        "metrics_csv": str(cfg.evaluation.metrics_csv),
        "clustered_h5ad": str(summary.get("clustered_h5ad", "")),
        "score": score,
    }
    for k, v in metrics.items():
        out_row[f"m__{k}"] = v
    return out_row


def append_results(root: Path, row: dict[str, Any]) -> None:
    out_dir = root / "reports" / "results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sweep_results.csv"
    df = pd.DataFrame([row])
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)


def finalize_best(root: Path) -> None:
    out_csv = root / "reports" / "results" / "tuning" / "sweep_results.csv"
    if not out_csv.is_file():
        print("No sweep_results.csv; skip finalize", file=sys.stderr)
        return
    df = pd.read_csv(out_csv)
    if df.empty:
        return
    if "score" not in df.columns:
        df["score"] = df.apply(
            lambda r: score_metrics_series(_metrics_from_result_row(r)),
            axis=1,
        )
    best_idx = int(df["score"].astype(float).idxmax())
    best = df.loc[best_idx]
    preset = str(best.get("preset", "") or "").strip() or None
    overrides = json.loads(str(best.get("overrides_json", "{}")))
    cfg_dir = _configs_dir(root)
    merged = build_merged_dict(cfg_dir, preset, overrides)
    merged.setdefault("checkpoint", {})
    merged["checkpoint"]["subdirectory"] = "tuning/best"
    merged.setdefault("preprocessing", {})
    merged["preprocessing"]["output_filename"] = "pbmc3k_tuning_best.h5ad"
    merged.setdefault("evaluation", {})
    merged["evaluation"]["metrics_csv"] = "tuning/runs/best_metrics.csv"
    best_path = cfg_dir / "tuning_best.yaml"
    with best_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {best_path} from sweep row run_tag={best.get('run_tag')}")


def main() -> int:
    root = _repo_root()
    os.chdir(root)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="PGM tuning sweep")
    parser.add_argument(
        "--phase",
        choices=["hvg", "alpha", "gmm", "kg", "finalize", "all"],
        default="hvg",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer bootstrap reps and smaller grids.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Merge smoke.yaml (tiny runs). Note: smoke overrides preprocessing "
            "(HVG/cells) and is not suitable for HVG preset comparison."
        ),
    )
    parser.add_argument(
        "--no-pipeline-force",
        action="store_true",
        help=(
            "Pass force=False into run_full_pipeline (reuse checkpoints). "
            "Requires compatible artifacts under each preset checkpoint subdirectory."
        ),
    )
    args = parser.parse_args()

    pipe_force = not args.no_pipeline_force

    fast_eval = {"evaluation": {"bootstrap_b": 5, "bootstrap_fraction": 0.75}}

    if args.phase == "all":
        phases = ["hvg", "alpha", "gmm", "kg", "finalize"]
    elif args.phase == "finalize":
        phases = ["finalize"]
    else:
        phases = [args.phase]

    for phase in phases:
        if phase == "finalize":
            finalize_best(root)
            continue

        if phase == "hvg":
            for pr in ["tuning_hvg200", "tuning_hvg300", "tuning_hvg500"]:
                extra = dict(fast_eval) if args.fast else {}
                append_results(
                    root,
                    run_one(
                        root=root,
                        smoke=args.smoke,
                        preset=pr,
                        extra_overrides=extra,
                        force=pipe_force,
                        run_tag=f"hvg::{pr}",
                        phase=phase,
                    ),
                )

        elif phase == "alpha":
            alphas = [0.08, 0.15] if args.fast else [0.03, 0.05, 0.08, 0.1, 0.15, 0.25]
            scales = [0.01, 0.05] if args.fast else [0.0, 0.01, 0.02, 0.05]
            shrinks = [0.1] if args.fast else [0.05, 0.1, 0.15]
            i = 0
            for a in alphas:
                for s in scales:
                    for sh in shrinks:
                        i += 1
                        sub = f"tuning/sweep_alpha/a{a}_s{s}_sh{sh}".replace(".", "p")
                        extra: dict[str, Any] = {
                            "checkpoint": {"subdirectory": sub},
                            "preprocessing": {
                                "output_filename": f"pbmc3k_tuning_alpha_{i}.h5ad",
                            },
                            "evaluation": {
                                "metrics_csv": f"tuning/runs/alpha_{i}_metrics.csv",
                            },
                            "models": {
                                "gl_alpha": float(a),
                                "gl_soft_ess_min_alpha_scale": float(s),
                                "covariance_shrinkage": float(sh),
                            },
                        }
                        if args.fast:
                            extra = deep_merge_dicts(extra, fast_eval)
                        append_results(
                            root,
                            run_one(
                                root=root,
                                smoke=args.smoke,
                                preset="tuning_hvg300",
                                extra_overrides=extra,
                                force=pipe_force,
                                run_tag=f"alpha::{i}",
                                phase=phase,
                            ),
                        )

        elif phase == "gmm":
            comps = [3, 4] if args.fast else [3, 4, 5, 6]
            covs = ["diag", "tied"] if args.fast else ["diag", "tied"]
            pcs_list = [35] if args.fast else [20, 30, 35, 40]
            i = 0
            for k in comps:
                for cov in covs:
                    for npcs in pcs_list:
                        i += 1
                        sub = f"tuning/sweep_gmm/k{k}_{cov}_pcs{npcs}"
                        extra = {
                            "checkpoint": {"subdirectory": sub},
                            "preprocessing": {
                                "output_filename": f"pbmc3k_tuning_gmm_{i}.h5ad",
                            },
                            "evaluation": {
                                "metrics_csv": f"tuning/runs/gmm_{i}_metrics.csv",
                            },
                            "clustering": {
                                "gmm_n_components": int(k),
                                "gmm_covariance_type": cov,
                                "neighbor_n_pcs": int(npcs),
                            },
                        }
                        if args.fast:
                            extra = deep_merge_dicts(extra, fast_eval)
                        append_results(
                            root,
                            run_one(
                                root=root,
                                smoke=args.smoke,
                                preset="tuning_hvg300",
                                extra_overrides=extra,
                                force=pipe_force,
                                run_tag=f"gmm::{i}",
                                phase=phase,
                            ),
                        )

        elif phase == "kg":
            thr = [0.5, 0.7] if args.fast else [0.4, 0.5, 0.7]
            scales = [0.05, 0.1] if args.fast else [0.02, 0.05, 0.1, 0.2, 0.5]
            i = 0
            for t in thr:
                for lam in scales:
                    i += 1
                    sub = f"tuning/sweep_kg/thr{t}_lam{lam}".replace(".", "p")
                    extra = {
                        "checkpoint": {"subdirectory": sub},
                        "preprocessing": {
                            "output_filename": f"pbmc3k_tuning_kg_{i}.h5ad",
                        },
                        "evaluation": {
                            "metrics_csv": f"tuning/runs/kg_{i}_metrics.csv",
                        },
                        "kg": {
                            "confidence_threshold": float(t),
                            "prior_covariance_scale": float(lam),
                        },
                    }
                    if args.fast:
                        extra = deep_merge_dicts(extra, fast_eval)
                    append_results(
                        root,
                        run_one(
                            root=root,
                            smoke=args.smoke,
                            preset="tuning_recommended",
                            extra_overrides=extra,
                            force=pipe_force,
                            run_tag=f"kg::{i}",
                            phase=phase,
                        ),
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
