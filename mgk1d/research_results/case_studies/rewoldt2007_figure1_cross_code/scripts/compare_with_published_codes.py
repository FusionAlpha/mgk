#!/usr/bin/env python3
"""Compare gk-eig and CGYRO with the digitized Rewoldt Figure 1 curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
GKEIG_ROOT = HERE.parents[4]


def workspace_root() -> Path:
    for parent in HERE.parents:
        if (parent / "物理" / "科研" / "gacode" / "cgyro").is_dir():
            return parent
    raise FileNotFoundError("could not locate 物理/科研/gacode/cgyro")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows])


def make_comparison(
    gkeig_path: Path, cgyro_path: Path
) -> tuple[list[dict[str, float | str | bool]], Path]:
    paper = read_rows(CASE_ROOT / "data" / "paper_digitized_points.csv")
    gkeig = read_rows(gkeig_path)
    cgyro = read_rows(cgyro_path)
    full = [row for row in paper if row["source_code"] == "FULL"]
    full_by_ky = {float(row["ky_rhoi"]): row for row in full}
    paper_by_ky: dict[float, list[dict[str, str]]] = {}
    for row in paper:
        paper_by_ky.setdefault(float(row["ky_rhoi"]), []).append(row)

    comparison: list[dict[str, float | str | bool]] = []
    for code, rows in (("gk-eig", gkeig), ("CGYRO", cgyro)):
        for row in rows:
            ky = float(row["ky_rhoi"])
            target = full_by_ky[ky]
            omega = float(row["omega_cs_over_Ln"])
            gamma = float(row["gamma_cs_over_Ln"])
            target_omega = float(target["omega_cs_over_Ln"])
            target_gamma = float(target["gamma_cs_over_Ln"])
            published = paper_by_ky[ky]
            paper_omega = [float(item["omega_cs_over_Ln"]) for item in published]
            paper_gamma = [float(item["gamma_cs_over_Ln"]) for item in published]
            omega_uncertainty = float(target["digitization_uncertainty_omega"])
            gamma_uncertainty = float(target["digitization_uncertainty_gamma"])
            comparison.append(
                {
                    "code": code,
                    "ky_rhoi": ky,
                    "omega_cs_over_Ln": omega,
                    "gamma_cs_over_Ln": gamma,
                    "paper_FULL_omega_cs_over_Ln": target_omega,
                    "paper_FULL_gamma_cs_over_Ln": target_gamma,
                    "omega_relative_error": abs(omega - target_omega) / abs(target_omega),
                    "gamma_relative_error": abs(gamma - target_gamma) / abs(target_gamma),
                    "gamma_within_digitization_uncertainty": abs(gamma - target_gamma)
                    <= gamma_uncertainty,
                    "omega_within_digitization_uncertainty": abs(omega - target_omega)
                    <= omega_uncertainty,
                    "gamma_within_published_spread": min(paper_gamma) - gamma_uncertainty
                    <= gamma
                    <= max(paper_gamma) + gamma_uncertainty,
                    "omega_within_published_spread": min(paper_omega) - omega_uncertainty
                    <= omega
                    <= max(paper_omega) + omega_uncertainty,
                }
            )

    output_csv = CASE_ROOT / "data" / "figure1_cross_code_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True)
    paper_styles = {
        "FULL": ("#248a3d", "s"),
        "GTC": ("#2468b4", "o"),
        "GT3D": ("#cc4433", "^"),
    }
    for source, (color, marker) in paper_styles.items():
        rows = [row for row in paper if row["source_code"] == source]
        ky = numeric(rows, "ky_rhoi")
        axes[0].plot(
            ky,
            numeric(rows, "gamma_cs_over_Ln"),
            marker=marker,
            color=color,
            linewidth=1.4,
            markersize=4,
            label=f"paper {source}",
        )
        axes[1].plot(
            ky,
            numeric(rows, "omega_cs_over_Ln"),
            marker=marker,
            color=color,
            linewidth=1.4,
            markersize=4,
            label=f"paper {source}",
        )
    for label, rows, color, marker in (
        ("gk-eig", gkeig, "#111111", "D"),
        ("CGYRO", cgyro, "#8b3fbf", "x"),
    ):
        ky = numeric(rows, "ky_rhoi")
        axes[0].plot(
            ky,
            numeric(rows, "gamma_cs_over_Ln"),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            ky,
            numeric(rows, "omega_cs_over_Ln"),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
    axes[0].set_ylabel(r"$\gamma/(c_s/L_n)$")
    axes[1].set_ylabel(r"$\omega_r/(c_s/L_n)$")
    axes[1].set_xlabel(r"$k_\theta\rho_i$")
    axes[0].set_title("Rewoldt et al. (2007), Fig. 1: adiabatic-electron ITG")
    axes[0].legend(ncol=2, fontsize=8)
    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output_figure = CASE_ROOT / "figures" / "rewoldt2007_fig1_mgk_vs_published_codes.png"
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=200)
    plt.close(fig)
    return comparison, output_figure


def main() -> None:
    default_cgyro = (
        workspace_root()
        / "物理"
        / "科研"
        / "gacode"
        / "cgyro"
        / "runs"
        / "rewoldt2007_fig1_adiabatic"
        / "results"
        / "scan_results.csv"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--gkeig", type=Path, default=CASE_ROOT / "data" / "mgk_results.csv")
    parser.add_argument("--cgyro", type=Path, default=default_cgyro)
    args = parser.parse_args()
    rows, figure = make_comparison(args.gkeig, args.cgyro)
    for code in ("gk-eig", "CGYRO"):
        selected = [row for row in rows if row["code"] == code]
        max_gamma = max(float(row["gamma_relative_error"]) for row in selected)
        max_omega = max(float(row["omega_relative_error"]) for row in selected)
        core = [row for row in selected if float(row["ky_rhoi"]) <= 0.5]
        max_core_gamma = max(float(row["gamma_relative_error"]) for row in core)
        max_core_omega = max(float(row["omega_relative_error"]) for row in core)
        spread_gamma = sum(bool(row["gamma_within_published_spread"]) for row in selected)
        spread_omega = sum(bool(row["omega_within_published_spread"]) for row in selected)
        print(
            f"{code}: max FULL-relative error all points gamma={max_gamma:.3%}, "
            f"omega={max_omega:.3%}; ky<=0.5 gamma={max_core_gamma:.3%}, "
            f"omega={max_core_omega:.3%}; published spread gamma={spread_gamma}/6, "
            f"omega={spread_omega}/6"
        )
    print(f"CSV:    {CASE_ROOT / 'data' / 'figure1_cross_code_comparison.csv'}")
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
