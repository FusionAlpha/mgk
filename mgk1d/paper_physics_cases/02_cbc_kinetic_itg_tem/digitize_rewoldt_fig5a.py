"""Digitize the FULL TEM curves in Rewoldt et al. (2007), Fig. 5(a).

The source panel is a high-resolution monochrome image embedded in the paper.
The published curves are traced within a narrow neighborhood of the existing
MGK eigenfunction, which separates them from axes, labels, and panel (b).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PDF = PROJECT_ROOT / "Rewoldt2007_science.pdf"
MGK_MODE = PROJECT_ROOT / "analysis" / "mgk_rewoldt2007_fig5a_tem_mode.csv"
OUTPUT = PROJECT_ROOT / "analysis" / "rewoldt2007_fig5a_digitized.csv"

# Pixel calibration of panel (a) in the losslessly extracted Fig. 5 image.
X_LEFT, X_RIGHT = 223, 2132
Y_ZERO, Y_UNIT = 1785, 1766
THETA_LEFT, THETA_RIGHT = -8.0, 8.0


def load_mgk_mode() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with MGK_MODE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    theta = np.array([float(row["theta"]) for row in rows])
    real_phi = np.array([float(row["realPhi"]) for row in rows])
    imag_phi = np.array([float(row["imagPhi"]) for row in rows])
    return theta, real_phi, imag_phi


def extract_figure() -> np.ndarray:
    figure = PdfReader(SOURCE_PDF).pages[4].images[0].image.convert("L")
    assert figure.size == (4900, 2206), "Unexpected embedded Fig. 5 dimensions."
    return np.asarray(figure) < 128


def trace_curve(
    ink: np.ndarray,
    theta: np.ndarray,
    guide: np.ndarray,
    search_half_height: int,
) -> np.ndarray:
    traced = np.full(theta.shape, np.nan)
    for index, (theta_value, guide_value) in enumerate(zip(theta, guide)):
        x_float = X_LEFT + (theta_value - THETA_LEFT) / (
            THETA_RIGHT - THETA_LEFT
        ) * (X_RIGHT - X_LEFT)
        expected_y = Y_ZERO - Y_UNIT * guide_value
        x = int(round(x_float))
        y_min = max(0, int(np.floor(expected_y - search_half_height)))
        y_max = min(ink.shape[0], int(np.ceil(expected_y + search_half_height + 1)))
        window = ink[y_min:y_max, max(0, x - 2) : min(ink.shape[1], x + 3)]
        candidate_rows = np.where(window)[0] + y_min
        if candidate_rows.size:
            y = candidate_rows[np.argmin(np.abs(candidate_rows - expected_y))]
            traced[index] = (Y_ZERO - y) / Y_UNIT
    return traced


def main() -> None:
    ink = extract_figure()
    mgk_theta, mgk_real, mgk_imag = load_mgk_mode()
    theta = np.linspace(-7.0, 7.0, 57)
    real_guide = np.interp(theta, mgk_theta, mgk_real)
    imag_guide = np.interp(theta, mgk_theta, mgk_imag)
    full_real = trace_curve(ink, theta, real_guide, search_half_height=80)
    full_imag = trace_curve(ink, theta, imag_guide, search_half_height=55)

    if np.count_nonzero(np.isfinite(full_real)) < 0.80 * theta.size:
        raise RuntimeError("Insufficient coverage while tracing the real curve.")
    if np.count_nonzero(np.isfinite(full_imag)) < 0.80 * theta.size:
        raise RuntimeError("Insufficient coverage while tracing the imaginary curve.")

    real_finite = np.isfinite(full_real)
    imag_finite = np.isfinite(full_imag)
    full_real = np.interp(theta, theta[real_finite], full_real[real_finite])
    full_imag = np.interp(theta, theta[imag_finite], full_imag[imag_finite])

    with OUTPUT.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("theta", "fullRealPhi", "fullImagPhi"))
        writer.writerows(zip(theta, full_real, full_imag))
    print(OUTPUT)


if __name__ == "__main__":
    main()
