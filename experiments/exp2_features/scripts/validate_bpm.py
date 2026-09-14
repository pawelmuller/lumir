#!/usr/bin/env python3
"""Eksperyment 2 (Sekcja 5.3): Walidacja estymacji tempa BPM w czasie rzeczywistym.

Porównuje estymowane wartości BPM z telemetrii LUMIR z referencyjną analizą librosa
oraz wyznacza czas stabilizacji estymacji (Tabela 5.6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import, TRACK_METADATA

setup_lumir_import()


def main():
    print("=" * 65)
    print("  EKSPERYMENT 2: Porównanie estymacji tempa BPM (Tabela 5.6)")
    print("=" * 65)

    ordered_keys = [
        "Less_Than_Zero",
        "Thunderstruck",
        "All_of_the_Lights",
        "Dead_Flowers",
        "Wodymidaj",
    ]

    results = []
    plot_trajectories = {}

    for key in ordered_keys:
        t_meta = TRACK_METADATA[key]
        csv_file = DATA_DIR / f"telemetry_{key.lower()}.csv"

        if not csv_file.exists():
            print(f"[OSTRZEŻENIE] Brak pliku {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        time_series = df["timestamp"].values
        bpm_series = df["bpm"].values

        final_bpm = float(bpm_series[-1])
        ref_bpm = float(t_meta["ref_bpm"])

        # Determine stabilization timestamp: the earliest point after which BPM stays at final_bpm
        stable_mask = (bpm_series == final_bpm)
        # Find last contiguous block of True
        first_stable_idx = np.where(stable_mask)[0][0]
        # Check if there were any reversions
        for idx in range(len(bpm_series) - 1, -1, -1):
            if bpm_series[idx] != final_bpm:
                first_stable_idx = idx + 1
                break

        stabilization_s = float(time_series[first_stable_idx])

        # Relative error
        if key == "Dead_Flowers":
            err_str = "nd.*"
        else:
            rel_err = abs(final_bpm - ref_bpm) / ref_bpm * 100
            err_str = f"{rel_err:.2f}%".replace(".", ",")

        results.append({
            "Utwór i gatunek": f"{t_meta['artist']} – {t_meta['title']} ({t_meta['genre']})",
            "Referencja [BPM]": f"{ref_bpm:.2f}".replace(".", ","),
            "System LUMIR [BPM]": f"{final_bpm:.2f}".replace(".", ","),
            "Błąd": err_str,
            "Stabilizacja [s]": f"{stabilization_s:.2f}".replace(".", ","),
        })

        plot_trajectories[key] = {
            "title": f"{t_meta['artist']} – {t_meta['title']}",
            "time": time_series,
            "bpm": bpm_series,
            "ref_bpm": ref_bpm,
            "final_bpm": final_bpm,
            "stabilization_s": stabilization_s,
        }

    df_table = pd.DataFrame(results)

    print("\n--- Tabela 5.6: Porównanie estymacji tempa BPM w czasie rzeczywistym ---")
    print(df_table.to_string(index=False))
    print("\n*W utworze Dead Flowers referencja librosa uległa błędowi podziału metrycznego (83,35 vs 112,35 BPM).")

    export_table(df_table, RESULTS_DIR / "table_5_6_bpm_estimation",
                 title="Tabela 5.6: Porównanie estymacji tempa BPM w czasie rzeczywistym względem analizy referencyjnej")

    # Wizualizacja trajektorii adaptacji tempa BPM
    setup_plot_style()
    fig, axes = plt.subplots(len(plot_trajectories), 1, figsize=(11, 2.2 * len(plot_trajectories)), sharex=True)
    if len(plot_trajectories) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i, (key, d) in enumerate(plot_trajectories.items()):
        ax = axes[i]
        ax.step(d["time"], d["bpm"], where="post", label="LUMIR (autokorelacja)", color=colors[i % len(colors)],
                linewidth=2)
        ax.axhline(d["ref_bpm"], color="#555555", linestyle="--", linewidth=1.2,
                   label=f"Referencja librosa ({d['ref_bpm']:.1f} BPM)")
        ax.axvline(d["stabilization_s"], color="red", linestyle=":", linewidth=1.2,
                   label=f"Stabilizacja ({d['stabilization_s']:.2f} s)")
        ax.set_ylabel("BPM")
        ax.set_title(f"{d['title']} – Final: {d['final_bpm']:.2f} BPM (Stabilizacja: {d['stabilization_s']:.2f} s)",
                     fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Czas [s]")
    save_plot(fig, RESULTS_DIR / "plot_bpm_stabilization.png")
    plt.close(fig)
    print("\nEksperyment 2 (BPM): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
