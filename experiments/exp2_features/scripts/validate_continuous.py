#!/usr/bin/env python3
"""Eksperyment 2 (Sekcja 5.3): Walidacja ekstrakcji cech ciągłych (RMS i centroid widmowy).

Porównuje cechy wyekstrahowane przyczynowo w czasie rzeczywistym przez klasę Analyzer
z referencyjnymi przebiegami obliczonymi offline przez bibliotekę librosa (Tabela 5.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REF_DIR = BASE_DIR / "reference"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import, TRACK_METADATA, TRACK_KEYS

setup_lumir_import()


def main():
    print("=" * 65)
    print("  EKSPERYMENT 2: Walidacja cech ciągłych (RMS, Centroid Widmowy)")
    print("=" * 65)

    per_track_results = []
    plot_data = {}

    for track_key in TRACK_KEYS:
        t_meta = TRACK_METADATA[track_key]
        csv_file = DATA_DIR / f"telemetry_{track_key.lower()}.csv"
        ref_file = REF_DIR / f"ref_{track_key.lower()}.npz"

        if not csv_file.exists() or not ref_file.exists():
            print(f"[OSTRZEŻENIE] Brak danych dla {track_key}")
            continue

        df = pd.read_csv(csv_file)
        ref = np.load(ref_file)

        raw_rms = df["rms"].values
        ref_rms = ref["rms"]

        raw_cent = df["spectral_centroid_hz"].values
        ref_cent = ref["centroid"]

        # Resample LUMIR frames to match Librosa frame rate (hop_length=512)
        old_time = np.linspace(0, 1, len(raw_rms))
        new_time = np.linspace(0, 1, len(ref_rms))
        lumir_rms = np.interp(new_time, old_time, raw_rms)
        lumir_cent = np.interp(new_time, old_time, raw_cent)

        # Cross-correlation alignment based on normalized RMS envelope
        ref_norm = ref_rms - np.mean(ref_rms)
        lum_norm = lumir_rms - np.mean(lumir_rms)
        correlation = signal.correlate(ref_norm, lum_norm, mode="full")
        lags = signal.correlation_lags(len(ref_norm), len(lum_norm), mode="full")
        lag = lags[np.argmax(correlation)]

        # Align both RMS and Centroid using the computed causal lag
        if lag < 0:
            shift = -lag
            aligned_lum_rms = lumir_rms[shift:]
            aligned_ref_rms = ref_rms[:-shift] if shift < len(ref_rms) else np.array([])
            aligned_lum_cent = lumir_cent[shift:]
            aligned_ref_cent = ref_cent[:-shift] if shift < len(ref_cent) else np.array([])
        elif lag > 0:
            shift = lag
            aligned_lum_rms = lumir_rms[:-shift] if shift < len(lumir_rms) else np.array([])
            aligned_ref_rms = ref_rms[shift:]
            aligned_lum_cent = lumir_cent[:-shift] if shift < len(lumir_cent) else np.array([])
            aligned_ref_cent = ref_cent[shift:]
        else:
            aligned_lum_rms = lumir_rms
            aligned_ref_rms = ref_rms
            aligned_lum_cent = lumir_cent
            aligned_ref_cent = ref_cent

        min_len = min(len(aligned_ref_rms), len(aligned_lum_rms))
        aligned_ref_rms = aligned_ref_rms[:min_len]
        aligned_lum_rms = aligned_lum_rms[:min_len]
        aligned_ref_cent = aligned_ref_cent[:min_len]
        aligned_lum_cent = aligned_lum_cent[:min_len]

        corr_rms, _ = pearsonr(aligned_ref_rms, aligned_lum_rms)
        corr_cent, _ = pearsonr(aligned_ref_cent, aligned_lum_cent)

        per_track_results.append({
            "Utwór": f"{t_meta['artist']} - {t_meta['title']}",
            "Gatunek": t_meta["genre"],
            "Korelacja RMS": corr_rms,
            "Korelacja Centroid": corr_cent,
            "Przesunięcie (lag)": int(lag),
        })

        plot_data[track_key] = {
            "title": f"{t_meta['artist']} – {t_meta['title']}",
            "ref_rms": aligned_ref_rms,
            "lum_rms": aligned_lum_rms,
            "ref_cent": aligned_ref_cent,
            "lum_cent": aligned_lum_cent,
        }

    df_per_track = pd.DataFrame(per_track_results)

    avg_rms = float(df_per_track["Korelacja RMS"].mean())
    avg_cent = float(df_per_track["Korelacja Centroid"].mean())

    # Tabela 5.3 z pracy magisterskiej
    df_thesis = pd.DataFrame({
        "Cecha": ["RMS", "Centroid widmowy"],
        "Korelacja Pearsona": [
            f"{avg_rms:.4f}".replace(".", ","),
            f"{avg_cent:.4f}".replace(".", ","),
        ],
    })

    print("\n--- Tabela 5.3: Zgodność ciągłych cech audio względem referencji librosa ---")
    print(df_thesis.to_string(index=False))

    print("\n--- Szczegółowe wyniki per utwór ---")
    df_track_display = df_per_track.copy()
    df_track_display["Korelacja RMS"] = df_track_display["Korelacja RMS"].map(lambda x: f"{x:.4f}".replace(".", ","))
    df_track_display["Korelacja Centroid"] = df_track_display["Korelacja Centroid"].map(
        lambda x: f"{x:.4f}".replace(".", ","))
    print(df_track_display.to_string(index=False))

    export_table(df_thesis, RESULTS_DIR / "table_5_3_continuous_correlation",
                 title="Tabela 5.3: Ocena zgodności ciągłych cech audio wyekstrahowanych w czasie rzeczywistym względem danych referencyjnych")
    export_table(df_track_display, RESULTS_DIR / "table_5_3_per_track_correlation",
                 title="Korelacja cech ciągłych w rozbiciu na poszczególne utwory")

    # Wizualizacja przebiegów czasowych
    setup_plot_style()
    n_tracks = len(plot_data)
    fig, axes = plt.subplots(n_tracks, 2, figsize=(14, 2.5 * n_tracks), sharex=False)
    if n_tracks == 1:
        axes = np.array([axes])

    for i, (track_key, d) in enumerate(plot_data.items()):
        n_samples = len(d["ref_rms"])
        time_axis = np.linspace(0, 45, n_samples)

        # RMS
        max_rms = max(np.max(d["ref_rms"]), 1e-6)
        max_lum = max(np.max(d["lum_rms"]), 1e-6)
        axes[i, 0].plot(time_axis, d["ref_rms"] / max_rms, label="Referencja (librosa)", color="#2b5c8f", alpha=0.8,
                        linewidth=1.2)
        axes[i, 0].plot(time_axis, d["lum_rms"] / max_lum, label="LUMIR (czas rzeczywisty)", color="#d95f02", alpha=0.8,
                        linewidth=1.2)
        axes[i, 0].set_ylabel("Znorm. RMS")
        r_val = df_per_track.loc[
            df_per_track['Utwór'].str.contains(TRACK_METADATA[track_key]['title']), 'Korelacja RMS'].values[0]
        axes[i, 0].set_title(f"{d['title']} – Obwiednia RMS (r = {r_val:.4f})", fontsize=10)
        axes[i, 0].grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=8)

        # Centroid
        axes[i, 1].plot(time_axis, d["ref_cent"], label="Referencja (librosa)", color="#2b5c8f", alpha=0.8,
                        linewidth=1.2)
        axes[i, 1].plot(time_axis, d["lum_cent"], label="LUMIR (czas rzeczywisty)", color="#2ca02c", alpha=0.8,
                        linewidth=1.2)
        axes[i, 1].set_ylabel("Centroid [Hz]")
        c_val = df_per_track.loc[
            df_per_track['Utwór'].str.contains(TRACK_METADATA[track_key]['title']), 'Korelacja Centroid'].values[0]
        axes[i, 1].set_title(f"{d['title']} – Centroid widmowy (r = {c_val:.4f})", fontsize=10)
        axes[i, 1].grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            axes[i, 1].legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Czas [s]")
    axes[-1, 1].set_xlabel("Czas [s]")
    save_plot(fig, RESULTS_DIR / "plot_continuous_features.png")
    plt.close(fig)
    print("\nEksperyment 2 (Cechy ciągłe): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
