#!/usr/bin/env python3
"""Eksperyment 1 (Sekcja 5.2): Profilowanie opóźnień przetwarzania bloku audio w pętli LUMIR.

Porównuje trzy warianty pętli przetwarzania:
1. FilePlayer (offline) - przetwarzanie bez ograniczeń czasowych (dolna granica obliczeniowa)
2. FilePlayer (real-time) - sztuczne opóźnienie symulujące odczyt z interfejsu (sleep ~93 ms)
3. LiveStream (BlackHole) - rzeczywisty strumień audio przez wirtualne urządzenie loopback
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup import path for common module
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import

setup_lumir_import()


def calculate_mode_statistics(csv_files: list[Path], label: str) -> dict:
    """Calculate latency statistics for a given set of telemetry files."""
    all_times: list[float] = []
    for f in csv_files:
        df = pd.read_csv(f)
        if "processing_time_ms" in df.columns:
            times = df["processing_time_ms"].dropna().values
            all_times.extend(times)

    t = np.array(all_times, dtype=np.float64)
    if len(t) == 0:
        return {}

    return {
        "Wariant": label,
        "Liczba bloków (n)": len(t),
        "Średnia [ms]": float(np.mean(t)),
        "Mediana [ms]": float(np.median(t)),
        "Jitter (SD) [ms]": float(np.std(t, ddof=1)),
        "P90 [ms]": float(np.percentile(t, 90)),
        "P95 [ms]": float(np.percentile(t, 95)),
        "P99 [ms]": float(np.percentile(t, 99)),
        "Maksimum [ms]": float(np.max(t)),
        "_raw": t,
    }


def main():
    print("=" * 65)
    print("  EKSPERYMENT 1: Profilowanie czasu przetwarzania bloku audio (DSP)")
    print("=" * 65)

    offline_files = sorted(DATA_DIR.glob("telemetry_offline_*.csv"))
    realtime_files = sorted(DATA_DIR.glob("telemetry_realtime_*.csv"))
    livestream_files = sorted(DATA_DIR.glob("telemetry_livestream_*.csv"))

    modes = [
        ("FilePlayer (offline)", offline_files),
        ("FilePlayer (real-time)", realtime_files),
        ("LiveStream (BlackHole)", livestream_files),
    ]

    stats_list = []
    raw_dict = {}

    for label, files in modes:
        if not files:
            print(f"[OSTRZEŻENIE] Brak plików dla wariantu: {label}")
            continue
        res = calculate_mode_statistics(files, label)
        raw_dict[label] = res.pop("_raw")
        stats_list.append(res)

    if not stats_list:
        print("[BŁĄD] Nie znaleziono danych pomiarowych w", DATA_DIR)
        return

    df_results = pd.DataFrame(stats_list)

    # Tabela 5.1 z pracy magisterskiej (Średnia, P95, Maksimum)
    df_thesis = pd.DataFrame({
        "Metryka": ["Średnia [ms]", "P95 [ms]", "Maksimum [ms]"],
        "FilePlayer (offline)": [
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (offline)', 'Średnia [ms]'].values[0]:.2f}".replace(
                ".", ","),
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (offline)', 'P95 [ms]'].values[0]:.2f}".replace(".",
                                                                                                                   ","),
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (offline)', 'Maksimum [ms]'].values[0]:.2f}".replace(
                ".", ","),
        ],
        "FilePlayer (real-time)": [
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (real-time)', 'Średnia [ms]'].values[0]:.2f}".replace(
                ".", ","),
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (real-time)', 'P95 [ms]'].values[0]:.2f}".replace(
                ".", ","),
            f"{df_results.loc[df_results['Wariant'] == 'FilePlayer (real-time)', 'Maksimum [ms]'].values[0]:.2f}".replace(
                ".", ","),
        ],
        "LiveStream (BlackHole)": [
            f"{df_results.loc[df_results['Wariant'] == 'LiveStream (BlackHole)', 'Średnia [ms]'].values[0]:.2f}".replace(
                ".", ","),
            f"{df_results.loc[df_results['Wariant'] == 'LiveStream (BlackHole)', 'P95 [ms]'].values[0]:.2f}".replace(
                ".", ","),
            f"{df_results.loc[df_results['Wariant'] == 'LiveStream (BlackHole)', 'Maksimum [ms]'].values[0]:.2f}".replace(
                ".", ","),
        ],
    })

    print("\n--- Wyniki: Zestawienie czasu przetwarzania bloku audio (Tabela 5.1) ---")
    print(df_thesis.to_string(index=False))

    print("\n--- Rozszerzone statystyki inżynierskie ---")
    display_cols = [c for c in df_results.columns if c != "_raw"]
    print(df_results[display_cols].to_string(index=False))

    # Eksport tabel
    export_table(df_thesis, RESULTS_DIR / "table_5_1_pipeline_latency",
                 title="Tabela 5.1: Zestawienie pomiarów czasu przetwarzania bloku audio")
    export_table(df_results[display_cols], RESULTS_DIR / "table_5_1_extended_metrics",
                 title="Rozszerzone metryki czasu przetwarzania")

    # Generowanie wykresu rozkładów
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"FilePlayer (offline)": "#1f77b4", "FilePlayer (real-time)": "#ff7f0e",
              "LiveStream (BlackHole)": "#2ca02c"}

    # Wykres 1: Boxplot (skala logarytmiczna dla czytelności rozrzutu)
    box_data = [raw_dict[label] for label, _ in modes if label in raw_dict]
    box_labels = [label for label, _ in modes if label in raw_dict]
    bplot = ax1.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black",
                                   "markersize": 6})
    for patch, color in zip(bplot["boxes"], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.axhline(92.88, color="red", linestyle="--", linewidth=1.5, label="Budżet bloku audio (92,88 ms)")
    ax1.set_ylabel("Czas przetwarzania [ms]")
    ax1.set_title("Rozkład czasu przetwarzania bloku audio")
    ax1.legend(loc="upper right")

    # Wykres 2: Empiryczna dystrybuanta (ECDF)
    for label, raw_vals in raw_dict.items():
        sorted_v = np.sort(raw_vals)
        y = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax2.plot(sorted_v, y * 100, label=f"{label} (P95={np.percentile(raw_vals, 95):.2f} ms)", color=colors[label],
                 linewidth=2)

    ax2.axvline(92.88, color="red", linestyle="--", linewidth=1.5, label="Budżet bloku (92,88 ms)")
    ax2.set_xlim(0, 35)
    ax2.set_xlabel("Czas przetwarzania [ms]")
    ax2.set_ylabel("Dystrybuanta empiryczna [%]")
    ax2.set_title("Dystrybuanta (ECDF) czasu przetwarzania")
    ax2.legend(loc="lower right")

    save_plot(fig, RESULTS_DIR / "plot_pipeline_latency.png")
    plt.close(fig)
    print("\nEksperyment 1 (Pipeline): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
