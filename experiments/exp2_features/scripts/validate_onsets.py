#!/usr/bin/env python3
"""Eksperyment 2 (Sekcja 5.3): Walidacja detekcji onsetów w standardzie MIR.

Porównuje dwa etapy detekcji onsetów:
1. Skuteczność analityczna (Moduł MIR) -- detekcja surowa przed filtracją (Tabela 5.4)
2. Skuteczność wykonawcza (Moduł Decyzyjny) -- detekcja po bramkowaniu i cooldownie (Tabela 5.5)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REF_DIR = BASE_DIR / "reference"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import, TRACK_METADATA

setup_lumir_import()


def compute_lag(ref_rms: np.ndarray, lumir_rms: np.ndarray) -> int:
    """Compute lag between signals via cross-correlation."""
    ref_norm = ref_rms - np.mean(ref_rms)
    lum_norm = lumir_rms - np.mean(lumir_rms)
    corr = signal.correlate(ref_norm, lum_norm, mode="full")
    lags = signal.correlation_lags(len(ref_norm), len(lum_norm), mode="full")
    return int(lags[np.argmax(corr)])


def main():
    print("=" * 65)
    print("  EKSPERYMENT 2: Walidacja detekcji onsetów (Tabele 5.4 i 5.5)")
    print("=" * 65)

    results_mir = []
    results_decision = []

    # Sort in thesis presentation order:
    # 1. Less Than Zero, 2. Thunderstruck, 3. All of the Lights, 4. Dead Flowers, 5. Wodymidaj
    ordered_keys = [
        "Less_Than_Zero",
        "Thunderstruck",
        "All_of_the_Lights",
        "Dead_Flowers",
        "Wodymidaj",
    ]

    for key in ordered_keys:
        t_meta = TRACK_METADATA[key]
        csv_file = DATA_DIR / f"telemetry_{key.lower()}.csv"
        ref_file = REF_DIR / f"ref_{key.lower()}.npz"

        if not csv_file.exists() or not ref_file.exists():
            print(f"[OSTRZEŻENIE] Brak danych dla {key}")
            continue

        df = pd.read_csv(csv_file)
        ref = np.load(ref_file)

        raw_rms = df["rms"].values
        ref_rms = ref["rms"]

        # Resample for cross-correlation lag
        old_time = np.linspace(0, 1, len(raw_rms))
        new_time = np.linspace(0, 1, len(ref_rms))
        lumir_rms = np.interp(new_time, old_time, raw_rms)
        lag = compute_lag(ref_rms, lumir_rms)

        # Time shift in seconds (LUMIR sample rate and hop length)
        time_shift = (-lag * 512) / 22050.0

        ref_onsets = ref["onsets"]

        # 1. Moduł MIR (detekcja surowa)
        lum_raw_beats = df.loc[df["is_raw_beat"] == True, "timestamp"].values
        lum_raw_shifted = lum_raw_beats - time_shift
        lum_raw_shifted = lum_raw_shifted[lum_raw_shifted >= 0]
        f_raw, p_raw, r_raw = mir_eval.onset.f_measure(ref_onsets, lum_raw_shifted, window=0.1)

        results_mir.append({
            "Utwór i gatunek": f"{t_meta['artist']} – {t_meta['title']} ({t_meta['genre']})",
            "Precyzja": p_raw,
            "Czułość": r_raw,
            "Miara F1": f_raw,
        })

        # 2. Moduł Decyzyjny (detekcja po filtracji i bramkowaniu)
        lum_beats = df.loc[df["is_beat"] == True, "timestamp"].values
        lum_beats_shifted = lum_beats - time_shift
        lum_beats_shifted = lum_beats_shifted[lum_beats_shifted >= 0]
        f_dec, p_dec, r_dec = mir_eval.onset.f_measure(ref_onsets, lum_beats_shifted, window=0.1)

        results_decision.append({
            "Utwór i gatunek": f"{t_meta['artist']} – {t_meta['title']} ({t_meta['genre']})",
            "Precyzja": p_dec,
            "Czułość": r_dec,
            "Miara F1": f_dec,
        })

    df_mir = pd.DataFrame(results_mir)
    df_decision = pd.DataFrame(results_decision)

    # Formatowanie zbieżne z Tabelami 5.4 i 5.5
    def format_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        df_out["Precyzja"] = df_out["Precyzja"].map(lambda x: f"{x:.4f}".replace(".", ","))
        df_out["Czułość"] = df_out["Czułość"].map(lambda x: f"{x:.4f}".replace(".", ","))
        df_out["Miara F1"] = df_out["Miara F1"].map(lambda x: f"{x:.4f}".replace(".", ","))
        return df_out

    df_mir_formatted = format_metrics(df_mir)
    df_decision_formatted = format_metrics(df_decision)

    print("\n--- Tabela 5.4: Skuteczność analityczna detekcji onsetów (Moduł MIR, detekcja surowa) ---")
    print(df_mir_formatted.to_string(index=False))

    print("\n--- Tabela 5.5: Skuteczność wykonawcza detekcji onsetów (Moduł Decyzyjny, po filtracji) ---")
    print(df_decision_formatted.to_string(index=False))

    export_table(df_mir_formatted, RESULTS_DIR / "table_5_4_onsets_mir_raw",
                 title="Tabela 5.4: Skuteczność analityczna detekcji onsetów (Moduł MIR) -- detekcja surowa przed filtracją")
    export_table(df_decision_formatted, RESULTS_DIR / "table_5_5_onsets_decision_filtered",
                 title="Tabela 5.5: Skuteczność wykonawcza detekcji onsetów (Moduł Decyzyjny) -- po zastosowaniu bramek i czasu wygaszenia")

    # Wykres porównawczy Precyzji, Czułości i F1 dla obu trybów
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    track_short_names = [f"{t.split('–')[1].split('(')[0].strip()}" for t in df_mir["Utwór i gatunek"]]
    x = np.arange(len(track_short_names))
    width = 0.26

    # Wykres 1: Moduł MIR (surowy)
    ax1.bar(x - width, df_mir["Precyzja"], width, label="Precyzja", color="#2b5c8f")
    ax1.bar(x, df_mir["Czułość"], width, label="Czułość", color="#41b6c4")
    ax1.bar(x + width, df_mir["Miara F1"], width, label="Miara F1", color="#feb24c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(track_short_names, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Wartość metryki")
    ax1.set_title("Moduł MIR – Detekcja surowa (Tabela 5.4)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Wykres 2: Moduł Decyzyjny (wykonawczy)
    ax2.bar(x - width, df_decision["Precyzja"], width, label="Precyzja", color="#2b5c8f")
    ax2.bar(x, df_decision["Czułość"], width, label="Czułość", color="#41b6c4")
    ax2.bar(x + width, df_decision["Miara F1"], width, label="Miara F1", color="#feb24c")
    ax2.axhline(0.90, color="red", linestyle="--", linewidth=1.2, label="Założony próg precyzji (0,90)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(track_short_names, rotation=20, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Wartość metryki")
    ax2.set_title("Moduł Decyzyjny – Detekcja wykonawcza (Tabela 5.5)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    save_plot(fig, RESULTS_DIR / "plot_onsets_evaluation.png")
    plt.close(fig)
    print("\nEksperyment 2 (Onsety): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
