#!/usr/bin/env python3
"""Eksperyment 3 (Sekcja 5.4): Klasyfikacja stanu energetycznego (LOW, MID, HIGH).

Ewaluuje predykcje wieloczynnikowego klasyfikatora systemu LUMIR względem
anotacji referencyjnych ekspertów (Ground Truth, 2420 bloków decyzyjnych).
Generuje macierz pomyłek (Tabela 5.7) oraz metryki syntetyczne.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
ANN_DIR = BASE_DIR / "annotations"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import, TRACK_KEYS
from exp3_classifier.annotations.expert_annotations import get_ground_truth_label, export_block_annotations_csv

setup_lumir_import()


def main():
    print("=" * 65)
    print("  EKSPERYMENT 3: Ewaluacja klasyfikatora stanów energetycznych")
    print("=" * 65)

    # Export block annotations if not present
    gt_csv = ANN_DIR / "ground_truth_blocks.csv"
    export_block_annotations_csv(DATA_DIR, gt_csv)

    y_true: list[str] = []
    y_pred: list[str] = []

    for key in TRACK_KEYS:
        csv_file = DATA_DIR / f"telemetry_{key.lower()}.csv"
        if not csv_file.exists():
            print(f"[OSTRZEŻENIE] Brak pliku {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            t = float(row["timestamp"])
            pred = str(row["energy_state"]).strip()
            true_label = get_ground_truth_label(key, t)

            y_true.append(true_label)
            y_pred.append(pred)

    print(f"Liczba ocenionych bloków decyzyjnych: {len(y_true)}")

    labels = ["LOW", "MID", "HIGH"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Konstrukcja Tabeli 5.7 z pracy magisterskiej
    df_cm = pd.DataFrame(
        cm,
        index=[f"Rzeczywistość {lbl}" for lbl in labels],
        columns=[f"Predykcja {lbl}" for lbl in labels],
    )
    df_cm["Suma"] = df_cm.sum(axis=1)
    col_sums = df_cm.sum(axis=0)
    col_sums.name = "Suma"
    df_cm_full = pd.concat([df_cm, pd.DataFrame([col_sums])])

    print("\n--- Tabela 5.7: Macierz pomyłek dla klasyfikacji stanów energetycznych ---")
    print(df_cm_full.to_string())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n--- Metryki jakości klasyfikacji systemu LUMIR ---")
    print(f"Dokładność (Accuracy):           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Makro-F1 (Macro-F1):             {macro_f1:.4f}")
    print(f"Współczynnik kappa (κ) Cohena:   {kappa:.4f}")

    # Raport klasyfikacji per klasa
    cls_rep = classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True)
    df_cls = pd.DataFrame(cls_rep).transpose().reset_index()
    df_cls.rename(columns={"index": "Klasa / Średnia"}, inplace=True)

    export_table(df_cm_full.reset_index(), RESULTS_DIR / "table_5_7_confusion_matrix",
                 title="Tabela 5.7: Macierz pomyłek dla klasyfikacji stanów energetycznych względem anotacji referencyjnych")
    export_table(df_cls, RESULTS_DIR / "classification_report_per_class",
                 title="Raport klasyfikacji per klasa energetyczna")

    # Wizualizacja macierzy pomyłek (Heatmap)
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > np.max(cm) / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predykcja systemu LUMIR", labelpad=10)
    ax.set_ylabel("Anotacja referencyjna (Ground Truth)", labelpad=10)
    ax.set_title(f"Macierz pomyłek (N={len(y_true)}, Acc={acc * 100:.2f}%, κ={kappa:.4f})", pad=15)

    save_plot(fig, RESULTS_DIR / "plot_confusion_matrix.png")
    plt.close(fig)
    print("\nEksperyment 3 (Ewaluacja klasyfikatora): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
