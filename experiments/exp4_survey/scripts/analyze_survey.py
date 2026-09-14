#!/usr/bin/env python3
"""Eksperyment 4 (Sekcje 5.5–5.6): Subiektywna ocena scenografii świetlnej (badanie ankietowe).

Analizuje oceny 64 respondentów w 5-stopniowej skali Likerta dla 6 próbek wideo:
1. Zestawienie 5 kryteriów oceny dla systemu LUMIR (Tabela 5.9)
2. Porównanie z wariantem losowym/kontrolnym (test t Welcha)
3. Podział respondentów według grup doświadczenia (Tabela 5.10)
4. Jednoczynnikowa analiza wariancji (ANOVA) oraz test post-hoc Tukeya
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import

setup_lumir_import()


def clean_likert_value(val) -> float:
    """Extract integer Likert value (1-5) from text answers."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s and s[0].isdigit():
        return float(s[0])
    return np.nan


def main():
    print("=" * 65)
    print("  EKSPERYMENT 4: Analiza statystyczna badania ankietowego")
    print("=" * 65)

    data_file = DATA_DIR / "survey_cleaned.csv"
    if not data_file.exists():
        data_file = DATA_DIR / "survey_raw.csv"

    if not data_file.exists():
        print(f"[BŁĄD] Nie znaleziono danych ankietowych w {DATA_DIR}")
        return

    # Semicolon or comma separator auto-detection
    try:
        df = pd.read_csv(data_file)
        if len(df.columns) <= 2:
            df = pd.read_csv(data_file, sep=";")
    except Exception as e:
        df = pd.read_csv(data_file, sep=";")

    print(f"Załadowano odpowiedzi od {len(df)} respondentów.")

    # Identify Likert question columns
    criteria_names = [
        "Synchronizacja rytmiczna",
        "Oddanie dynamiki",
        "Reakcja na strukturę utworu",
        "Adekwatność kolorów i efektów",
        "Ogólna estetyka wizualna",
    ]

    likert_cols = [
        c for c in df.columns
        if any(k in c for k in ["Synchronizacja", "Oddanie", "Reakcja", "Adekwatność", "Ogólna"])
    ]

    for c in likert_cols:
        df[c] = df[c].apply(clean_likert_value)

    # 6 video samples * 5 questions = 30 columns
    # Samples 1-5: LUMIR (columns 0..24)
    # Sample 6: Kontrolny (columns 25..29)
    lumir_cols = likert_cols[:25]
    control_cols = likert_cols[25:30]

    # --- 1. Tabela 5.9: Zestawienie ocen ankietowych dla systemu LUMIR ---
    crit_stats = []
    for i, crit in enumerate(criteria_names):
        cols_for_crit = [likert_cols[s * 5 + i] for s in range(5)]
        vals = df[cols_for_crit].values.flatten()
        vals_clean = vals[~np.isnan(vals)]
        crit_stats.append({
            "Kryterium": crit,
            "Średnia": float(np.mean(vals_clean)),
            "Odch. std.": float(np.std(vals_clean, ddof=1)),
        })

    lum_flat = df[lumir_cols].values.flatten()
    lum_clean = lum_flat[~np.isnan(lum_flat)]
    mean_overall_lumir = float(np.mean(lum_clean))
    std_overall_lumir = float(np.std(lum_clean, ddof=1))

    df_table_5_9 = pd.DataFrame(crit_stats)
    df_table_5_9_full = pd.concat([
        df_table_5_9,
        pd.DataFrame([{"Kryterium": "Ogółem (LUMIR)", "Średnia": mean_overall_lumir, "Odch. std.": std_overall_lumir}])
    ])

    df_5_9_display = df_table_5_9_full.copy()
    df_5_9_display["Średnia"] = df_5_9_display["Średnia"].map(lambda x: f"{x:.2f}".replace(".", ","))
    df_5_9_display["Odch. std."] = df_5_9_display["Odch. std."].map(lambda x: f"{x:.2f}".replace(".", ","))

    print("\n--- Tabela 5.9: Zestawienie średnich ocen ankietowych (system LUMIR, nagrania 1–5) ---")
    print(df_5_9_display.to_string(index=False))

    export_table(df_5_9_display, RESULTS_DIR / "table_5_9_survey_criteria",
                 title="Tabela 5.9: Zestawienie średnich ocen ankietowych w pięciostopniowej skali Likerta (system LUMIR, nagrania 1–5)")

    # --- 2. Porównanie z wariantem kontrolnym (Welch t-test) ---
    ctl_flat = df[control_cols].values.flatten()
    ctl_clean = ctl_flat[~np.isnan(ctl_flat)]
    mean_ctl = float(np.mean(ctl_clean))
    std_ctl = float(np.std(ctl_clean, ddof=1))

    t_stat, p_welch = stats.ttest_ind(lum_clean, ctl_clean, equal_var=False)

    print("\n--- Porównanie LUMIR vs Wariant Kontrolny (próbka 6) ---")
    print(f"LUMIR (próbki 1–5):   M = {mean_overall_lumir:.2f}, SD = {std_overall_lumir:.2f} (n={len(lum_clean)})")
    print(f"Kontrolny (próbka 6): M = {mean_ctl:.2f}, SD = {std_ctl:.2f} (n={len(ctl_clean)})")
    print(f"Test t Welcha:        t = {t_stat:.2f}, p = {p_welch:.4f} (istotna przewaga wariantu kontrolnego)")

    df_comparison = pd.DataFrame([
        {"Wariant": "LUMIR (nagrania 1–5)", "Średnia (M)": f"{mean_overall_lumir:.2f}".replace(".", ","),
         "Odch. std. (SD)": f"{std_overall_lumir:.2f}".replace(".", ","), "Liczba ocen": len(lum_clean)},
        {"Wariant": "Kontrolny (losowy, nagranie 6)", "Średnia (M)": f"{mean_ctl:.2f}".replace(".", ","),
         "Odch. std. (SD)": f"{std_ctl:.2f}".replace(".", ","), "Liczba ocen": len(ctl_clean)},
        {"Wariant": "Różnica (Welch t-test)", "Średnia (M)": f"t = {t_stat:.2f}".replace(".", ","),
         "Odch. std. (SD)": f"p = {p_welch:.4f}".replace(".", ","), "Liczba ocen": "-"},
    ])
    export_table(df_comparison, RESULTS_DIR / "comparison_lumir_vs_control",
                 title="Porównanie średnich ocen systemu LUMIR względem wariantu kontrolnego")

    # --- 3. Tabela 5.10: Średnie oceny w podziale na grupy doświadczenia ---
    exp_col = [c for c in df.columns if "doświadczenie" in c.lower()][0]
    df["lumir_mean"] = df[lumir_cols].mean(axis=1)

    group_mapping = {
        "Brak (jestem po prostu odbiorcą / widzem)": "Zwykli widzowie (brak doświadczenia)",
        "Hobbystyczne lub teoretyczne (interesuję się tym, mam podstawową wiedzę)": "Hobbyści (wiedza teoretyczna)",
        "Profesjonalne (pracuję w branży / realizuję światło lub dźwięk)": "Profesjonalni realizatorzy",
    }

    group_order = [
        "Zwykli widzowie (brak doświadczenia)",
        "Hobbyści (wiedza teoretyczna)",
        "Profesjonalni realizatorzy",
    ]

    group_rows = []
    group_data_arrays = []

    for orig_name, clean_name in group_mapping.items():
        grp_df = df[df[exp_col] == orig_name]
        m = float(grp_df["lumir_mean"].mean())
        s = float(grp_df["lumir_mean"].std(ddof=1))
        pct = len(grp_df) / len(df) * 100
        group_rows.append({
            "Grupa": clean_name,
            "Liczebność (n)": len(grp_df),
            "Średnia": m,
            "Odch. std.": s,
            "Udział w próbie": f"{pct:.1f}%".replace(".", ","),
        })

    df_groups = pd.DataFrame(group_rows)
    df_groups.sort_values(by="Średnia", ascending=False, inplace=True)

    df_groups_display = df_groups.copy()
    df_groups_display["Średnia"] = df_groups_display["Średnia"].map(lambda x: f"{x:.2f}".replace(".", ","))
    df_groups_display["Odch. std."] = df_groups_display["Odch. std."].map(lambda x: f"{x:.2f}".replace(".", ","))

    print("\n--- Tabela 5.10: Średnie oceny w podziale na grupy doświadczenia ---")
    print(df_groups_display.to_string(index=False))

    export_table(df_groups_display, RESULTS_DIR / "table_5_10_survey_groups",
                 title="Tabela 5.10: Średnie oceny systemu LUMIR w podziale na grupy doświadczenia respondentów")

    # --- 4. Jednoczynnikowa ANOVA i test post-hoc Tukeya ---
    groups_list = [df[df[exp_col] == orig]["lumir_mean"].dropna().values for orig in group_mapping.keys()]
    f_stat, p_anova = stats.f_oneway(*groups_list)

    print(f"\nJednoczynnikowa analiza wariancji (ANOVA): F = {f_stat:.2f}, p = {p_anova:.4f}")
    if p_anova < 0.01:
        print("-> Wpływ poziomu doświadczenia na ocenę jest statystycznie istotny (p < 0.01).")

    tukey_res = stats.tukey_hsd(*groups_list)
    names_clean = list(group_mapping.values())
    print("\nTest Tukeya (porównania parami):")
    tukey_rows = []
    for i in range(len(names_clean)):
        for j in range(i + 1, len(names_clean)):
            p_val = tukey_res.pvalue[i, j]
            diff = tukey_res.statistic[i, j]
            print(f"  {names_clean[i]} vs {names_clean[j]}: różnica = {diff:.2f}, p = {p_val:.4f}")
            tukey_rows.append({
                "Porównanie": f"{names_clean[i]} vs {names_clean[j]}",
                "Różnica średnich": f"{diff:.2f}".replace(".", ","),
                "Wartość p": f"{p_val:.4f}".replace(".", ","),
                "Istotność (alfa=0.05)": "TAK" if p_val < 0.05 else "NIE",
            })

    export_table(pd.DataFrame(tukey_rows), RESULTS_DIR / "anova_tukey_posthoc",
                 title="Wyniki jednoczynnikowej analizy wariancji ANOVA i testu post-hoc Tukeya")

    # --- 5. Wizualizacje ---
    setup_plot_style()

    # Wykres 1: Słupki kryteriów (LUMIR vs Kontrolny)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(criteria_names))
    width = 0.35

    lum_means = [df_table_5_9.loc[df_table_5_9["Kryterium"] == c, "Średnia"].values[0] for c in criteria_names]
    lum_stds = [df_table_5_9.loc[df_table_5_9["Kryterium"] == c, "Odch. std."].values[0] for c in criteria_names]

    ctl_means = [float(np.mean(df[control_cols[i]].dropna())) for i in range(5)]
    ctl_stds = [float(np.std(df[control_cols[i]].dropna(), ddof=1)) for i in range(5)]

    ax.bar(x - width / 2, lum_means, width, yerr=lum_stds, label="LUMIR (automatyczny, 1–5)", color="#2b5c8f",
           capsize=4, alpha=0.9)
    ax.bar(x + width / 2, ctl_means, width, yerr=ctl_stds, label="Kontrolny (losowy, 6)", color="#d95f02", capsize=4,
           alpha=0.9)

    ax.set_ylabel("Ocena w skali Likerta (1–5)")
    ax.set_title("Oceny w podziale na kryteria ewaluacji (Tabela 5.9)")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names, rotation=15, ha="right")
    ax.set_ylim(1, 5)
    ax.axhline(3.5, color="red", linestyle="--", linewidth=1.2, label="Założony próg sukcesu (3,5)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_plot(fig, RESULTS_DIR / "plot_survey_criteria.png")
    plt.close(fig)

    # Wykres 2: Oceny per grupa doświadczenia
    fig, ax = plt.subplots(figsize=(8, 5))
    group_box_data = [df[df[exp_col] == orig]["lumir_mean"].dropna().values for orig in group_mapping.keys()]
    bplot = ax.boxplot(group_box_data, tick_labels=list(group_mapping.values()), patch_artist=True, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black",
                                  "markersize": 7})

    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    ax.axhline(3.5, color="red", linestyle="--", linewidth=1.5, label="Próg sukcesu H4 (3,50)")
    ax.set_ylabel("Średnia ocena pokazu (1–5)")
    ax.set_title("Średnie oceny systemu LUMIR w podziale na grupy doświadczenia (Tabela 5.10)")
    ax.set_ylim(1.5, 5.0)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_plot(fig, RESULTS_DIR / "plot_survey_groups.png")
    plt.close(fig)

    print("\nEksperyment 4 (Badanie ankietowe): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
