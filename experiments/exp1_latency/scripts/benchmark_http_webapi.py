#!/usr/bin/env python3
"""Eksperyment 1 (Sekcja 5.2): Benchmark i analiza opóźnień komunikacji HTTP WebAPI z konsolą Titan.

Porównuje parametry transmisji sieciowej dla dwóch mediów:
1. Ethernet (przewodowe)
2. Wi-Fi (bezprzewodowe)
"""

from __future__ import annotations

import argparse
import sys
import time
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

from common.utils import export_table, setup_plot_style, save_plot, setup_lumir_import

setup_lumir_import()


def analyze_latencies(csv_file: Path, interface_label: str) -> dict:
    """Analyze latency distribution from CSV log."""
    if not csv_file.exists():
        raise FileNotFoundError(f"Plik danych {csv_file} nie istnieje.")

    df = pd.read_csv(csv_file)
    lat_arr = df["latency_ms"].dropna().values

    return {
        "Medium transmisyjne": interface_label,
        "Liczba próbek (n)": len(lat_arr),
        "Średnia [ms]": float(np.mean(lat_arr)),
        "Mediana [ms]": float(np.median(lat_arr)),
        "P95 [ms]": float(np.percentile(lat_arr, 95)),
        "P99 [ms]": float(np.percentile(lat_arr, 99)),
        "Maks. [ms]": float(np.max(lat_arr)),
        "Jitter (SD) [ms]": float(np.std(lat_arr, ddof=1)),
        "_raw": lat_arr,
    }


def run_live_benchmark(
        host: str = "127.0.0.1",
        port: int = 4430,
        endpoint: str = "/titan/script/Playbacks/FirePlaybackAtLevel",
        iterations: int = 500,
        delay_ms: float = 50.0,
        interface_label: str = "Ethernet",
        output_csv: Path | None = None,
) -> dict:
    """Execute live HTTP WebAPI benchmark against Avolites Titan console."""
    import requests

    url = f"http://{host}:{port}{endpoint}"
    params = {"userNumber": 1, "level": "0.0000", "bool": "false"}
    print(f"\nUruchamianie testu na żywo [{interface_label}]: {url}, n={iterations}")

    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    latencies = []

    # Rozgrzewka (warmup)
    try:
        session.get(url, params=params, timeout=2.0)
    except Exception as e:
        print(f"[Ostrzeżenie] Warmup request nie powiódł się: {e}")

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        try:
            resp = session.get(url, params=params, timeout=2.0)
            if resp.status_code < 500:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
        except requests.RequestException as e:
            print(f"[{i}/{iterations}] Błąd: {e}")

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    session.close()
    lat_arr = np.array(latencies)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_idx": range(1, len(lat_arr) + 1), "latency_ms": lat_arr}).to_csv(output_csv, index=False)
        print(f"Zapisano wyniki do: {output_csv}")

    return {
        "Medium transmisyjne": interface_label,
        "Liczba próbek (n)": len(lat_arr),
        "Średnia [ms]": float(np.mean(lat_arr)),
        "Mediana [ms]": float(np.median(lat_arr)),
        "P95 [ms]": float(np.percentile(lat_arr, 95)),
        "P99 [ms]": float(np.percentile(lat_arr, 99)),
        "Maks. [ms]": float(np.max(lat_arr)),
        "Jitter (SD) [ms]": float(np.std(lat_arr, ddof=1)),
        "_raw": lat_arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Analiza i benchmark opóźnień sieciowych HTTP WebAPI")
    parser.add_argument("--live", action="store_true", help="Uruchom pomiar na żywo z konsolą Titan")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Adres hosta konsoli Titan")
    parser.add_argument("--port", type=int, default=4430, help="Port WebAPI (domyślnie 4430)")
    parser.add_argument("--runs", type=int, default=500, help="Liczba zapytań testowych (domyślnie 500)")
    parser.add_argument("--interface", type=str, default="Ethernet", help="Etykieta medium (Ethernet / Wi-Fi)")
    args = parser.parse_args()

    print("=" * 65)
    print("  EKSPERYMENT 1: Pomiar opóźnień sieciowych HTTP WebAPI (Tabela 5.2)")
    print("=" * 65)

    if args.live:
        target_csv = DATA_DIR / f"http_latency_{args.interface.lower().replace('-', '_')}.csv"
        res = run_live_benchmark(
            host=args.host,
            port=args.port,
            iterations=args.runs,
            interface_label=args.interface,
            output_csv=target_csv,
        )
        print(res)
        return

    # Tryb analizy istniejących danych telemetrycznych
    eth_file = DATA_DIR / "http_latency_ethernet.csv"
    wifi_file = DATA_DIR / "http_latency_wifi.csv"

    results = []
    raw_data = {}

    for path, label in [(eth_file, "Ethernet (przewodowe)"), (wifi_file, "Wi-Fi (bezprzewodowe)")]:
        if path.exists():
            r = analyze_latencies(path, label)
            raw_data[label] = r.pop("_raw")
            results.append(r)
        else:
            print(f"[OSTRZEŻENIE] Brak pliku: {path}")

    if not results:
        print("[BŁĄD] Brak danych do analizy.")
        return

    df_table = pd.DataFrame(results)

    # Formatowanie zbieżne z Tabelą 5.2
    df_thesis = pd.DataFrame({
        "Medium transmisyjne": df_table["Medium transmisyjne"],
        "Średnia [ms]": df_table["Średnia [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
        "Mediana [ms]": df_table["Mediana [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
        "P95 [ms]": df_table["P95 [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
        "P99 [ms]": df_table["P99 [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
        "Maks. [ms]": df_table["Maks. [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
        "Jitter (SD) [ms]": df_table["Jitter (SD) [ms]"].map(lambda x: f"{x:.2f}".replace(".", ",")),
    })

    print("\n--- Tabela 5.2: Porównanie latencji pojedynczego żądania HTTP WebAPI (n = 500) ---")
    print(df_thesis.to_string(index=False))

    export_table(df_thesis, RESULTS_DIR / "table_5_2_http_latency",
                 title="Tabela 5.2: Porównanie latencji pojedynczego żądania HTTP do interfejsu WebAPI konsoli (n = 500)")

    # Wykresy porównawcze
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Boxplot
    bplot = ax1.boxplot(list(raw_data.values()), tick_labels=list(raw_data.keys()), patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black",
                                   "markersize": 6})
    colors = ["#2b5c8f", "#d95f02"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax1.set_ylabel("Czas odpowiedzi RTT [ms]")
    ax1.set_title("Rozrzut czasu odpowiedzi żądań HTTP WebAPI")

    # ECDF
    for (lbl, raw), c in zip(raw_data.items(), colors):
        sorted_v = np.sort(raw)
        y = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
        ax2.plot(sorted_v, y, label=f"{lbl} (Mediana={np.median(raw):.2f} ms)", color=c, linewidth=2)

    ax2.set_xlabel("Czas odpowiedzi RTT [ms]")
    ax2.set_ylabel("Dystrybuanta empiryczna [%]")
    ax2.set_title("Dystrybuanta (ECDF) opóźnień WebAPI")
    ax2.legend(loc="lower right")

    save_plot(fig, RESULTS_DIR / "plot_http_latency.png")
    plt.close(fig)
    print("\nEksperyment 1 (HTTP WebAPI): Zakończono pomyślnie.")


if __name__ == "__main__":
    main()
