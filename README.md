# LUMIR — Materiały badawcze i replikacja eksperymentów

Repozytorium badawcze towarzyszące pracy magisterskiej:  
**„Zastosowanie metod wyszukiwania informacji muzycznej (MIR) w czasie rzeczywistym do automatyzacji oświetlenia
scenicznego”**  
(*Application of real-time Music Information Retrieval (MIR) methods to stage lighting automation*)

- **Autor:** inż. Paweł Müller
- **Promotor:** dr inż. Mateusz Modrzejewski
- **Uczelnia:** Politechnika Warszawska, Wydział Elektroniki i Technik Informacyjnych (WEiTI)
- **Rok akademicki:** 2025/2026

Niniejsze repozytorium zawiera kompletny zestaw danych telemetrycznych, anotacji eksperckich, surowych wyników badań
ankietowych, próbek audiowizualnych oraz skryptów analitycznych umożliwiających 100% replikację wyników empirycznych
przedstawionych w **Rozdziale 5** pracy magisterskiej.

---

## Struktura katalogów

Układ repozytorium ściśle odzwierciedla podział Rozdziału 5 na procedury badawcze:

```text
MGR_public/
├── README.md                           # Niniejsza dokumentacja środowiska i replikacji
├── requirements.txt                    # Wymagane zależności biblioteczne języka Python
├── experiments/                        # Kompletny pakiet badawczy Rozdziału 5
│   ├── common/                         # Współdzielone funkcje pomocnicze (ścieżki, eksport tabel, wykresy)
│   │   ├── __init__.py
│   │   └── utils.py
│   │
│   ├── exp1_latency/                   # Rozdział 5.2: Pomiar opóźnień pętli przetwarzania i sieci
│   │   ├── scripts/
│   │   │   ├── benchmark_pipeline.py   # Profilowanie czasu bloku audio: offline, realtime, livestream (Tabela 5.1)
│   │   │   └── benchmark_http_webapi.py# Profilowanie i analiza opóźnień HTTP WebAPI: Ethernet vs Wi-Fi (Tabela 5.2)
│   │   ├── data/                       # 15 logów CSV telemetrii audio oraz 2 serie po 500 pomiarów HTTP
│   │   └── results/                    # Wygenerowane tabele (MD/CSV) oraz wykresy rozkładów (PNG)
│   │
│   ├── exp2_features/                  # Rozdział 5.3: Walidacja ekstrakcji cech (RMS, centroid, onsety, BPM)
│   │   ├── scripts/
│   │   │   ├── validate_continuous.py  # Korelacja Pearsona obwiedni RMS i centroidu widmowego (Tabela 5.3)
│   │   │   ├── validate_onsets.py      # Precyzja, czułość i F1 onsetów surowych i filtrowanych (Tabele 5.4 i 5.5)
│   │   │   └── validate_bpm.py         # Estymacja BPM, błąd względny i czas stabilizacji (Tabela 5.6)
│   │   ├── reference/                  # Dane referencyjne offline z biblioteki librosa (.npz) dla 5 utworów
│   │   ├── data/                       # Zrzuty telemetrii wyekstrahowanej w czasie rzeczywistym przez LUMIR
│   │   └── results/                    # Zestawienia tabelaryczne metryk MIR oraz wykresy przebiegów czasowych
│   │
│   ├── exp3_classifier/                # Rozdział 5.4: Klasyfikacja stanu energetycznego (LOW, MID, HIGH)
│   │   ├── scripts/
│   │   │   ├── evaluate_classifier.py  # Ewaluacja wieloczynnikowego klasyfikatora LUMIR vs Ground Truth (Tabela 5.7)
│   │   │   └── baseline_classifier.py  # Ewaluacja modelu bazowego (tercyle dynamiczne RMS) vs LUMIR (Tabela 5.8)
│   │   ├── annotations/
│   │   │   ├── expert_annotations.py   # Anotacje referencyjne dwóch ekspertów (przedziały czasowe)
│   │   │   └── ground_truth_blocks.csv # Zestawienie etykiet referencyjnych dla wszystkich 2420 bloków
│   │   ├── data/                       # Dane telemetryczne z predykcjami stanów
│   │   └── results/                    # Macierz pomyłek, raporty klasyfikacji, współczynnik kappa Cohena
│   │
│   └── exp4_survey/                    # Rozdziały 5.5–5.6: Subiektywna ocena scenografii świetlnej
│       ├── scripts/
│       │   └── analyze_survey.py       # Średnie Likerta, test Welcha, podział na grupy, ANOVA, test Tukeya
│       ├── data/
│       │   ├── survey_raw.csv          # Surowe odpowiedzi 64 respondentów w skali Likerta (Google Forms)
│       │   └── survey_cleaned.csv      # Oczyszczony arkusz danych ze znormalizowanymi skalami
│       └── results/                    # Zestawienia średnich (Tabele 5.9, 5.10), analizy ANOVA i wykresy słupkowe
│
└── videos/                             # Załącznik B: Zestawienie 6 próbek demonstracyjnych (Full HD 60 fps)
    ├── [LUMIR] Sample 1.mp4            # Kanye West – All of the Lights (LUMIR)
    ├── [LUMIR] Sample 2.mp4            # Kwiat Jabłoni – Wodymidaj (LUMIR)
    ├── [LUMIR] Sample 3.mp4            # AC/DC – Thunderstruck (LUMIR)
    ├── [LUMIR] Sample 4.mp4            # The Weeknd – Less Than Zero (LUMIR)
    ├── [LUMIR] Sample 5.mp4            # Jady – Dead Flowers (LUMIR)
    └── [LUMIR] Sample 6.mp4            # Kanye West – All of the Lights (Wariant kontrolny / losowy)
```

---

## Mapowanie tabel z pracy magisterskiej na skrypty

| Tabela w pracy  | Zakres tematyczny                                               | Skrypt generujący                                            | Plik wynikowy                           |
|:----------------|:----------------------------------------------------------------|:-------------------------------------------------------------|:----------------------------------------|
| **Tabela 5.1**  | Czas przetwarzania bloku audio (offline, real-time, livestream) | `experiments/exp1_latency/scripts/benchmark_pipeline.py`     | `table_5_1_pipeline_latency.md`         |
| **Tabela 5.2**  | Latencja pojedynczego żądania HTTP WebAPI (Ethernet vs Wi-Fi)   | `experiments/exp1_latency/scripts/benchmark_http_webapi.py`  | `table_5_2_http_latency.md`             |
| **Tabela 5.3**  | Zgodność cech ciągłych (korelacja Pearsona RMS i centroidu)     | `experiments/exp2_features/scripts/validate_continuous.py`   | `table_5_3_continuous_correlation.md`   |
| **Tabela 5.4**  | Detekcja onsetów: Moduł MIR (detekcja surowa)                   | `experiments/exp2_features/scripts/validate_onsets.py`       | `table_5_4_onsets_mir_raw.md`           |
| **Tabela 5.5**  | Detekcja onsetów: Moduł Decyzyjny (po filtracji i wygaszeniu)   | `experiments/exp2_features/scripts/validate_onsets.py`       | `table_5_5_onsets_decision_filtered.md` |
| **Tabela 5.6**  | Estymacja tempa BPM, błąd względny i czas stabilizacji          | `experiments/exp2_features/scripts/validate_bpm.py`          | `table_5_6_bpm_estimation.md`           |
| **Tabela 5.7**  | Macierz pomyłek klasyfikacji makrostanów energetycznych         | `experiments/exp3_classifier/scripts/evaluate_classifier.py` | `table_5_7_confusion_matrix.md`         |
| **Tabela 5.8**  | Miary statystyczne stanu energetycznego (LUMIR vs baseline)     | `experiments/exp3_classifier/scripts/baseline_classifier.py` | `table_5_8_classifier_metrics.md`       |
| **Tabela 5.9**  | Średnie oceny ankietowe w 5 kryteriach (skala Likerta)          | `experiments/exp4_survey/scripts/analyze_survey.py`          | `table_5_9_survey_criteria.md`          |
| **Tabela 5.10** | Średnie oceny ankietowe w podziale na grupy doświadczenia       | `experiments/exp4_survey/scripts/analyze_survey.py`          | `table_5_10_survey_groups.md`           |

---

## Wymagania i instalacja środowiska

Badania przeprowadzono w środowisku języka Python 3.12 na platformie macOS Sequoia 15.7.

### 1. Przygotowanie wirtualnego środowiska

```bash
git clone https://github.com/pawelmuller/lumir.git
cd lumir
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Wymagane biblioteki zewnętrzne

- `numpy` (>= 1.26.0)
- `pandas` (>= 2.0.0)
- `scipy` (>= 1.11.0)
- `matplotlib` (>= 3.8.0)
- `scikit-learn` (>= 1.3.0)
- `mir_eval` (>= 0.7)
- `librosa` (>= 0.10.0)
- `requests` (>= 2.31.0)

---

## Replikacja analiz z poziomu CLI

Każdy skrypt można uruchomić niezależnie bezpośrednio z katalogu głównego repozytorium:

### Eksperyment 1: Pomiary opóźnień (Sekcja 5.2)

```bash
# 1. Profilowanie opóźnień bloku audio w pętli DSP (Tabela 5.1 oraz wykres rozkładów)
python experiments/exp1_latency/scripts/benchmark_pipeline.py

# 2. Analiza opóźnień transmisji HTTP WebAPI Ethernet vs Wi-Fi (Tabela 5.2)
python experiments/exp1_latency/scripts/benchmark_http_webapi.py
```

### Eksperyment 2: Walidacja ekstrakcji cech MIR (Sekcja 5.3)

```bash
# 1. Obliczenie korelacji Pearsona cech ciągłych (Tabela 5.3 oraz przebiegi časowe)
python experiments/exp2_features/scripts/validate_continuous.py

# 2. Ewaluacja precyzji, czułości i miary F1 detekcji onsetów (Tabele 5.4 i 5.5)
python experiments/exp2_features/scripts/validate_onsets.py

# 3. Ewaluacja estymacji tempa BPM i czasu stabilizacji (Tabela 5.6)
python experiments/exp2_features/scripts/validate_bpm.py
```

### Eksperyment 3: Klasyfikacja stanu energetycznego (Sekcja 5.4)

```bash
# 1. Macierz pomyłek i metryki Accuracy, Macro-F1, kappa Cohena (Tabela 5.7)
python experiments/exp3_classifier/scripts/evaluate_classifier.py

# 2. Porównanie z klasyfikatorem bazowym tercyli dynamicznych RMS (Tabela 5.8)
python experiments/exp3_classifier/scripts/baseline_classifier.py
```

### Eksperyment 4: Badanie ankietowe (Sekcje 5.5–5.6)

```bash
# Analiza statystyczna ankiety: kryteria Likerta, test Welcha, grupy doświadczenia, ANOVA i test Tukeya
python experiments/exp4_survey/scripts/analyze_survey.py
```

---

## Materiały demonstracyjne (Wideo)

Wszystkie nagrania wideo umieszczone w katalogu `videos/` zarejestrowano w środowisku symulatora 3D Capture
Visualisation (v2025.1.15) zintegrowanego z konsolą Avolites Titan (v19.1) w rozdzielczości Full HD przy 60 fps:

- `[LUMIR] Sample 1.mp4` — Kanye West – *All of the Lights* (LUMIR, Hip-hop)
- `[LUMIR] Sample 2.mp4` — Kwiat Jabłoni – *Wodymidaj* (LUMIR, Folk-pop)
- `[LUMIR] Sample 3.mp4` — AC/DC – *Thunderstruck* (LUMIR, Hard Rock)
- `[LUMIR] Sample 4.mp4` — The Weeknd – *Less Than Zero* (LUMIR, Synth-pop)
- `[LUMIR] Sample 5.mp4` — Jady – *Dead Flowers* (LUMIR, Indie / Alt-rock)
- `[LUMIR] Sample 6.mp4` — Kanye West – *All of the Lights* (Wariant kontrolny / pseudolosowy)

---

## Licencja

Projekt udostępniony jest na licencji MIT. Dane ankietowe zostały zanonimizowane zgodnie z przepisami o ochronie danych
osobowych.
