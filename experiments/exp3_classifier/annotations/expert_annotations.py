"""Anotacje referencyjne ekspertów (Ground Truth) dla 2420 bloków decyzyjnych (Eksperyment 3).

Anotacja została przeprowadzona niezależnie przez dwóch ekspertów na podstawie odsłuchu
oraz podglądu spektrogramu/przebiegu czasowego sygnału (waveform).
Rozbieżności ustalono wspólnie w drodze dyskusji.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd


class GroundTruthEnergyState(Enum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


# Format: lista krotek (czas_poczatkowy_s, czas_koncowy_s, stan)
ANNOTATIONS = {
    "All_of_the_Lights": [
        (0.0, 7.27, GroundTruthEnergyState.MID),
        (7.27, 34.83, GroundTruthEnergyState.HIGH),
        (34.83, 45.0, GroundTruthEnergyState.MID),
    ],
    "Dead_Flowers": [
        (0.0, 4.34, GroundTruthEnergyState.MID),
        (4.34, 16.44, GroundTruthEnergyState.LOW),
        (16.44, 38.27, GroundTruthEnergyState.HIGH),
        (38.27, 41.45, GroundTruthEnergyState.MID),
        (41.45, 45.0, GroundTruthEnergyState.LOW),
    ],
    "Less_Than_Zero": [
        (0.0, 6.41, GroundTruthEnergyState.MID),
        (6.41, 30.32, GroundTruthEnergyState.HIGH),
        (30.32, 45.0, GroundTruthEnergyState.LOW),
    ],
    "Thunderstruck": [
        (0.0, 9.48, GroundTruthEnergyState.LOW),
        (9.48, 45.0, GroundTruthEnergyState.MID),
    ],
    "Wodymidaj": [
        (0.0, 11.30, GroundTruthEnergyState.MID),
        (11.30, 43.95, GroundTruthEnergyState.HIGH),
        (43.95, 45.0, GroundTruthEnergyState.HIGH),
    ],
}


def get_ground_truth_label(track_key: str, timestamp_s: float) -> str:
    """Return the expert ground truth label (LOW, MID, HIGH) for a given track and timestamp."""
    segments = ANNOTATIONS.get(track_key)
    if not segments:
        raise KeyError(f"Nieznany utwór: {track_key}")

    for start_t, end_t, state in segments:
        if start_t <= timestamp_s <= end_t:
            return state.value

    return segments[-1][2].value


def export_block_annotations_csv(telemetry_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Generate and export tabular Ground Truth per audio block (2420 rows)."""
    rows = []
    for track_key in sorted(ANNOTATIONS.keys()):
        csv_file = telemetry_dir / f"telemetry_{track_key.lower()}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            t = float(row["timestamp"])
            label = get_ground_truth_label(track_key, t)
            rows.append({
                "block_idx": idx,
                "track": track_key,
                "timestamp_s": t,
                "ground_truth_label": label,
                "lumir_prediction": row.get("energy_state", ""),
                "rms": row.get("rms", 0.0),
            })

    df_out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "data"
    out_file = current_dir / "ground_truth_blocks.csv"
    if data_dir.exists():
        df_exp = export_block_annotations_csv(data_dir, out_file)
        print(f"Wygenerowano anotacje blokowe ({len(df_exp)} bloków): {out_file}")
