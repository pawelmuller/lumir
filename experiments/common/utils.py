"""Common utilities for LUMIR research experiments and evaluation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def setup_lumir_import() -> None:
    """Ensure the lumir package can be imported from parent or sibling directories."""
    current = Path(__file__).resolve().parent
    exp_root = current.parent
    repo_root = exp_root.parent
    possible_roots = [
        repo_root,
        repo_root / "lumir",
        repo_root.parent / "MGR",
        repo_root.parent / "MGR" / "lumir",
        exp_root,
    ]
    for p in possible_roots:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


# Standard track metadata used across experiments
TRACK_METADATA = {
    "All_of_the_Lights": {
        "title": "All of the Lights",
        "artist": "Kanye West",
        "genre": "Hip-hop",
        "ref_bpm": 143.55,
        "sample_nr": 1,
    },
    "Wodymidaj": {
        "title": "Wodymidaj",
        "artist": "Kwiat Jabłoni",
        "genre": "Folk-pop",
        "ref_bpm": 107.67,
        "sample_nr": 2,
    },
    "Thunderstruck": {
        "title": "Thunderstruck",
        "artist": "AC/DC",
        "genre": "Hard Rock",
        "ref_bpm": 136.00,
        "sample_nr": 3,
    },
    "Less_Than_Zero": {
        "title": "Less Than Zero",
        "artist": "The Weeknd",
        "genre": "Synth-pop",
        "ref_bpm": 143.55,
        "sample_nr": 4,
    },
    "Dead_Flowers": {
        "title": "Dead Flowers",
        "artist": "Jady",
        "genre": "Indie / Alt-rock",
        "ref_bpm": 83.35,  # Librosa reference (execution tempo ~112 BPM)
        "sample_nr": 5,
    },
}

TRACK_KEYS = [
    "All_of_the_Lights",
    "Dead_Flowers",
    "Less_Than_Zero",
    "Thunderstruck",
    "Wodymidaj",
]


def export_markdown_table(df: pd.DataFrame, output_path: Path, title: str | None = None) -> None:
    """Save a DataFrame as formatted GitHub Markdown table without external dependencies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.values]

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    separator_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    data_lines = ["| " + " | ".join(val.ljust(w) for val, w in zip(r, col_widths)) + " |" for r in rows]

    md_content = ""
    if title:
        md_content += f"# {title}\n\n"
    md_content += "\n".join([header_line, separator_line] + data_lines) + "\n"
    output_path.write_text(md_content, encoding="utf-8")


def export_table(df: pd.DataFrame, base_path: Path, title: str | None = None) -> None:
    """Export DataFrame both as CSV and Markdown table."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    md_path = base_path.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    export_markdown_table(df, md_path, title=title)
    print(f"Exported: {csv_path.name} & {md_path.name}")


def setup_plot_style() -> None:
    """Apply consistent, professional styling for scientific publication plots."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.autolayout": True,
            "savefig.dpi": 300,
        })
    except ImportError:
        pass


def save_plot(fig, output_path: Path, dpi: int = 300) -> None:
    """Save matplotlib figure to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path.name}")
