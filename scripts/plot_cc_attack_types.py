#!/usr/bin/env python3
"""Plot MONZA CC recall/FPR by attack type from cc_type_results_*.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-dir",
        type=Path,
        default=Path("PFLlibMonza/system"),
        help="Directory containing cc_type_results_*.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory where plots and summary CSV are written.",
    )
    parser.add_argument(
        "--tail-rounds",
        type=int,
        default=30,
        help="Number of final rounds used for the summary.",
    )
    return parser.parse_args()


def load_cc_type(system_dir: Path, min_rounds: int) -> pd.DataFrame:
    frames = []
    for path in sorted(system_dir.glob("cc_type_results_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = latest_run(df, min_rounds=min_rounds)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum cc_type_results_*.csv encontrado em {system_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["Defense"] = "cc=" + out["CC"].astype(str)
    return out


def latest_run(df: pd.DataFrame, min_rounds: int) -> pd.DataFrame:
    if "RunID" in df.columns:
        run_ids = list(df["RunID"].drop_duplicates())
        for run_id in reversed(run_ids):
            run = df[df["RunID"] == run_id].copy()
            if run["Round"].astype(int).nunique() >= min_rounds:
                return run
        return df[df["RunID"] == run_ids[-1]].copy()
    rounds = df["Round"].astype(int)
    starts = df.index[rounds < rounds.shift(fill_value=rounds.iloc[0])].tolist()
    start = starts[-1] if starts else 0
    return df.loc[start:].copy()


def summarize_tail(df: pd.DataFrame, tail_rounds: int) -> pd.DataFrame:
    rows = []
    for cc, cc_group in df.groupby("CC", sort=True):
        tail_round_values = sorted(cc_group["Round"].astype(int).unique())[-tail_rounds:]
        tail_cc = cc_group[cc_group["Round"].isin(tail_round_values)]
        for attack_type, group in tail_cc.groupby("AttackType", sort=True):
            total = group["Total"].sum()
            removed = group["Removed"].sum()
            rows.append(
                {
                    "CC": cc,
                    "Defense": f"cc={cc}",
                    "AttackType": attack_type,
                    "Total": int(total),
                    "Removed": int(removed),
                    "Rate": float(removed / total) if total else 0.0,
                    "Metric": "FPR" if attack_type == "benign" else "recall",
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["benign", "malicious_label", "malicious_zeros", "malicious_random", "malicious_shuffle"]
    pivot = summary.pivot_table(index="AttackType", columns="Defense", values="Rate", aggfunc="mean")
    pivot = pivot.reindex([x for x in order if x in pivot.index] + [x for x in pivot.index if x not in order])

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot.bar(ax=ax, width=0.78)
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.3, label="FPR target 5%")
    ax.set_title("FPR em benignos e recall por tipo de ataque nos CCs")
    ax.set_ylabel("Taxa")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, float(pivot.max().max()) * 1.15))
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_recall_by_attack_type.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_label(summary: pd.DataFrame, out_dir: Path) -> None:
    label = summary[summary["AttackType"] == "malicious_label"].sort_values("CC")
    if label.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(label["Defense"], label["Rate"], color="#8c564b")
    ax.set_title("Recall do CC em malicious_label")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, max(1.0, float(label["Rate"].max()) * 1.2))
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(label["Rate"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_malicious_label_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cc_type(args.system_dir, args.tail_rounds)
    summary = summarize_tail(df, args.tail_rounds)
    summary.to_csv(args.out_dir / "cc_attack_type_summary.csv", index=False)
    plot_summary(summary, args.out_dir)
    plot_label(summary, args.out_dir)
    print(summary.sort_values(["CC", "AttackType"]).to_string(index=False))
    print(f"\nArquivos salvos em {args.out_dir}")


if __name__ == "__main__":
    main()
