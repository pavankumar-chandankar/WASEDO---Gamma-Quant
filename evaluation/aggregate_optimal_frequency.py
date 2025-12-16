import glob
import os

import numpy as np
import pandas as pd


def main():
    freq_tags = ["25hz", "12hz", "6hz"]

    all_dfs = []

    for freq in freq_tags:
        pattern = os.path.join(
            "logs", "deepconvlstm", freq, "*", "per_class_metrics_*.csv"
        )
        csv_files = glob.glob(pattern)
        if not csv_files:
            print(
                f"[WARN] No CSV files found for frequency tag '{freq}' with pattern: {pattern}"
            )
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        print(
            "No per-class metrics found. "
            "Have you trained and saved results with the updated train.py?"
        )
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    required_cols = {"class_id", "class_name", "freq_tag", "acc", "prec", "rec", "f1"}
    missing = required_cols.difference(all_df.columns)
    if missing:
        raise ValueError(f"Missing columns in aggregated DataFrame: {missing}")
    summary = (
        all_df.groupby(["class_id", "class_name", "freq_tag"])[
            ["acc", "prec", "rec", "f1"]
        ]
        .mean()
        .reset_index()
    )

    idx_best = summary.groupby("class_id")["f1"].idxmax()
    best_per_activity = summary.loc[idx_best].sort_values("class_id")

    out_path = os.path.join("logs", "deepconvlstm", "best_frequency_per_activity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_per_activity.to_csv(out_path, index=False)

    print("Best frequency per activity (by mean F1 across splits):")
    print(best_per_activity.to_string(index=False))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
