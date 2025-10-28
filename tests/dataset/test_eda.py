from aidose import END_POINT_HF_DATASET_PATH, RESOURCES_DIR

from datasets import load_from_disk, DatasetDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import unittest
import json
from pathlib import Path
from typing import Dict, List
import os



def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_df_csv(df: pd.DataFrame, out: Path):
    df.to_csv(out, index=True)


def _save_json(obj, out: Path):
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _plot_hist(data: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 30):
    fig = plt.figure()
    plt.hist(data.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_barh(index: List[str], values: List[float], title: str, xlabel: str, out_path: Path):
    fig = plt.figure()
    y_pos = np.arange(len(index))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, index)
    plt.xlabel(xlabel)
    plt.title(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_line(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


class ClinicalTrialsDatasetEDAWriteoutTest(unittest.TestCase):
    """
    EDA for the CT-DOSING-ERRORS dataset.
    Writes CSV/JSON + PNG plots to disk (no assertions).
    """

    @classmethod
    def setUpClass(cls):
        out_dir = os.path.join(RESOURCES_DIR, "EDA-TEST", "CT-DOSING-ERRORS")
        cls.dataset: DatasetDict = load_from_disk(END_POINT_HF_DATASET_PATH)
        cls.out_root = _ensure_dir(Path(out_dir))
        cls.split_dirs = {split: _ensure_dir(cls.out_root / split) for split in cls.dataset.keys()}
        cls.global_dir = _ensure_dir(cls.out_root / "_global")

        # Convert splits to pandas (preserving HF dtypes best-effort)
        cls.dfs: Dict[str, pd.DataFrame] = {split: ds.to_pandas() for split, ds in cls.dataset.items()}

        # Column categories
        sample_df = next(iter(cls.dfs.values()))
        cls.feature_cols = [c for c in sample_df.columns if c.startswith("FEATURE_")]
        cls.meta_cols = [c for c in sample_df.columns if c.startswith("METADATA_")]
        cls.label_cols = [c for c in sample_df.columns if c.startswith("LABEL_")]

        # ADE-related term counts
        cls.ade_count_cols = [c for c in cls.meta_cols if c.startswith("METADATA_count_")]

        # Dates if present
        cls.start_col = "METADATA_startDate" if "METADATA_startDate" in sample_df.columns else None
        cls.complete_col = "METADATA_completionDate" if "METADATA_completionDate" in sample_df.columns else None

        # Save a manifest
        manifest = {
            "dataset_path": END_POINT_HF_DATASET_PATH,
            "splits": list(cls.dataset.keys()),
            "n_columns": {split: int(df.shape[1]) for split, df in cls.dfs.items()},
            "n_rows": {split: int(len(df)) for split, df in cls.dfs.items()},
            "feature_counts": {
                "FEATURE": len(cls.feature_cols),
                "METADATA": len(cls.meta_cols),
                "LABEL": len(cls.label_cols),
                "ADE_count_columns": len(cls.ade_count_cols),
            },
            "date_columns": {"startDate": bool(cls.start_col), "completionDate": bool(cls.complete_col)},
        }
        _save_json(manifest, cls.global_dir / "manifest.json")

    # --------------------- CORE SUMMARIES ---------------------

    def test_overall_sizes_and_dtypes(self):
        # Sizes per split
        sizes = []
        for split, df in self.dfs.items():
            sizes.append({"split": split, "rows": len(df), "columns": df.shape[1]})
        _save_df_csv(pd.DataFrame(sizes).set_index("split"), self.global_dir / "sizes_per_split.csv")

        # Dtypes for train (representative)
        train_df = self.dfs.get("train", next(iter(self.dfs.values())))
        dtype_counts = train_df.dtypes.value_counts().rename_axis("dtype").to_frame("count")
        _save_df_csv(dtype_counts, self.global_dir / "dtypes_train.csv")

        # Column category listings
        _save_df_csv(pd.Series(self.feature_cols, name="FEATURE_columns").to_frame(), self.global_dir / "FEATURE_columns.csv")
        _save_df_csv(pd.Series(self.meta_cols, name="METADATA_columns").to_frame(), self.global_dir / "METADATA_columns.csv")
        _save_df_csv(pd.Series(self.label_cols, name="LABEL_columns").to_frame(), self.global_dir / "LABEL_columns.csv")

    def test_missing_values(self):
        for split, df in self.dfs.items():
            miss = df.isna().sum()
            miss_df = miss[miss > 0].sort_values(ascending=False).rename("n_missing").to_frame()
            _save_df_csv(miss_df, self.split_dirs[split] / "missing_values.csv")

        # Global union of columns with any missing
        cols_with_missing = set()
        for df in self.dfs.values():
            cols_with_missing |= set(df.columns[df.isna().any(axis=0)])
        _save_df_csv(pd.DataFrame(sorted(cols_with_missing), columns=["column"]), self.global_dir / "columns_with_missing_any_split.csv")

    def test_label_statistics_and_correlations(self):
        for split, df in self.dfs.items():
            split_dir = self.split_dirs[split]
            label_descs = []
            for col in self.label_cols:
                s = df[col]
                desc = s.describe(include="all")
                label_descs.append(pd.DataFrame(desc).rename(columns={col: "value"}).assign(metric=desc.index, column=col))
            if label_descs:
                out = pd.concat(label_descs, axis=0)
                _save_df_csv(out, split_dir / "label_describe.csv")

            # numeric-only correlations among labels
            label_num = df[self.label_cols].select_dtypes(include=[np.number])
            if label_num.shape[1] >= 2:
                corr = label_num.corr()
                _save_df_csv(corr, split_dir / "label_correlations.csv")

            # Plots for key labels if present
            if "LABEL_dosing_error_rate" in df.columns:
                _plot_hist(
                    df["LABEL_dosing_error_rate"],
                    title=f"{split}: LABEL_dosing_error_rate histogram",
                    xlabel="dosing_error_rate",
                    out_path=split_dir / "plot_LABEL_dosing_error_rate_hist.png",
                    bins=40,
                )
            if "LABEL_wilson_label" in df.columns:
                # Bar plot of class balance
                vc = df["LABEL_wilson_label"].value_counts(dropna=False).sort_index()
                _plot_barh(
                    index=[str(i) for i in vc.index.tolist()],
                    values=vc.values.tolist(),
                    title=f"{split}: LABEL_wilson_label counts",
                    xlabel="count",
                    out_path=split_dir / "plot_LABEL_wilson_label_counts.png",
                )

    # --------------------- FEATURE SUMMARIES ---------------------

    def test_key_feature_distributions(self):
        for split, df in self.dfs.items():
            split_dir = self.split_dirs[split]

            # Enrollment distribution if present
            if "FEATURE_enrollmentCount" in df.columns:
                _plot_hist(
                    df["FEATURE_enrollmentCount"].astype("float64"),
                    title=f"{split}: FEATURE_enrollmentCount",
                    xlabel="enrollmentCount",
                    out_path=split_dir / "plot_enrollmentCount_hist.png",
                    bins=50,
                )
                _save_df_csv(
                    df["FEATURE_enrollmentCount"].describe().to_frame(name="enrollmentCount"),
                    split_dir / "enrollmentCount_describe.csv",
                )

            # Clinical phase one-hot columns
            phase_cols = [c for c in self.feature_cols if c.startswith("FEATURE_phases.")]
            if phase_cols:
                phase_counts = df[phase_cols].sum(numeric_only=True).sort_values(ascending=False)
                _save_df_csv(phase_counts.to_frame(name="count"), split_dir / "phase_counts.csv")
                _plot_barh(
                    index=phase_counts.index.tolist()[::-1],
                    values=phase_counts.values.tolist()[::-1],
                    title=f"{split}: Clinical phases (one-hot)",
                    xlabel="count",
                    out_path=split_dir / "plot_phase_counts_barh.png",
                )

            # Intervention types (counts)
            itype_cols = [c for c in self.feature_cols if c.startswith("FEATURE_interventionTypes.")]
            if itype_cols:
                itype_counts = df[itype_cols].sum(numeric_only=True).sort_values(ascending=False)
                _save_df_csv(itype_counts.to_frame(name="count"), split_dir / "intervention_type_counts.csv")
                _plot_barh(
                    index=itype_counts.index.tolist()[::-1],
                    values=itype_counts.values.tolist()[::-1],
                    title=f"{split}: Intervention types",
                    xlabel="count",
                    out_path=split_dir / "plot_intervention_types_barh.png",
                )

            # Arm group types (counts)
            arm_cols = [c for c in self.feature_cols if c.startswith("FEATURE_armGroupTypes.")]
            if arm_cols:
                arm_counts = df[arm_cols].sum(numeric_only=True).sort_values(ascending=False)
                _save_df_csv(arm_counts.to_frame(name="count"), split_dir / "arm_group_type_counts.csv")
                _plot_barh(
                    index=arm_counts.index.tolist()[::-1],
                    values=arm_counts.values.tolist()[::-1],
                    title=f"{split}: Arm group types",
                    xlabel="count",
                    out_path=split_dir / "plot_arm_group_types_barh.png",
                )

    # --------------------- ADE / ERROR TERM SUMMARIES ---------------------

    def test_ade_term_counts(self):
        # Per split: total counts and Top-K ADE terms
        K = 25
        for split, df in self.dfs.items():
            split_dir = self.split_dirs[split]
            if not self.ade_count_cols:
                continue

            counts_df = df[self.ade_count_cols].sum(numeric_only=True).sort_values(ascending=False).to_frame("total_count")
            _save_df_csv(counts_df, split_dir / "ade_term_total_counts.csv")

            topk = counts_df.head(K)
            _plot_barh(
                index=topk.index.tolist()[::-1],
                values=topk["total_count"].tolist()[::-1],
                title=f"{split}: Top {K} ADE terms",
                xlabel="total count across studies",
                out_path=split_dir / f"plot_top_{K}_ade_terms_barh.png",
            )

        # Global: which ADE terms are most common in TRAIN and how they correlate with labels
        train_df = self.dfs.get("train")
        if train_df is not None and self.ade_count_cols:
            ade_sums = train_df[self.ade_count_cols].sum(numeric_only=True).sort_values(ascending=False).to_frame("total_count")
            _save_df_csv(ade_sums, self.global_dir / "train_ade_term_total_counts.csv")

            # Correlation (Spearman) between each ADE count col and dosing_error_rate, if available
            if "LABEL_dosing_error_rate" in train_df.columns:
                target = train_df["LABEL_dosing_error_rate"]
                rows = []
                for col in self.ade_count_cols:
                    s = train_df[col].astype("float64")
                    if s.notna().any() and target.notna().any():
                        rho = s.corr(target, method="spearman")
                    else:
                        rho = np.nan
                    rows.append({"term": col, "spearman_rho_vs_dosing_error_rate": rho})
                corr_df = pd.DataFrame(rows).sort_values("spearman_rho_vs_dosing_error_rate", ascending=False)
                _save_df_csv(corr_df, self.global_dir / "ade_term_vs_dosing_error_rate_spearman.csv")

    # --------------------- TIME-BASED SUMMARIES ---------------------

    def test_time_trends(self):
        # Aggregate by completion date year, plot counts and average dosing_error_rate
        train_df = self.dfs.get("train")
        if train_df is None:
            return
        if self.complete_col is None:
            return

        df = train_df.copy()
        df[self.complete_col] = pd.to_datetime(df[self.complete_col], errors="coerce")

        year = df[self.complete_col].dt.year.rename("year")
        yearly_counts = year.value_counts().sort_index().rename("n_studies").to_frame()
        _save_df_csv(yearly_counts, self.global_dir / "train_study_counts_by_completion_year.csv")

        if "LABEL_dosing_error_rate" in df.columns:
            grp = df.groupby(year)["LABEL_dosing_error_rate"].mean().dropna().rename("mean_dosing_error_rate")
            _save_df_csv(grp.to_frame(), self.global_dir / "train_mean_dosing_error_rate_by_year.csv")
            # plot lines
            _plot_line(
                x=yearly_counts.index.to_series(),
                y=yearly_counts["n_studies"],
                title="TRAIN: Study counts by completion year",
                xlabel="year",
                ylabel="n_studies",
                out_path=self.global_dir / "plot_train_counts_by_year.png",
            )
            _plot_line(
                x=grp.index.to_series(),
                y=grp,
                title="TRAIN: Mean dosing error rate by completion year",
                xlabel="year",
                ylabel="mean_dosing_error_rate",
                out_path=self.global_dir / "plot_train_mean_dosing_error_rate_by_year.png",
            )

    # --------------------- TEXT FIELD LENGTHS (QUALITY SIGNALS) ---------------------

    def test_text_length_qc(self):
        # Simple QC: length distributions for long text fields if present
        text_cols = [c for c in self.feature_cols if c in {
            "FEATURE_briefSummary",
            "FEATURE_detailedDescription",
            "FEATURE_protocolPdfText",
            "FEATURE_armDescriptions",
            "FEATURE_interventionDescriptions",
            "FEATURE_locationDetails",
        }]
        if not text_cols:
            return

        for split, df in self.dfs.items():
            split_dir = self.split_dirs[split]
            len_stats = {}
            for col in text_cols:
                if col in df.columns:
                    lens = df[col].astype("string").fillna("").str.len()
                    len_stats[col] = {
                        "count": int(lens.count()),
                        "mean": float(lens.mean()),
                        "std": float(lens.std(ddof=1)) if lens.count() > 1 else 0.0,
                        "min": int(lens.min()) if lens.count() else 0,
                        "p50": int(lens.quantile(0.50)) if lens.count() else 0,
                        "p90": int(lens.quantile(0.90)) if lens.count() else 0,
                        "max": int(lens.max()) if lens.count() else 0,
                    }
                    _plot_hist(
                        lens.astype("float64"),
                        title=f"{split}: length({col})",
                        xlabel="characters",
                        out_path=split_dir / f"plot_len_{col}_hist.png",
                        bins=50,
                    )
            _save_json(len_stats, split_dir / "text_length_stats.json")



if __name__ == "__main__":
    unittest.main(verbosity=2)
