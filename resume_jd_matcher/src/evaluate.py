"""
Benchmarks the matcher's scores against the 20 manually labeled resume/JD pairs:
  - Point-biserial correlation between the continuous match score and the binary label
  - Top-1 ranking accuracy: for each resume, is its highest-scoring JD the correct one?
  - Precision / Recall / F1 at a chosen score threshold

Run directly:
    python src/evaluate.py
"""
import os
import sys

import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matcher import fit_on_sample_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def load_data():
    resumes_df = pd.read_csv(os.path.join(DATA_DIR, "resumes.csv")).set_index("resume_id")
    jds_df = pd.read_csv(os.path.join(DATA_DIR, "job_descriptions.csv")).set_index("jd_id")
    pairs_df = pd.read_csv(os.path.join(DATA_DIR, "labeled_pairs.csv"))
    return resumes_df, jds_df, pairs_df


def score_all_pairs(matcher, resumes_df, jds_df, pairs_df) -> pd.DataFrame:
    scores = []
    for _, row in pairs_df.iterrows():
        resume_text = resumes_df.loc[row["resume_id"], "text"]
        jd_text = jds_df.loc[row["jd_id"], "text"]
        score = matcher.score(resume_text, jd_text)
        scores.append(score)
    result = pairs_df.copy()
    result["score"] = scores
    return result


def top1_ranking_accuracy(scored_df: pd.DataFrame) -> float:
    """For each resume, check whether the JD it scored highest against is the
    one labeled as the true match (only counted for resumes that HAVE a true
    match in the JD set)."""
    correct = 0
    total = 0
    for resume_id, group in scored_df.groupby("resume_id"):
        true_matches = group[group["label"] == 1]
        if true_matches.empty:
            continue  # this resume has no correct JD in the set (e.g. the PM resume)
        total += 1
        top_jd = group.loc[group["score"].idxmax(), "jd_id"]
        true_jd = true_matches.iloc[0]["jd_id"]
        if top_jd == true_jd:
            correct += 1
    return correct / total if total else float("nan")


def precision_recall_f1_at_threshold(scored_df: pd.DataFrame, threshold_percentile: float = 80):
    """Predicts a pair as a 'match' if its score is at or above the given
    percentile of all scores (default: top 20%, i.e. the 80th percentile cutoff),
    then compares against the true labels."""
    cutoff = scored_df["score"].quantile(threshold_percentile / 100)
    predicted = (scored_df["score"] >= cutoff).astype(int)
    actual = scored_df["label"]

    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    return precision, recall, f1, cutoff


def run_evaluation():
    matcher = fit_on_sample_data()
    resumes_df, jds_df, pairs_df = load_data()
    scored_df = score_all_pairs(matcher, resumes_df, jds_df, pairs_df)

    # Point-biserial correlation: score (continuous) vs label (binary)
    correlation, p_value = stats.pointbiserialr(scored_df["label"], scored_df["score"])

    top1_acc = top1_ranking_accuracy(scored_df)

    precision, recall, f1, cutoff = precision_recall_f1_at_threshold(scored_df, threshold_percentile=80)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")

    lines = []
    lines.append("Resume-JD Matcher — Evaluation Report")
    lines.append("=" * 40)
    lines.append(f"Pairs evaluated: {len(scored_df)}")
    lines.append("")
    lines.append(f"Point-biserial correlation (score vs. label): {correlation:.4f}  (p={p_value:.4g})")
    lines.append(f"Top-1 ranking accuracy: {top1_acc * 100:.1f}%")
    lines.append(f"Precision / Recall / F1 @ top-20% threshold (score >= {cutoff:.2f}): "
                 f"{precision:.2f} / {recall:.2f} / {f1:.2f}")
    lines.append("")
    lines.append("Per-pair breakdown")
    lines.append("-" * 40)
    for _, row in scored_df.sort_values(["resume_id", "score"], ascending=[True, False]).iterrows():
        lines.append(
            f"  {row['resume_id']:<22} vs {row['jd_id']:<22}  "
            f"score={row['score']:6.2f}  label={row['label']}"
        )

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFull report written to: {report_path}")
    return {
        "correlation": correlation,
        "top1_accuracy": top1_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    run_evaluation()
