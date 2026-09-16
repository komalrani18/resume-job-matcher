"""
evaluate.py
-----------
Evaluates matcher scores against manually labeled resume-JD pairs.
Since labels here are binary (1 = relevant, 0 = not relevant) rather
than a continuous relevance score, we evaluate by:

  1. Correlation between predicted similarity and human label
  2. Ranking accuracy: for each resume, does the matcher rank the
     human-labeled "relevant" JD above all "not relevant" JDs?
  3. Threshold classification metrics at a chosen cutoff

Run:
    python src/evaluate.py
Output:
    reports/evaluation_report.txt
"""

from pathlib import Path

import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from matcher import ResumeJDMatcher, build_and_save_default_matcher

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"
THRESHOLD = 20.0  # match-score cutoff (%) for "recommend this pairing"


def load_data():
    resumes = pd.read_csv(DATA_DIR / "resumes.csv").set_index("resume_id")
    jds = pd.read_csv(DATA_DIR / "job_descriptions.csv").set_index("jd_id")
    labeled = pd.read_csv(DATA_DIR / "labeled_pairs.csv")
    return resumes, jds, labeled


def score_all_pairs(matcher: ResumeJDMatcher, resumes, jds, labeled: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for _, row in labeled.iterrows():
        r_text = resumes.loc[row.resume_id, "text"]
        j_text = jds.loc[row.jd_id, "text"]
        scores.append(matcher.score(r_text, j_text))
    out = labeled.copy()
    out["predicted_score"] = scores
    out["predicted_label"] = (out["predicted_score"] >= THRESHOLD).astype(int)
    return out


def ranking_accuracy(scored: pd.DataFrame) -> float:
    """For each resume, check whether the top-scored JD matches the
    human-labeled relevant one."""
    correct = 0
    total = 0
    for resume_id, group in scored.groupby("resume_id"):
        if group["human_label"].sum() == 0:
            continue  # no positive label for this resume in the pair set
        total += 1
        top_jd = group.loc[group["predicted_score"].idxmax(), "jd_id"]
        true_jd = group.loc[group["human_label"] == 1, "jd_id"].iloc[0]
        if top_jd == true_jd:
            correct += 1
    return correct / total if total else float("nan")


def run_evaluation():
    resumes, jds, labeled = load_data()
    matcher = build_and_save_default_matcher()
    scored = score_all_pairs(matcher, resumes, jds, labeled)

    corr, p_value = pointbiserialr(scored["human_label"], scored["predicted_score"])
    rank_acc = ranking_accuracy(scored)

    acc = accuracy_score(scored["human_label"], scored["predicted_label"])
    prec = precision_score(scored["human_label"], scored["predicted_label"], zero_division=0)
    rec = recall_score(scored["human_label"], scored["predicted_label"], zero_division=0)
    f1 = f1_score(scored["human_label"], scored["predicted_label"], zero_division=0)

    lines = [
        "Resume-JD Matcher Evaluation",
        "=============================",
        f"Labeled pairs: {len(scored)}",
        f"Point-biserial correlation (score vs. human label): {corr:.3f} (p={p_value:.4f})",
        f"Top-1 ranking accuracy (correct JD ranked highest per resume): {rank_acc*100:.1f}%",
        "",
        f"At threshold={THRESHOLD}%:",
        f"  Accuracy:  {acc:.3f}",
        f"  Precision: {prec:.3f}",
        f"  Recall:    {rec:.3f}",
        f"  F1:        {f1:.3f}",
        "",
        "Per-pair detail:",
        scored.to_string(index=False),
    ]
    report = "\n".join(lines)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation_report.txt").write_text(report)
    print(report)
    return scored


if __name__ == "__main__":
    run_evaluation()
