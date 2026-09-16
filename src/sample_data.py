"""
sample_data.py
--------------
Creates a small synthetic dataset of resumes, job descriptions, and
manually labeled resume-JD relevance pairs, used to demo and evaluate
the matcher. All text is hand-written for this project (not scraped).

Run:
    python src/sample_data.py
Output:
    data/resumes.csv
    data/job_descriptions.csv
    data/labeled_pairs.csv
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"

RESUMES = [
    {
        "resume_id": "R1",
        "candidate": "Aditi Sharma",
        "text": """
            Computer Science student with experience in Python, Java, and Android
            development. Built a resume-matching tool using TF-IDF and cosine
            similarity in scikit-learn. Familiar with Pandas, NumPy, Git, and
            Streamlit for building interactive dashboards. Completed coursework
            in Data Structures, Algorithms, Database Management Systems, and
            Machine Learning. Solved 300+ problems on LeetCode and Codeforces.
        """,
    },
    {
        "resume_id": "R2",
        "candidate": "Rohan Verma",
        "text": """
            Backend developer with 2 years of experience in Java, Spring Boot,
            and REST API design. Worked extensively with MySQL and PostgreSQL,
            Docker, and CI/CD pipelines using Jenkins. Some exposure to AWS
            (EC2, S3). Strong in object-oriented design and unit testing with
            JUnit.
        """,
    },
    {
        "resume_id": "R3",
        "candidate": "Priya Nair",
        "text": """
            Data analyst skilled in Python, Pandas, NumPy, and SQL. Built
            dashboards in Tableau and Power BI. Experience with A/B testing,
            statistical analysis, and communicating insights to non-technical
            stakeholders. Basic familiarity with scikit-learn for regression
            and classification tasks.
        """,
    },
    {
        "resume_id": "R4",
        "candidate": "Karan Mehta",
        "text": """
            Frontend engineer specializing in React, JavaScript, TypeScript,
            HTML, and CSS. Built responsive web applications and worked with
            REST and GraphQL APIs. Familiar with Figma for design handoff and
            Jest for testing. No formal machine learning background.
        """,
    },
    {
        "resume_id": "R5",
        "candidate": "Sneha Iyer",
        "text": """
            Machine learning engineer with experience in Python, scikit-learn,
            TensorFlow, and NLP techniques including TF-IDF, word embeddings,
            and transformer models. Built text classification and semantic
            similarity systems. Comfortable deploying models with Flask and
            Streamlit, and working with Git and Docker.
        """,
    },
]

JOB_DESCRIPTIONS = [
    {
        "jd_id": "J1",
        "title": "Machine Learning Intern (NLP)",
        "text": """
            We are looking for a Machine Learning Intern with strong Python
            skills and hands-on experience with scikit-learn. Experience with
            NLP techniques such as TF-IDF, text vectorization, and cosine
            similarity is a big plus. Familiarity with Pandas, NumPy, and
            building simple Streamlit demos is desired. You will help build
            and evaluate text-matching and classification models.
        """,
    },
    {
        "jd_id": "J2",
        "title": "Backend Developer (Java)",
        "text": """
            Seeking a Backend Developer proficient in Java and Spring Boot to
            design and maintain REST APIs. Experience with relational
            databases (MySQL/PostgreSQL), Docker, and CI/CD pipelines
            required. Knowledge of AWS and unit testing frameworks like JUnit
            is a plus.
        """,
    },
    {
        "jd_id": "J3",
        "title": "Data Analyst",
        "text": """
            Looking for a Data Analyst comfortable with SQL, Python, and
            Pandas to build reporting dashboards and support data-driven
            decision-making. Experience with Tableau or Power BI and basic
            statistics (A/B testing, hypothesis testing) preferred.
        """,
    },
    {
        "jd_id": "J4",
        "title": "Frontend Engineer",
        "text": """
            Hiring a Frontend Engineer skilled in React, JavaScript, and CSS
            to build responsive, accessible web interfaces. Experience with
            REST/GraphQL API integration and testing frameworks like Jest is
            required. Design collaboration experience with Figma is a plus.
        """,
    },
]

# Manually labeled relevance pairs (0 = not relevant, 1 = strong match).
# In a real project these would be assigned by a human reviewer; here
# they're hand-labeled to reflect an obviously correct ranking, used to
# sanity-check that the TF-IDF matcher's scores agree with human judgment.
LABELED_PAIRS = [
    ("R1", "J1", 1),  # CS student w/ TF-IDF project -> ML/NLP internship: strong match
    ("R1", "J2", 0),  # some Java overlap but not backend-focused
    ("R1", "J3", 0),
    ("R1", "J4", 0),
    ("R2", "J2", 1),  # backend dev -> backend role: strong match
    ("R2", "J1", 0),
    ("R2", "J3", 0),
    ("R2", "J4", 0),
    ("R3", "J3", 1),  # data analyst -> data analyst role: strong match
    ("R3", "J1", 0),
    ("R3", "J2", 0),
    ("R3", "J4", 0),
    ("R4", "J4", 1),  # frontend engineer -> frontend role: strong match
    ("R4", "J1", 0),
    ("R4", "J2", 0),
    ("R4", "J3", 0),
    ("R5", "J1", 1),  # ML engineer w/ NLP -> ML/NLP internship: strong match
    ("R5", "J2", 0),
    ("R5", "J3", 0),
    ("R5", "J4", 0),
]


def build():
    DATA_DIR.mkdir(exist_ok=True)

    resumes_df = pd.DataFrame(RESUMES)
    resumes_df["text"] = resumes_df["text"].str.strip()
    resumes_df.to_csv(DATA_DIR / "resumes.csv", index=False)

    jd_df = pd.DataFrame(JOB_DESCRIPTIONS)
    jd_df["text"] = jd_df["text"].str.strip()
    jd_df.to_csv(DATA_DIR / "job_descriptions.csv", index=False)

    labeled_df = pd.DataFrame(LABELED_PAIRS, columns=["resume_id", "jd_id", "human_label"])
    labeled_df.to_csv(DATA_DIR / "labeled_pairs.csv", index=False)

    print(f"Wrote {len(resumes_df)} resumes, {len(jd_df)} job descriptions, "
          f"{len(labeled_df)} labeled pairs -> {DATA_DIR}")


if __name__ == "__main__":
    build()
