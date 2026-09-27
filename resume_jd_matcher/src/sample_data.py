"""
Generates a small synthetic dataset: 5 resumes, 4 job descriptions, and the
5x4=20 manually labeled resume/JD pairs used for evaluation.

Labels are 1 if the resume's role genuinely matches the JD's role, else 0.
This is deliberately synthetic/hand-built rather than scraped, so there's no
copyright or real-person-data concern, and the "ground truth" labels are
unambiguous (each resume was written to match exactly one JD's role).

Run directly to (re)generate data/resumes.csv, data/job_descriptions.csv, and
data/labeled_pairs.csv:
    python src/sample_data.py
"""
import os
import csv

RESUMES = {
    "R1_data_scientist": """
        Data Scientist with 4 years of experience building machine learning models
        for churn prediction and demand forecasting. Proficient in Python, pandas,
        scikit-learn, and TensorFlow. Built and deployed regression and classification
        models into production using Flask and Docker. Strong background in statistics,
        A/B testing, and SQL for data extraction. Experience with feature engineering,
        cross-validation, and hyperparameter tuning. Familiar with Jupyter notebooks,
        matplotlib and seaborn for data visualization. Collaborated with product teams
        to translate business questions into predictive modeling problems.
    """,
    "R2_backend_engineer": """
        Backend Software Engineer with 5 years building scalable REST APIs using
        Python, Django, and FastAPI. Deep experience with PostgreSQL, Redis, and
        message queues like RabbitMQ and Kafka. Designed microservices architecture
        handling millions of requests per day. Strong understanding of database
        indexing, query optimization, and caching strategies. Experience with unit
        testing, CI/CD pipelines using Jenkins, and containerization with Docker and
        Kubernetes. Comfortable working in an agile environment with code reviews.
    """,
    "R3_frontend_engineer": """
        Frontend Engineer with 3 years of experience building responsive web
        applications using React, TypeScript, and Redux. Skilled in HTML5, CSS3,
        and modern JavaScript (ES6+). Built reusable component libraries and
        implemented state management with Redux Toolkit. Experience with webpack,
        Jest for unit testing, and Cypress for end-to-end testing. Worked closely
        with UX designers to implement pixel-perfect, accessible interfaces.
        Familiar with REST API integration and basic Node.js for tooling scripts.
    """,
    "R4_devops_engineer": """
        DevOps Engineer with 6 years managing cloud infrastructure on AWS and GCP.
        Expert in Terraform for infrastructure as code, Kubernetes for container
        orchestration, and Docker for containerization. Built CI/CD pipelines using
        GitHub Actions and Jenkins. Strong background in monitoring and observability
        using Prometheus, Grafana, and the ELK stack. Experience automating deployments,
        managing secrets with Vault, and reducing infrastructure costs through
        autoscaling and rightsizing. Comfortable with bash and Python scripting.
    """,
    "R5_product_manager": """
        Product Manager with 5 years leading cross-functional teams to ship consumer
        mobile and web products. Skilled in writing product requirement documents,
        prioritizing roadmaps, and running user research and A/B tests. Experience
        working closely with engineering and design teams using agile/scrum
        methodology. Strong stakeholder communication skills, data-driven
        decision-making using SQL and analytics dashboards, and experience
        launching features from concept to GA across iOS, Android, and web.
    """,
}

JOB_DESCRIPTIONS = {
    "JD1_data_scientist": """
        We are looking for a Data Scientist to join our analytics team. Responsibilities
        include building machine learning models for prediction and classification
        problems, performing statistical analysis, and running A/B tests to evaluate
        product changes. Required skills: strong Python programming, experience with
        pandas, scikit-learn, and either TensorFlow or PyTorch. SQL proficiency for
        querying large datasets is required. Experience deploying models to production
        (Flask, Docker) is a plus. Strong communication skills to present findings to
        non-technical stakeholders.
    """,
    "JD2_backend_developer": """
        We are hiring a Backend Developer to design and build scalable REST APIs and
        microservices. Required: strong Python experience with Django or FastAPI,
        solid understanding of relational databases (PostgreSQL), caching (Redis), and
        message queues (Kafka or RabbitMQ). Experience with Docker and Kubernetes for
        deployment. Familiarity with CI/CD pipelines and automated testing. Should be
        comfortable optimizing database queries and designing for high availability
        and scalability.
    """,
    "JD3_frontend_developer": """
        Seeking a Frontend Developer to build modern, responsive user interfaces.
        Required: strong experience with React and TypeScript, solid CSS/HTML skills,
        and familiarity with state management (Redux or Context API). Experience
        writing unit tests with Jest and end-to-end tests with Cypress is a plus.
        Should be comfortable collaborating with designers to implement accessible,
        pixel-perfect UI, and integrating with REST or GraphQL APIs.
    """,
    "JD4_devops_engineer": """
        We need a DevOps Engineer to manage and scale our cloud infrastructure on AWS.
        Required: experience with Terraform for infrastructure as code, Kubernetes and
        Docker for container orchestration, and building CI/CD pipelines (GitHub
        Actions or Jenkins). Experience with monitoring tools like Prometheus and
        Grafana required. Strong scripting skills in Python or bash. Experience
        reducing cloud costs and improving deployment reliability is a big plus.
    """,
}

# Ground-truth labels: 1 if the resume's true role matches the JD's role, else 0.
# 5 resumes x 4 JDs = 20 pairs total.
_ROLE_MATCH = {
    "R1_data_scientist": "JD1_data_scientist",
    "R2_backend_engineer": "JD2_backend_developer",
    "R3_frontend_engineer": "JD3_frontend_developer",
    "R4_devops_engineer": "JD4_devops_engineer",
    # R5 (product manager) has no matching JD in this small JD set — all 0s,
    # which is realistic: a PM resume shouldn't score high against any of these.
}


def generate_labeled_pairs():
    pairs = []
    for resume_id in RESUMES:
        for jd_id in JOB_DESCRIPTIONS:
            label = 1 if _ROLE_MATCH.get(resume_id) == jd_id else 0
            pairs.append({"resume_id": resume_id, "jd_id": jd_id, "label": label})
    return pairs


def save_csvs(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "resumes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_id", "text"])
        for rid, text in RESUMES.items():
            writer.writerow([rid, " ".join(text.split())])

    with open(os.path.join(output_dir, "job_descriptions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["jd_id", "text"])
        for jid, text in JOB_DESCRIPTIONS.items():
            writer.writerow([jid, " ".join(text.split())])

    pairs = generate_labeled_pairs()
    with open(os.path.join(output_dir, "labeled_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "jd_id", "label"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Wrote {len(RESUMES)} resumes, {len(JOB_DESCRIPTIONS)} job descriptions, "
          f"and {len(pairs)} labeled pairs to '{output_dir}/'.")


if __name__ == "__main__":
    save_csvs()
