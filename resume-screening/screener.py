"""AI Resume Screening: text extraction, NLP preprocessing, TF-IDF matching and ranking."""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_stemmer = PorterStemmer()

# Skill vocabulary used for keyword matching (lower-case; multi-word skills allowed).
SKILLS = sorted({
    # data / ML
    "python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost",
    "machine learning", "deep learning", "nlp", "computer vision", "statistics", "a/b testing",
    "data visualization", "matplotlib", "seaborn", "power bi", "tableau", "excel", "spacy", "nltk",
    "transformers", "opencv", "regression", "classification", "clustering", "forecasting",
    # data engineering / cloud
    "apache spark", "spark", "airflow", "kafka", "snowflake", "etl", "data warehousing", "aws", "azure",
    "gcp", "docker", "kubernetes", "linux", "git", "postgresql", "mysql", "mongodb",
    # web
    "javascript", "typescript", "react", "next.js", "angular", "node.js", "html", "css", "tailwind css",
    "redux", "rest apis", "graphql", "jest", "django", "flask", "fastapi", "java", "spring boot",
    "microservices", "figma", "responsive design", "agile",
    # business / soft
    "communication", "leadership", "seo", "google ads", "google analytics", "accounting", "ifrs",
    "recruitment", "research",
})

# A few aliases so that "sklearn" counts as "scikit-learn" etc.
ALIASES = {"sklearn": "scikit-learn", "postgres": "postgresql", "nextjs": "next.js", "nodejs": "node.js",
           "tailwind": "tailwind css", "rest api": "rest apis", "ml": "machine learning"}


# ---------------------------------------------------------------- text extraction
def extract_text(file_name: str, data: bytes) -> str:
    """Return plain text from PDF or TXT bytes."""
    if file_name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def load_folder(folder: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in {".pdf", ".txt"}:
            out.append({"file": p.name, "text": extract_text(p.name, p.read_bytes())})
    return out


# ---------------------------------------------------------------- preprocessing
def preprocess(text: str) -> str:
    """Lower-case → remove punctuation/special chars → tokenize → remove stopwords → stem."""
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    text = re.sub(r"http\S+", " ", text)            # urls
    text = re.sub(r"[^a-z\s]", " ", text)           # punctuation, digits, special chars
    tokens = text.split()                            # tokenization
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(_stemmer.stem(t) for t in tokens)  # stemming


# ---------------------------------------------------------------- information extraction
def _norm(text: str) -> str:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    for a, canon in ALIASES.items():
        t = re.sub(rf"(?<![\w.]){re.escape(a)}(?![\w])", canon, t)
    return t


def extract_skills(text: str) -> set[str]:
    t = _norm(text)
    found = set()
    for s in SKILLS:
        if re.search(rf"(?<![\w]){re.escape(s)}(?![\w-])", t):
            found.add(s)
    if "apache spark" in found:
        found.discard("spark")
    return found


def extract_name(text: str, fallback: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip()
        if line and "@" not in line and len(line.split()) <= 4 and re.fullmatch(r"[A-Za-zÀ-ž .'-]+", line):
            return line.title()
    return Path(fallback).stem.replace("_", " ").title()


def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    return m.group(0) if m else ""


def extract_years(text: str) -> float:
    """Largest 'N years of experience' mention, else sum of (start - end) year ranges."""
    m = [int(x) for x in re.findall(r"(\d{1,2})\+?\s*years?", text, flags=re.I)]
    if m:
        return float(max(m))
    total = 0
    for a, b in re.findall(r"(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|19\d{2}|present)", text, flags=re.I):
        end = 2026 if b.lower() == "present" else int(b)
        total += max(0, end - int(a))
    return float(total)


# ---------------------------------------------------------------- matching + ranking
def rank_resumes(job_description: str, resumes: list[dict], method: str = "tfidf",
                 w_similarity: float = 0.6, threshold: float = 50.0) -> pd.DataFrame:
    """Score every resume against the JD and return a ranked DataFrame.

    match % = w_similarity * text similarity  +  (1 - w_similarity) * skill coverage
    Text similarity is cosine similarity of TF-IDF (or Bag-of-Words) vectors, rescaled so the
    best resume in the batch is 100. Skill coverage = share of JD skills found in the resume.
    """
    docs = [preprocess(job_description)] + [preprocess(r["text"]) for r in resumes]
    vec = (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True) if method == "tfidf"
           else CountVectorizer(ngram_range=(1, 2)))
    matrix = vec.fit_transform(docs)
    raw_sim = cosine_similarity(matrix[0], matrix[1:]).ravel()
    top = raw_sim.max() if raw_sim.max() > 0 else 1.0

    jd_skills = extract_skills(job_description)
    rows = []
    for r, sim in zip(resumes, raw_sim):
        skills = extract_skills(r["text"])
        matched = sorted(jd_skills & skills)
        missing = sorted(jd_skills - skills)
        coverage = len(matched) / len(jd_skills) if jd_skills else 0.0
        score = 100 * (w_similarity * (sim / top) + (1 - w_similarity) * coverage)
        rows.append({
            "Candidate": extract_name(r["text"], r["file"]),
            "Email": extract_email(r["text"]),
            "File": r["file"],
            "Match %": round(score, 1),
            "Cosine similarity": round(float(sim), 3),
            "Skill coverage %": round(100 * coverage, 1),
            "Years exp.": extract_years(r["text"]),
            "Matched skills": ", ".join(matched),
            "Missing skills": ", ".join(missing),
        })
    df = pd.DataFrame(rows).sort_values("Match %", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Status"] = ["✅ Shortlisted" if s >= threshold else "❌ Rejected" for s in df["Match %"]]
    return df
