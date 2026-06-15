import re
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


@dataclass
class ATSResult:
    overall_score: float
    keyword_score: float
    semantic_score: float
    skills_coverage: float
    missing_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    """Lowercase, split into words, remove stopwords and short tokens."""
    words = re.findall(r"[a-z][a-z0-9+#.\-]{1,}", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _keyword_score(resume_text: str, job_description: str) -> float:
    """TF-IDF cosine similarity between resume and JD (0-100)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)


def _semantic_score(resume_text: str, job_description: str) -> float:
    """Sentence-transformer embedding similarity (0-100).
    Captures meaning even when exact keywords differ."""
    model = _get_semantic_model()

    resume_chunks = _split_into_chunks(resume_text, max_words=200)
    jd_chunks = _split_into_chunks(job_description, max_words=200)

    resume_embeddings = model.encode(resume_chunks)
    jd_embeddings = model.encode(jd_chunks)

    sim_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
    avg_max_sim = float(sim_matrix.max(axis=0).mean())
    return round(avg_max_sim * 100, 2)


def _split_into_chunks(text: str, max_words: int = 200) -> list[str]:
    """Split text into overlapping chunks for better embedding coverage."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words // 2
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _skills_coverage(resume_text: str, job_description: str, top_n: int = 20):
    """What percentage of important JD keywords appear in the resume.
    Also returns matched and missing keyword lists."""
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([job_description])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = vectorizer.transform([job_description]).toarray()[0]

    keyword_scores = {
        feature_names[i]: tfidf_scores[i]
        for i in range(len(feature_names))
        if tfidf_scores[i] > 0 and len(feature_names[i]) > 2
    }

    resume_tokens = _tokenize(resume_text)

    matched = []
    missing = []
    for kw, score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True):
        if kw in resume_tokens:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(keyword_scores)
    coverage = (len(matched) / total * 100) if total > 0 else 0

    return round(coverage, 2), matched[:top_n], missing[:top_n]


WEIGHTS = {
    "keyword": 0.30,
    "semantic": 0.45,
    "skills_coverage": 0.25,
}

def calculate_ats_score(resume_text: str, job_description: str) -> ATSResult:
    """Multi-dimensional ATS scoring combining keyword, semanti, and skills metrics."""
    kw_score = _keyword_score(resume_text, job_description)
    sem_score = _semantic_score(resume_text, job_description)
    coverage, matched, missing = _skills_coverage(resume_text, job_description)

    overall = (
        WEIGHTS["keyword"] * kw_score
        + WEIGHTS["semantic"] * sem_score
        + WEIGHTS["skills_coverage"] * coverage
    )

    return ATSResult(
        overall_score=round(overall, 1),
        keyword_score=kw_score,
        semantic_score=sem_score,
        skills_coverage=coverage,
        matched_keywords=matched,
        missing_keywords=missing,
    )
