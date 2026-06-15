# 📄 AI Resume Analyzer with ATS Scoring

A Streamlit-based web application that analyzes resumes against job descriptions using NLP and LLM techniques. Get actionable feedback to improve your resume's ATS compatibility.

## Features

- **ATS Score** — Multi-dimensional scoring combining TF-IDF keyword matching, semantic similarity (sentence-transformers), and skills coverage analysis
- **Missing Skills Detection** — Identifies important JD keywords absent from your resume, ranked by TF-IDF relevance
- **AI-Powered Analysis** — Groq LLM-powered weaknesses, improvements, and a tailored professional summary
- **STAR Bullet Rewriting** — Experience bullets rewritten using the STAR method (Situation, Task, Action, Result)
- **Resume Section Detection** — Checks for standard sections (Education, Experience, Skills, Projects, Certifications, Summary, Contact)
- **Export** — Download the full analysis as a text report
- **Fallback Analysis** — Rule-based feedback when API rate limits are hit

## Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | PDF text extraction |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorization & cosine similarity |
| [sentence-transformers](https://www.sbert.net/) | Semantic similarity (all-MiniLM-L6-v2 model) |
| [NLTK](https://www.nltk.org/) | Stopword removal |
| [Groq](https://groq.com/) | LLM API (Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B) |

## Project Structure

```
Ai_resume_analyzer/
├── app.py               # Streamlit UI and orchestration
├── resume_parser.py     # PDF text extraction and section detection
├── ats.py               # TF-IDF + semantic scoring and missing keyword detection
├── llm_utils.py         # Groq API integration with model fallback
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your API key (not committed to git)
└── sample_resume.pdf    # Sample resume for testing
```

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository

```bash
git clone <repository-url>
cd Ai_resume_analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -m nltk.downloader stopwords
```

### 5. Configure your API key

Copy the example env file and add your Groq API key:

```bash
copy .env.example .env
```

Edit `.env` and replace the placeholder:

```
GROQ_API_KEY=your_actual_groq_api_key
```

Alternatively, set it as an environment variable:

**Windows (cmd):**
```cmd
set GROQ_API_KEY=your_key_here
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "your_key_here"
```

**Linux / macOS:**
```bash
export GROQ_API_KEY=your_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. Upload a PDF resume using the sidebar file uploader
2. Paste the target job description into the text area
3. Click **Analyze**
4. Review your results:
   - Overall ATS score with keyword, semantic, and skills breakdown
   - Missing and matched skills chips
   - Detected resume sections
   - AI-generated weaknesses, suggestions, and tailored summary
   - STAR-method enhanced experience bullets
5. Download the full report as a text file

## How Scoring Works

The overall ATS score is a weighted combination of three metrics:

| Metric | Weight | Method |
|---|---|---|
| Keyword Match | 30% | TF-IDF cosine similarity between resume and JD |
| Semantic Similarity | 45% | Sentence-transformer embedding similarity (captures meaning beyond exact keywords) |
| Skills Coverage | 25% | Percentage of top JD keywords found in resume |

## LLM Model Fallback

The app uses Groq's API with automatic model fallback:

1. `llama-3.3-70b-versatile` (primary)
2. `llama-3.1-8b-instant` (fallback)
3. `gemma2-9b-it` (final fallback)

If all models hit rate limits, a rule-based analysis is shown instead.

## License

This project is for educational and personal use.
