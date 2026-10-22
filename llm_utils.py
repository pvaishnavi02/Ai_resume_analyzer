import os
import time

from groq import Groq, RateLimitError, APIStatusError

MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]              

MAX_RETRIES = 2
RETRY_DELAY = 5

ANALYSIS_PROMPT = """You are an expert ATS and HR evaluator.

Analyze the following resume against the job description.

Provide your response in these exact sections with markdown headers:

## Key Weaknesses
(list weaknesses as bullet points)

## Missing Technical Skills
(list missing skills as bullet points)

## Suggested Improvements
(list concrete, actionable improvements as bullet points)

## Tailored Professional Summary
(write a 3-4 sentence professional summary tailored to the job description)

Resume:
{resume_text}

Job Description:
{job_description}"""

BULLET_PROMPT = """You are an expert resume writer who uses the STAR method
(Situation, Task, Action, Result) to craft impactful bullet points.

Rewrite the experience bullet points from this resume to be more ATS-friendly
and results-oriented. Use strong action verbs and quantify achievements where
possible. Output only the improved bullet points grouped by role.

Resume:
{resume_text}

Target Job Description:
{job_description}"""


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
    return Groq(api_key=api_key)


class RateLimitExhausted(Exception):
    """Raised when all model fallbacks are exhausted due to rate limits."""
    pass


def _call_with_fallback(prompt: str) -> str:
    """Try each model in MODELS list; retry on rate limits, fall back to next model."""
    client = _get_client()

    for model in MODELS:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                break  # move to next model
            except APIStatusError as e:
                if e.status_code == 429:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    break
                raise

    raise RateLimitExhausted(
        "All API models are rate-limited. Please wait a few minutes and try again."
    )


def analyze_resume(resume_text: str, job_description: str) -> str:
    """Send resume + JD to Groq and return structured suggestions."""
    prompt = ANALYSIS_PROMPT.format(
        resume_text=resume_text, job_description=job_description
    )
    return _call_with_fallback(prompt)


def generate_tailored_bullets(resume_text: str, job_description: str) -> str:
    """Rewrite experience bullets in STAR format tailored to the JD."""
    prompt = BULLET_PROMPT.format(
        resume_text=resume_text, job_description=job_description
    )
    return _call_with_fallback(prompt)
