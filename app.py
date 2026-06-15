import os
from pathlib import Path

import streamlit as st

from resume_parser import extract_text_from_pdf, extract_sections
from ats import calculate_ats_score
from llm_utils import analyze_resume, generate_tailored_bullets, RateLimitExhausted

# Load .env file if it exists
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .score-high { color: #0f5132; background: #d1e7dd; padding: 8px 16px; border-radius: 8px; font-size: 2rem; font-weight: 700; text-align: center; }
    .score-med  { color: #664d03; background: #fff3cd; padding: 8px 16px; border-radius: 8px; font-size: 2rem; font-weight: 700; text-align: center; }
    .score-low  { color: #842029; background: #f8d7da; padding: 8px 16px; border-radius: 8px; font-size: 2rem; font-weight: 700; text-align: center; }
    .skill-chip {
        display: inline-block; padding: 4px 12px; margin: 4px;
        background: #fee2e2; color: #991b1b; border-radius: 16px;
        font-size: 0.85rem; font-weight: 500;
    }
    .section-present { display: inline-block; padding: 4px 12px; margin: 4px; background: #d1e7dd; color: #0f5132; border-radius: 16px; font-size: 0.85rem; }
    .section-missing { display: inline-block; padding: 4px 12px; margin: 4px; background: #f8d7da; color: #842029; border-radius: 16px; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and paste a job description to get an **ATS score**, **missing skills**, and **AI-powered suggestions**.")
st.divider()

# ---------------------------------------------------------------------------
# Fallback analysis when API is rate-limited
# ---------------------------------------------------------------------------
def _show_fallback_analysis(ats_result, sections):
    """Rule-based analysis shown when LLM API is unavailable."""

    st.markdown("## Key Weaknesses (rule-based)")
    issues = []
    if ats_result.keyword_score < 30:
        issues.append("- Very low keyword overlap with the job description. Your resume uses different terminology than the JD.")
    if ats_result.skills_coverage < 40:
        issues.append(f"- Only {ats_result.skills_coverage}% of important JD keywords found in your resume.")
    if not sections.get("Summary", False):
        issues.append("- Missing a **Professional Summary** section -- most ATS systems look for this.")
    if not sections.get("Skills", False):
        issues.append("- Missing a dedicated **Skills** section -- add one with keywords from the JD.")
    if not sections.get("Certifications", False):
        issues.append("- No **Certifications** section detected -- add relevant certifications if you have them.")
    if not issues:
        issues.append("- No major structural issues detected.")
    st.markdown("\n".join(issues))

    st.markdown("## Missing Technical Skills")
    if ats_result.missing_keywords:
        st.markdown("\n".join(f"- {kw}" for kw in ats_result.missing_keywords))
    else:
        st.markdown("- No major gaps detected.")

    st.markdown("## Suggested Improvements")
    tips = [
        f"- Add these missing keywords naturally into your experience bullets: **{', '.join(ats_result.missing_keywords[:5])}**" if ats_result.missing_keywords else None,
        "- Use action verbs like *Developed*, *Implemented*, *Architected*, *Optimized*, *Led* to start bullet points.",
        "- Quantify achievements with numbers (e.g., \"reduced load time by 40%\", \"served 10K+ users\").",
        "- Mirror the exact phrasing from the job description where it genuinely applies to your experience.",
        "- Keep your resume to 1-2 pages with the most relevant experience first.",
    ]
    st.markdown("\n".join(t for t in tips if t))

    if ats_result.semantic_score > ats_result.keyword_score + 15:
        st.info(
            "Your resume's **meaning** aligns better than its **keywords** "
            f"(semantic {ats_result.semantic_score}% vs keyword {ats_result.keyword_score}%). "
            "You have relevant experience but need to use the exact terms from the job description."
        )


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📤 Upload & Input")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Paste the job description",
        height=300,
        placeholder="Paste the full job description here...",
    )
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if analyze_btn:
    # --- Validation ---
    if not uploaded_file:
        st.error("Please upload a PDF resume.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    # --- Extract resume text ---
    with st.spinner("Parsing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        sections = extract_sections(resume_text)

    if not resume_text.strip():
        st.error("Could not extract text from the PDF. Make sure it is not a scanned image.")
        st.stop()

    # --- ATS Score ---
    with st.spinner("Calculating ATS score (keyword + semantic + skills analysis)..."):
        ats = calculate_ats_score(resume_text, job_description)

    # --- Display Overall Score ---
    st.subheader("📊 ATS Score")
    col_score, col_bar = st.columns([1, 3])

    with col_score:
        if ats.overall_score >= 70:
            css_class = "score-high"
        elif ats.overall_score >= 40:
            css_class = "score-med"
        else:
            css_class = "score-low"
        st.markdown(f'<div class="{css_class}">{ats.overall_score}%</div>', unsafe_allow_html=True)

    with col_bar:
        st.progress(min(ats.overall_score / 100, 1.0))
        if ats.overall_score >= 70:
            st.success("Great match! Your resume aligns well with this job description.")
        elif ats.overall_score >= 40:
            st.warning("Moderate match. Consider adding more relevant keywords and aligning experience.")
        else:
            st.error("Low match. Significant improvements needed to pass ATS filters.")

    # --- Score Breakdown ---
    st.subheader("📈 Score Breakdown")
    col_kw, col_sem, col_cov = st.columns(3)

    with col_kw:
        st.metric("Keyword Match", f"{ats.keyword_score}%")
        st.caption("TF-IDF exact keyword overlap")
        st.progress(min(ats.keyword_score / 100, 1.0))

    with col_sem:
        st.metric("Semantic Similarity", f"{ats.semantic_score}%")
        st.caption("Meaning-level alignment (even without exact keywords)")
        st.progress(min(ats.semantic_score / 100, 1.0))

    with col_cov:
        st.metric("Skills Coverage", f"{ats.skills_coverage}%")
        st.caption(f"{len(ats.matched_keywords)} matched / {len(ats.matched_keywords) + len(ats.missing_keywords)} total JD keywords")
        st.progress(min(ats.skills_coverage / 100, 1.0))

    st.divider()

    # --- Missing & Matched Skills ---
    col_missing, col_matched = st.columns(2)

    with col_missing:
        st.subheader("❌ Missing Skills")
        if ats.missing_keywords:
            chips_html = "".join(
                f'<span class="skill-chip">{skill}</span>' for skill in ats.missing_keywords
            )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.success("No major missing keywords detected!")

    with col_matched:
        st.subheader("✅ Matched Skills")
        if ats.matched_keywords:
            matched_html = "".join(
                f'<span class="section-present">{skill}</span>' for skill in ats.matched_keywords
            )
            st.markdown(matched_html, unsafe_allow_html=True)
        else:
            st.warning("No keyword matches found.")

    st.divider()

    # --- Resume Sections ---
    st.subheader("📑 Resume Sections Detected")
    section_html = ""
    for name, present in sections.items():
        cls = "section-present" if present else "section-missing"
        icon = "✓" if present else "✗"
        section_html += f'<span class="{cls}">{icon} {name}</span>'
    st.markdown(section_html, unsafe_allow_html=True)

    st.divider()

    # --- LLM Suggestions ---
    st.subheader("💡 AI-Powered Analysis")
    analysis = None
    bullets = None

    try:
        with st.spinner("Getting AI suggestions (this may take a moment)..."):
            analysis = analyze_resume(resume_text, job_description)

        st.markdown(analysis)

        st.divider()

        st.subheader("✨ Enhanced Experience Bullets (STAR Method)")
        with st.spinner("Generating tailored bullet points..."):
            bullets = generate_tailored_bullets(resume_text, job_description)
        st.markdown(bullets)

    except ValueError as e:
        st.error(str(e))
    except RateLimitExhausted:
        st.warning("API rate limit reached on all models. Showing rule-based analysis instead.")
        _show_fallback_analysis(ats, sections)
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            st.warning("API rate limit reached. Showing rule-based analysis instead.")
            _show_fallback_analysis(ats, sections)
        else:
            st.error(f"LLM analysis failed: {e}")

    st.divider()

    # --- Export ---
    report_lines = [
        "AI RESUME ANALYZER — REPORT",
        "=" * 40,
        f"\nOverall ATS Score: {ats.overall_score}%",
        f"  Keyword Match:       {ats.keyword_score}%",
        f"  Semantic Similarity: {ats.semantic_score}%",
        f"  Skills Coverage:     {ats.skills_coverage}%\n",
        "Matched Skills: " + ", ".join(ats.matched_keywords) if ats.matched_keywords else "Matched Skills: None",
        "Missing Skills: " + ", ".join(ats.missing_keywords) if ats.missing_keywords else "Missing Skills: None",
        "\n--- AI Analysis ---\n",
        analysis if "analysis" in dir() else "N/A",
        "\n--- Enhanced Bullets ---\n",
        bullets if "bullets" in dir() else "N/A",
    ]
    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
elif not analyze_btn:
    # Landing state
    st.info("👈 Upload a PDF resume and paste a job description in the sidebar, then click **Analyze**.")
