import re
import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()

    raw_text = "\n".join(text_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", raw_text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


SECTION_PATTERNS = [
    ("Education", r"(?i)\b(education|academic|qualification|degree)\b"),
    ("Experience", r"(?i)\b(experience|employment|work\s*history|professional\s*background)\b"),
    ("Skills", r"(?i)\b(skills|technical\s*skills|core\s*competencies|technologies)\b"),
    ("Projects", r"(?i)\b(projects|personal\s*projects|academic\s*projects)\b"),
    ("Certifications", r"(?i)\b(certification|certifications|licenses|accreditation)\b"),
    ("Summary", r"(?i)\b(summary|objective|profile|about\s*me)\b"),
    ("Contact", r"(?i)\b(contact|email|phone|address)\b"),
]


def extract_sections(text: str) -> dict[str, bool]:
    """Detect which standard resume sections are present in the text."""
    found = {}
    for section_name, pattern in SECTION_PATTERNS:
        found[section_name] = bool(re.search(pattern, text))
    return found
