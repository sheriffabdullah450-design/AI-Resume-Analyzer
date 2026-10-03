import os
import re
import json
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="Resume",
    layout="wide"
)


# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


# =========================================================
# STEP 1 — EXTRACT RESUME TEXT
# =========================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# STEP 2 — EXTRACT CANDIDATE DETAILS
# =========================================================

def extract_candidate_details(text):

    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone_match = re.search(
        r"\+91[\s-]?[6-9]\d{9}",
        text
    )

    email = email_match.group(0) if email_match else "Not found"

    phone = phone_match.group(0) if phone_match else "Not found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    name = lines[0] if lines else "Not found"

    return name, email, phone


# =========================================================
# STEP 3 — SKILLS
# =========================================================

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "git",
    "github",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "power bi",
    "tableau",
    "excel",
    "streamlit",
    "flask",
    "django",
    "mongodb",
    "aws",
    "azure",
    "docker"
]


def find_skills(text):

    text_lower = text.lower()

    matched = []

    for skill in SKILLS:

        if skill in text_lower:
            matched.append(skill)

    return matched


# =========================================================
# STEP 4 — RESUME SECTIONS
# =========================================================

SECTION_KEYWORDS = {
    "Summary": [
        "summary",
        "profile",
        "objective"
    ],
    "Education": [
        "education",
        "academic"
    ],
    "Skills": [
        "skills",
        "technical skills"
    ],
    "Projects": [
        "projects",
        "project"
    ],
    "Experience": [
        "experience",
        "internship",
        "work experience"
    ],
    "Certifications": [
        "certifications",
        "certification"
    ],
    "Achievements": [
        "achievements",
        "awards"
    ]
}


def analyze_sections(text):

    text_lower = text.lower()

    result = {}

    for section, keywords in SECTION_KEYWORDS.items():

        found = False

        for keyword in keywords:

            if keyword in text_lower:
                found = True
                break

        result[section] = found

    return result


# =========================================================
# STEP 5 — KEYWORD SCORE
# =========================================================

def calculate_keyword_score(text):

    important_keywords = [
        "resume",
        "education",
        "skills",
        "project",
        "experience",
        "certification",
        "achievement",
        "email",
        "phone"
    ]

    text_lower = text.lower()

    matched = 0

    for keyword in important_keywords:

        if keyword in text_lower:
            matched += 1

    return min(
        round((matched / len(important_keywords)) * 100),
        100
    )


# =========================================================
# STEP 6 — SKILL SCORE
# =========================================================

def calculate_skill_score(matched_skills):

    if len(matched_skills) >= 10:
        return 100

    if len(matched_skills) >= 7:
        return 85

    if len(matched_skills) >= 5:
        return 70

    if len(matched_skills) >= 3:
        return 55

    if len(matched_skills) >= 1:
        return 35

    return 0


# =========================================================
# STEP 7 — QUALITY SCORE
# =========================================================

def calculate_quality_score(text):

    score = 0

    if len(text) > 500:
        score += 20

    if len(text) > 1000:
        score += 20

    if len(text) > 1500:
        score += 20

    if len(text) > 2000:
        score += 20

    if len(text.split()) > 250:
        score += 20

    return min(score, 100)


# =========================================================
# STEP 8 — ATS SCORE
# =========================================================

def calculate_ats_score(
    text,
    matched_skills,
    sections
):

    score = 0

    if len(matched_skills) >= 5:
        score += 30
    elif len(matched_skills) >= 3:
        score += 20
    elif len(matched_skills) >= 1:
        score += 10

    section_count = sum(
        1 for value in sections.values()
        if value
    )

    score += min(section_count * 10, 50)

    if "@" in text:
        score += 10

    if re.search(r"\d", text):
        score += 10

    return min(score, 100)


# =========================================================
# STEP 9 — STRUCTURE SCORE
# =========================================================

def calculate_structure_score(sections):

    total = len(sections)

    found = sum(
        1 for value in sections.values()
        if value
    )

    return round(
        (found / total) * 100
    )


# =========================================================
# STEP 10 — FORMATTING SCORE
# =========================================================

def calculate_formatting_score(text):

    score = 0

    lines = text.splitlines()

    if len(lines) >= 10:
        score += 25

    if len(lines) >= 20:
        score += 25

    if len(text) >= 1000:
        score += 25

    if "\n" in text:
        score += 25

    return min(score, 100)


# =========================================================
# STEP 11 — PROJECT SCORE
# =========================================================

def calculate_project_score(text):

    text_lower = text.lower()

    project_words = [
        "project",
        "developed",
        "built",
        "created",
        "implemented",
        "designed"
    ]

    found = sum(
        1
        for word in project_words
        if word in text_lower
    )

    return min(
        found * 15,
        100
    )


# =========================================================
# STEP 12 — JOB MATCH
# =========================================================

def calculate_job_match(
    resume_text,
    job_description
):

    if not job_description.strip():
        return 0, []

    job_keywords = re.findall(
        r"\b[A-Za-z][A-Za-z0-9+#.-]{2,}\b",
        job_description.lower()
    )

    job_keywords = list(
        dict.fromkeys(job_keywords)
    )

    resume_lower = resume_text.lower()

    matched = []

    for keyword in job_keywords:

        if keyword in resume_lower:
            matched.append(keyword)

    if not job_keywords:
        return 0, []

    percentage = round(
        len(matched) /
        len(job_keywords) *
        100
    )

    return min(percentage, 100), matched


# =========================================================
# STEP 13 — OVERALL SCORE
# =========================================================

def calculate_overall_score(
    keyword_score,
    skill_score,
    quality_score,
    ats_score,
    structure_score,
    formatting_score,
    project_score
):

    score = (
        keyword_score * 0.10
        + skill_score * 0.15
        + quality_score * 0.10
        + ats_score * 0.20
        + structure_score * 0.15
        + formatting_score * 0.10
        + project_score * 0.20
    )

    return round(score)


# =========================================================
# STEP 38 — REAL AI REVIEW
# =========================================================

def generate_real_ai_review(
    resume_text,
    job_description
):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
You are an expert resume reviewer.

Analyze the following resume.

RESUME:
{resume_text[:12000]}

JOB DESCRIPTION:
{job_description[:8000]}

Give:

1. Overall review
2. Strengths
3. Weaknesses
4. ATS improvements
5. Skills to improve
6. Project improvements
7. Formatting improvements
8. Final recommendations

Do not invent information.
Use professional and simple English.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 41 — AI RESUME REWRITER
# =========================================================

def generate_ai_resume_rewrite(resume_text):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Rewrite the following resume professionally.

RESUME:
{resume_text[:12000]}

Rules:

- Do not invent information.
- Do not add fake skills.
- Do not add fake experience.
- Do not add fake achievements.
- Improve grammar.
- Improve clarity.
- Make it ATS-friendly.
- Keep the candidate's original information.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 42 — AI SECTION EDITOR
# =========================================================

def generate_ai_section_edit(
    section_name,
    section_text,
    job_description
):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Improve this resume section.

SECTION:
{section_name}

CURRENT CONTENT:
{section_text}

TARGET JOB:
{job_description[:5000]}

Rules:

- Do not invent information.
- Improve grammar.
- Make it professional.
- Make it ATS-friendly.
- Keep the original meaning.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 43 — JOB SPECIFIC OPTIMIZER
# =========================================================

def generate_job_specific_optimizer(
    resume_text,
    job_description
):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Optimize this resume for the target job.

RESUME:
{resume_text[:12000]}

JOB DESCRIPTION:
{job_description[:8000]}

Provide:

1. Matching skills
2. Missing relevant skills
3. Resume keywords
4. Summary improvements
5. Project improvements
6. ATS recommendations

Do not invent candidate information.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 44 — INTERVIEW QUESTIONS
# =========================================================

def generate_ai_interview_questions(
    resume_text,
    job_description
):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Create interview questions for this candidate.

RESUME:
{resume_text[:10000]}

JOB DESCRIPTION:
{job_description[:7000]}

Create:

- 5 HR questions
- 5 technical questions
- 5 project questions
- 5 job-specific questions

Do not invent facts about the candidate.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 45 — MOCK INTERVIEW
# =========================================================

def generate_mock_interview_question(
    resume_text,
    job_description,
    previous_questions
):

    client = get_openai_client()

    if client is None:
        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    previous = "\n".join(previous_questions)

    prompt = f"""
Act as a professional interviewer.

Resume:
{resume_text[:8000]}

Job Description:
{job_description[:6000]}

Previous questions:
{previous}

Ask ONE new interview question.

Do not repeat previous questions.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


def evaluate_mock_interview_answer(
    question,
    answer,
    resume_text,
    job_description
):

    client = get_openai_client()

    if client is None:
        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Evaluate this interview answer.

QUESTION:
{question}

ANSWER:
{answer}

RESUME:
{resume_text[:6000]}

JOB:
{job_description[:5000]}

Give:

1. Strengths
2. Improvements
3. Communication feedback
4. Technical feedback
5. Suggested improved answer

Do not invent experience.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 46 — FINAL MATCH REPORT
# =========================================================

def generate_final_match_report(
    resume_text,
    job_description
):

    client = get_openai_client()

    if client is None:
        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
Create a final resume-to-job match report.

RESUME:
{resume_text[:10000]}

JOB DESCRIPTION:
{job_description[:7000]}

Include:

- Resume strengths
- Job matching skills
- Missing keywords
- Project relevance
- ATS improvements
- Final improvement checklist

Do not invent information.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 47 — PROFESSIONAL RESUME PDF
# =========================================================

def create_professional_resume_pdf(
    name,
    email,
    phone,
    resume_text
):

    file_path = "AI_Professional_Resume.pdf"

    c = canvas.Canvas(
        file_path,
        pagesize=A4
    )

    width, height = A4

    x = 50
    y = height - 50

    c.setFont(
        "Helvetica-Bold",
        18
    )

    c.drawString(
        x,
        y,
        name[:60]
    )

    y -= 25

    c.setFont(
        "Helvetica",
        10
    )

    c.drawString(
        x,
        y,
        f"{email} | {phone}"
    )

    y -= 30

    c.setFont(
        "Helvetica",
        10
    )

    for line in resume_text.splitlines():

        line = line.strip()

        if not line:
            y -= 8
            continue

        if y < 50:

            c.showPage()

            y = height - 50

            c.setFont(
                "Helvetica",
                10
            )

        c.drawString(
            x,
            y,
            line[:110]
        )

        y -= 14

    c.save()

    return file_path


# =========================================================
# STEP 49 — AI RESUME BUILDER
# =========================================================

def generate_ai_resume_builder(
    candidate_name,
    email,
    phone,
    education,
    skills,
    projects,
    experience,
    certifications,
    job_description
):

    client = get_openai_client()

    if client is None:

        return {
            "success": False,
            "message": "OPENAI_API_KEY is not configured."
        }

    prompt = f"""
You are a professional resume writer.

Create a professional fresher resume using ONLY
the information provided below.

CANDIDATE NAME:
{candidate_name}

EMAIL:
{email}

PHONE:
{phone}

EDUCATION:
{education}

SKILLS:
{skills}

PROJECTS:
{projects}

EXPERIENCE:
{experience}

CERTIFICATIONS:
{certifications}

TARGET JOB DESCRIPTION:
{job_description[:8000]}

IMPORTANT RULES:

1. Do not invent information.
2. Do not invent skills.
3. Do not invent projects.
4. Do not invent experience.
5. Do not invent achievements.
6. Do not invent certifications.
7. Do not invent education.
8. Improve only the wording of information provided.
9. Keep it suitable for a fresher.
10. Make it ATS-friendly.
11. Use clear professional English.

Return the resume in this order:

NAME AND CONTACT

PROFESSIONAL SUMMARY

EDUCATION

TECHNICAL SKILLS

PROJECTS

EXPERIENCE

CERTIFICATIONS

ACHIEVEMENTS

Use only sections for which information was provided.
"""

    try:

        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        return {
            "success": True,
            "text": response.output_text
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# STEP 49 — AI RESUME PDF
# =========================================================

def create_ai_builder_pdf(resume_text):

    file_path = "AI_Built_Professional_Resume.pdf"

    c = canvas.Canvas(
        file_path,
        pagesize=A4
    )

    width, height = A4

    x = 50
    y = height - 50

    lines = resume_text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:

            y -= 8

            continue

        if y < 50:

            c.showPage()

            y = height - 50

        upper_line = line.upper()

        if upper_line in [
            "PROFESSIONAL SUMMARY",
            "EDUCATION",
            "TECHNICAL SKILLS",
            "PROJECTS",
            "EXPERIENCE",
            "CERTIFICATIONS",
            "ACHIEVEMENTS"
        ]:

            c.setFont(
                "Helvetica-Bold",
                12
            )

            y -= 8

            c.drawString(
                x,
                y,
                line[:80]
            )

            y -= 16

            c.setFont(
                "Helvetica",
                10
            )

        else:

            words = line.split()

            current_line = ""

            for word in words:

                test_line = (
                    current_line + " " + word
                ).strip()

                if c.stringWidth(
                    test_line,
                    "Helvetica",
                    10
                ) < 500:

                    current_line = test_line

                else:

                    c.drawString(
                        x,
                        y,
                        current_line
                    )

                    y -= 14

                    current_line = word

                    if y < 50:

                        c.showPage()

                        y = height - 50

            if current_line:

                c.drawString(
                    x,
                    y,
                    current_line
                )

                y -= 14

    c.save()

    return file_path


# =========================================================
# FINAL REPORT PDF
# =========================================================

def create_pdf_report(
    name,
    email,
    phone,
    overall_score,
    ats_score,
    skill_score,
    project_score,
    matched_skills,
    missing_skills
):

    file_path = "resume_analysis_report.pdf"

    c = canvas.Canvas(
        file_path,
        pagesize=A4
    )

    width, height = A4

    x = 50
    y = height - 50

    c.setFont(
        "Helvetica-Bold",
        18
    )

    c.drawString(
        x,
        y,
        "AI Resume Analysis Report"
    )

    y -= 35

    c.setFont(
        "Helvetica",
        11
    )

    c.drawString(
        x,
        y,
        f"Name: {name}"
    )

    y -= 20

    c.drawString(
        x,
        y,
        f"Email: {email}"
    )

    y -= 20

    c.drawString(
        x,
        y,
        f"Phone: {phone}"
    )

    y -= 35

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        x,
        y,
        "Resume Scores"
    )

    y -= 25

    c.setFont(
        "Helvetica",
        11
    )

    scores = [
        f"Overall Score: {overall_score}/100",
        f"ATS Score: {ats_score}/100",
        f"Skill Score: {skill_score}/100",
        f"Project Score: {project_score}/100"
    ]

    for score in scores:

        c.drawString(
            x,
            y,
            score
        )

        y -= 20

    y -= 15

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        x,
        y,
        "Matched Skills"
    )

    y -= 20

    c.setFont(
        "Helvetica",
        10
    )

    for skill in matched_skills:

        if y < 50:

            c.showPage()

            y = height - 50

        c.drawString(
            x,
            y,
            "- " + skill
        )

        y -= 16

    y -= 10

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        x,
        y,
        "Missing Skills"
    )

    y -= 20

    c.setFont(
        "Helvetica",
        10
    )

    for skill in missing_skills:

        if y < 50:

            c.showPage()

            y = height - 50

        c.drawString(
            x,
            y,
            "- " + skill
        )

        y -= 16

    c.save()

    return file_path


# =========================================================
# MAIN TITLE
# =========================================================

st.title("AI Resume Analyzer")

st.write(
    "Analyze, improve, optimize and build your resume using AI."
)


# =========================================================
# UPLOAD RESUME
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# =========================================================
# JOB DESCRIPTION
# =========================================================

job_description = st.text_area(
    "Paste Job Description",
    height=200
)


# =========================================================
# MAIN RESUME ANALYSIS
# =========================================================

if uploaded_file is not None:

    resume_text = extract_resume_text(
        uploaded_file
    )

    if not resume_text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

        st.stop()

    # Candidate details

    name, email, phone = extract_candidate_details(
        resume_text
    )

    # Skills

    matched_skills = find_skills(
        resume_text
    )

    missing_skills = [
        skill
        for skill in SKILLS
        if skill not in matched_skills
    ]

    # Sections

    sections = analyze_sections(
        resume_text
    )

    # Scores

    keyword_score = calculate_keyword_score(
        resume_text
    )

    skill_score = calculate_skill_score(
        matched_skills
    )

    quality_score = calculate_quality_score(
        resume_text
    )

    ats_score = calculate_ats_score(
        resume_text,
        matched_skills,
        sections
    )

    structure_score = calculate_structure_score(
        sections
    )

    formatting_score = calculate_formatting_score(
        resume_text
    )

    project_score = calculate_project_score(
        resume_text
    )

    overall_score = calculate_overall_score(
        keyword_score,
        skill_score,
        quality_score,
        ats_score,
        structure_score,
        formatting_score,
        project_score
    )

    job_match_percentage, matched_job_keywords = (
        calculate_job_match(
            resume_text,
            job_description
        )
    )

    # =====================================================
    # BASIC DASHBOARD
    # =====================================================

    st.header("Resume Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Overall Score",
            f"{overall_score}/100"
        )

    with c2:
        st.metric(
            "ATS Score",
            f"{ats_score}/100"
        )

    with c3:
        st.metric(
            "Skill Score",
            f"{skill_score}/100"
        )

    with c4:
        st.metric(
            "Project Score",
            f"{project_score}/100"
        )

    # =====================================================
    # CANDIDATE DETAILS
    # =====================================================

    st.subheader("Candidate Details")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.write("Name")
        st.write(name)

    with d2:
        st.write("Email")
        st.write(email)

    with d3:
        st.write("Phone")
        st.write(phone)

    # =====================================================
    # SKILLS
    # =====================================================

    st.subheader("Skills Analysis")

    s1, s2 = st.columns(2)

    with s1:

        st.write("Matched Skills")

        if matched_skills:

            for skill in matched_skills:
                st.success(skill)

        else:

            st.info(
                "No skills detected."
            )

    with s2:

        st.write("Missing Skills")

        for skill in missing_skills[:15]:
            st.warning(skill)

    # =====================================================
    # SECTION ANALYSIS
    # =====================================================

    st.subheader("Resume Structure Analysis")

    section_df = pd.DataFrame({
        "Section": list(sections.keys()),
        "Present": [
            "Yes" if value else "No"
            for value in sections.values()
        ]
    })

    st.dataframe(
        section_df,
        use_container_width=True
    )

    # =====================================================
    # SCORE ANALYSIS
    # =====================================================

    st.subheader("Score Analysis")

    score_df = pd.DataFrame({
        "Category": [
            "Keyword",
            "Skills",
            "Quality",
            "ATS",
            "Structure",
            "Formatting",
            "Projects"
        ],
        "Score": [
            keyword_score,
            skill_score,
            quality_score,
            ats_score,
            structure_score,
            formatting_score,
            project_score
        ]
    })

    st.bar_chart(
        score_df.set_index("Category")
    )

    # =====================================================
    # JOB MATCH
    # =====================================================

    st.subheader("Job Description Match")

    if job_description.strip():

        st.metric(
            "Job Match",
            f"{job_match_percentage}%"
        )

        st.progress(
            job_match_percentage / 100
        )

        if matched_job_keywords:

            st.write(
                "Matched Job Keywords:"
            )

            st.write(
                ", ".join(
                    matched_job_keywords[:30]
                )
            )

    else:

        st.info(
            "Paste a job description above."
        )

    # =====================================================
    # STEP 38 — AI REVIEW
    # =====================================================

    st.header("Step 38 — Real AI Resume Review")

    if st.button(
        "Generate AI Resume Review",
        key="ai_review_button"
    ):

        with st.spinner(
            "AI is analyzing your resume..."
        ):

            ai_review = generate_real_ai_review(
                resume_text,
                job_description
            )

        if ai_review["success"]:

            st.markdown(
                ai_review["text"]
            )

        else:

            st.error(
                ai_review["message"]
            )

    # =====================================================
    # STEP 41 — AI REWRITER
    # =====================================================

    st.header("Step 41 — AI Resume Rewriter")

    if st.button(
        "Rewrite My Resume with AI",
        key="rewrite_resume_button"
    ):

        with st.spinner(
            "AI is rewriting your resume..."
        ):

            rewrite_result = generate_ai_resume_rewrite(
                resume_text
            )

        if rewrite_result["success"]:

            st.markdown(
                rewrite_result["text"]
            )

            st.download_button(
                "Download Rewritten Resume",
                data=rewrite_result["text"],
                file_name="AI_Rewritten_Resume.txt",
                mime="text/plain",
                key="rewritten_resume_download"
            )

        else:

            st.error(
                rewrite_result["message"]
            )

    # =====================================================
    # STEP 42 — SECTION EDITOR
    # =====================================================

    st.header("Step 42 — AI Resume Section Editor")

    section_name = st.selectbox(
        "Select Section",
        [
            "Professional Summary",
            "Education",
            "Technical Skills",
            "Projects",
            "Experience",
            "Certifications",
            "Achievements"
        ]
    )

    section_text = st.text_area(
        "Enter Section Content",
        height=150
    )

    if st.button(
        "Improve Section with AI",
        key="section_editor_button"
    ):

        with st.spinner(
            "AI is improving your section..."
        ):

            section_result = generate_ai_section_edit(
                section_name,
                section_text,
                job_description
            )

        if section_result["success"]:

            st.markdown(
                section_result["text"]
            )

        else:

            st.error(
                section_result["message"]
            )

    # =====================================================
    # STEP 43 — JOB OPTIMIZER
    # =====================================================

    st.header("Step 43 — AI Job-Specific Resume Optimizer")

    if st.button(
        "Optimize Resume for Job",
        key="job_optimizer_button"
    ):

        with st.spinner(
            "AI is optimizing your resume..."
        ):

            optimizer_result = (
                generate_job_specific_optimizer(
                    resume_text,
                    job_description
                )
            )

        if optimizer_result["success"]:

            st.markdown(
                optimizer_result["text"]
            )

        else:

            st.error(
                optimizer_result["message"]
            )

    # =====================================================
    # STEP 44 — INTERVIEW QUESTIONS
    # =====================================================

    st.header("Step 44 — AI Interview Questions")

    if st.button(
        "Generate Interview Questions",
        key="interview_questions_button"
    ):

        with st.spinner(
            "Generating interview questions..."
        ):

            interview_result = (
                generate_ai_interview_questions(
                    resume_text,
                    job_description
                )
            )

        if interview_result["success"]:

            st.markdown(
                interview_result["text"]
            )

        else:

            st.error(
                interview_result["message"]
            )

    # =====================================================
    # STEP 45 — MOCK INTERVIEW
    # =====================================================

    st.header("Step 45 — AI Mock Interview")

    if "mock_previous_questions" not in st.session_state:
        st.session_state.mock_previous_questions = []

    if "mock_question" not in st.session_state:
        st.session_state.mock_question = ""

    if "mock_feedback" not in st.session_state:
        st.session_state.mock_feedback = ""

    if st.button(
        "Ask Me an Interview Question",
        key="mock_question_button"
    ):

        with st.spinner(
            "AI is preparing a question..."
        ):

            mock_result = generate_mock_interview_question(
                resume_text,
                job_description,
                st.session_state.mock_previous_questions
            )

        if mock_result["success"]:

            st.session_state.mock_question = (
                mock_result["text"]
            )

            st.session_state.mock_previous_questions.append(
                mock_result["text"]
            )

        else:

            st.error(
                mock_result["message"]
            )

    if st.session_state.mock_question:

        st.write(
            "### Interview Question"
        )

        st.info(
            st.session_state.mock_question
        )

        answer = st.text_area(
            "Your Answer",
            height=150,
            key="mock_answer"
        )

        if st.button(
            "Evaluate My Answer",
            key="evaluate_mock_button"
        ):

            with st.spinner(
                "AI is evaluating your answer..."
            ):

                feedback = evaluate_mock_interview_answer(
                    st.session_state.mock_question,
                    answer,
                    resume_text,
                    job_description
                )

            if feedback["success"]:

                st.session_state.mock_feedback = (
                    feedback["text"]
                )

            else:

                st.error(
                    feedback["message"]
                )

    if st.session_state.mock_feedback:

        st.write(
            "### AI Feedback"
        )

        st.markdown(
            st.session_state.mock_feedback
        )

    # =====================================================
    # STEP 46 — FINAL MATCH REPORT
    # =====================================================

    st.header("Step 46 — Final Resume Match Report")

    if st.button(
        "Generate Final Match Report",
        key="final_match_button"
    ):

        with st.spinner(
            "Generating final report..."
        ):

            final_match = generate_final_match_report(
                resume_text,
                job_description
            )

        if final_match["success"]:

            st.markdown(
                final_match["text"]
            )

        else:

            st.error(
                final_match["message"]
            )

    # =====================================================
    # STEP 47 — PROFESSIONAL PDF
    # =====================================================

    st.header(
        "Step 47 — Professional Resume PDF Generator"
    )

    if st.button(
        "Generate Professional Resume PDF",
        key="professional_pdf_button"
    ):

        try:

            professional_pdf_path = (
                create_professional_resume_pdf(
                    name,
                    email,
                    phone,
                    resume_text
                )
            )

            with open(
                professional_pdf_path,
                "rb"
            ) as pdf_file:

                professional_pdf_data = (
                    pdf_file.read()
                )

            st.success(
                "Professional Resume PDF generated successfully."
            )

            st.download_button(
                "Download Professional Resume",
                data=professional_pdf_data,
                file_name="AI_Professional_Resume.pdf",
                mime="application/pdf",
                key="professional_resume_download"
            )

        except Exception as e:

            st.error(
                f"Professional PDF generation error: {e}"
            )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    st.header("Resume Analysis Report")

    if st.button(
        "Generate Resume Analysis PDF",
        key="analysis_pdf_button"
    ):

        try:

            report_path = create_pdf_report(
                name,
                email,
                phone,
                overall_score,
                ats_score,
                skill_score,
                project_score,
                matched_skills,
                missing_skills
            )

            with open(
                report_path,
                "rb"
            ) as report_file:

                report_data = report_file.read()

            st.success(
                "Resume Analysis PDF generated."
            )

            st.download_button(
                "Download Analysis Report",
                data=report_data,
                file_name="Final_AI_Resume_Report.pdf",
                mime="application/pdf",
                key="analysis_report_download"
            )

        except Exception as e:

            st.error(
                f"Report generation error: {e}"
            )


# =========================================================
# STEP 48 — AI RESUME BUILDER
# =========================================================

st.header("Step 48 — AI Resume Builder")

st.write(
    "Create professional resume content using your own information."
)

builder_name = st.text_input(
    "Full Name",
    key="builder_name"
)

builder_email = st.text_input(
    "Email",
    key="builder_email"
)

builder_phone = st.text_input(
    "Phone",
    key="builder_phone"
)

builder_education = st.text_area(
    "Education",
    placeholder=(
        "Example: B.Tech Artificial Intelligence "
        "and Data Science, 2025-2029"
    ),
    height=100,
    key="builder_education"
)

builder_skills = st.text_area(
    "Skills",
    placeholder=(
        "Example: Python, SQL, Pandas, NumPy, "
        "Machine Learning"
    ),
    height=100,
    key="builder_skills"
)

builder_projects = st.text_area(
    "Projects",
    placeholder=(
        "Enter your actual projects and what "
        "you did in them."
    ),
    height=150,
    key="builder_projects"
)

builder_experience = st.text_area(
    "Experience",
    placeholder=(
        "Enter internship/work experience "
        "if you have any."
    ),
    height=100,
    key="builder_experience"
)

builder_certifications = st.text_area(
    "Certifications",
    placeholder=(
        "Enter your actual certifications."
    ),
    height=100,
    key="builder_certifications"
)

if st.button(
    "Build My AI Resume",
    key="build_ai_resume_button"
):

    if not builder_name.strip():

        st.warning(
            "Please enter your name."
        )

    elif not builder_skills.strip():

        st.warning(
            "Please enter your skills."
        )

    else:

        with st.spinner(
            "AI is building your resume..."
        ):

            builder_result = generate_ai_resume_builder(
                builder_name,
                builder_email,
                builder_phone,
                builder_education,
                builder_skills,
                builder_projects,
                builder_experience,
                builder_certifications,
                job_description
            )

        if builder_result["success"]:

            st.success(
                "AI Resume Created Successfully"
            )

            st.markdown(
                builder_result["text"]
            )

            st.download_button(
                "Download AI Resume Text",
                data=builder_result["text"],
                file_name="AI_Built_Resume.txt",
                mime="text/plain",
                key="ai_built_resume_download"
            )

            # Save generated resume in session
            st.session_state[
                "builder_result_text"
            ] = builder_result["text"]

        else:

            st.error(
                builder_result["message"]
            )


# =========================================================
# STEP 49 — AI RESUME BUILDER PDF
# =========================================================

st.header(
    "Step 49 — Download AI Resume as Professional PDF"
)

st.write(
    "Convert your AI-generated resume into a downloadable PDF."
)

if "builder_result_text" in st.session_state:

    if st.button(
        "Generate AI Resume PDF",
        key="generate_ai_builder_pdf"
    ):

        try:

            ai_pdf_path = create_ai_builder_pdf(
                st.session_state[
                    "builder_result_text"
                ]
            )

            with open(
                ai_pdf_path,
                "rb"
            ) as pdf_file:

                ai_pdf_data = pdf_file.read()

            st.success(
                "AI Resume PDF generated successfully."
            )

            st.download_button(
                "Download AI Resume PDF",
                data=ai_pdf_data,
                file_name="AI_Built_Professional_Resume.pdf",
                mime="application/pdf",
                key="download_ai_builder_pdf"
            )

        except Exception as e:

            st.error(
                f"PDF generation error: {e}"
            )

else:

    st.info(
        "First create your AI Resume in Step 48."
    )


# =========================================================
# STEP 50 — FINAL AI RESUME DASHBOARD
# =========================================================

st.header(
    "Step 50 — Final AI Resume Dashboard"
)

st.write(
    "Your complete AI Resume Analyzer dashboard."
)

if uploaded_file is not None:

    st.subheader(
        "Final Resume Overview"
    )

    # -----------------------------------------------------
    # SCORE CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Resume Score",
            f"{overall_score}/100"
        )

    with col2:

        st.metric(
            "ATS Score",
            f"{ats_score}/100"
        )

    with col3:

        st.metric(
            "Skill Score",
            f"{skill_score}/100"
        )

    with col4:

        st.metric(
            "Project Score",
            f"{project_score}/100"
        )

    st.divider()

    # -----------------------------------------------------
    # CANDIDATE INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "Candidate Information"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            "**Name:**",
            name
        )

        st.write(
            "**Email:**",
            email
        )

    with info_col2:

        st.write(
            "**Phone:**",
            phone
        )

        st.write(
            "**Skills Found:**",
            len(matched_skills)
        )

    st.divider()

    # -----------------------------------------------------
    # SKILLS ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Skills Analysis"
    )

    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:

        st.write(
            "### Matched Skills"
        )

        if matched_skills:

            for skill in matched_skills:

                st.success(skill)

        else:

            st.info(
                "No matched skills found."
            )

    with skill_col2:

        st.write(
            "### Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills[:20]:

                st.warning(skill)

        else:

            st.success(
                "No major missing skills found."
            )

    st.divider()

    # -----------------------------------------------------
    # RESUME SECTION ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Resume Section Analysis"
    )

    final_section_df = pd.DataFrame({
        "Section": list(sections.keys()),
        "Status": [
            "Present" if value else "Missing"
            for value in sections.values()
        ]
    })

    st.dataframe(
        final_section_df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # JOB MATCH
    # -----------------------------------------------------

    st.subheader(
        "Job Description Match"
    )

    if job_description.strip():

        st.metric(
            "Job Match",
            f"{job_match_percentage}%"
        )

        st.progress(
            job_match_percentage / 100
        )

        if matched_job_keywords:

            st.write(
                "Matched Job Keywords:"
            )

            st.write(
                ", ".join(
                    matched_job_keywords[:30]
                )
            )

    else:

        st.info(
            "Paste a job description above "
            "to calculate job match."
        )

    st.divider()

    # -----------------------------------------------------
    # FINAL RECOMMENDATIONS
    # -----------------------------------------------------

    st.subheader(
        "Final Recommendations"
    )

    recommendations = []

    if overall_score < 70:

        recommendations.append(
            "Improve the overall resume score."
        )

    if ats_score < 70:

        recommendations.append(
            "Improve ATS-friendly keywords "
            "and formatting."
        )

    if skill_score < 70:

        recommendations.append(
            "Add relevant technical skills "
            "that you actually know."
        )

    if project_score < 70:

        recommendations.append(
            "Add stronger project descriptions "
            "with your actual work."
        )

    if not job_description.strip():

        recommendations.append(
            "Add a target job description "
            "for job-specific optimization."
        )

    if recommendations:

        for recommendation in recommendations:

            st.warning(
                recommendation
            )

    else:

        st.success(
            "Your resume currently meets "
            "the basic checks of this analyzer."
        )

    st.divider()

    # -----------------------------------------------------
    # FINAL PDF
    # -----------------------------------------------------

    st.subheader(
        "Download Final Resume Report"
    )

    if st.button(
        "Generate Final Resume Report",
        key="final_resume_report"
    ):

        try:

            final_report_path = create_pdf_report(
                name,
                email,
                phone,
                overall_score,
                ats_score,
                skill_score,
                project_score,
                matched_skills,
                missing_skills
            )

            with open(
                final_report_path,
                "rb"
            ) as report_file:

                final_report_data = (
                    report_file.read()
                )

            st.success(
                "Final resume report generated successfully."
            )

            st.download_button(
                "Download Final Report",
                data=final_report_data,
                file_name="Final_AI_Resume_Report.pdf",
                mime="application/pdf",
                key="final_report_download"
            )

        except Exception as e:

            st.error(
                f"Report generation error: {e}"
            )

else:

    st.info(
        "Upload your Resume PDF above "
        "to view the Final Dashboard."
    )