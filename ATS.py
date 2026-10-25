import os
import re
from datetime import datetime
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# Regex patterns
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_PATTERN = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|\+\d{1,3}[-.\s]?\d{6,14}'
LINKEDIN_PATTERN = r'(?:linkedin\.com/in/|linkedin:?\s*)([a-zA-Z0-9-]+)'
GITHUB_PATTERN = r'(?:github\.com/|github:?\s*)([a-zA-Z0-9-]+)'
URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+'

# Action verbs for experience analysis (tuple, so str.startswith accepts it)
ACTION_VERBS = (
    'achieved', 'administered', 'analyzed', 'architected', 'automated',
    'built', 'collaborated', 'configured', 'created', 'delivered',
    'designed', 'developed', 'drove', 'enhanced', 'established',
    'executed', 'implemented', 'improved', 'increased', 'integrated',
    'launched', 'led', 'managed', 'mentored', 'migrated',
    'optimized', 'orchestrated', 'oversaw', 'pioneered', 'planned',
    'reduced', 'refactored', 'resolved', 'scaled', 'secured',
    'spearheaded', 'streamlined', 'supervised', 'transformed', 'upgraded'
)

months = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}
skills_weight = 40
experience_weight = 30
keywords_weight = 30

target_skills = [
    "python", "django", "javascript", "react", "sql",
    "machine learning", "deep learning", "data analysis",
    "artificial intelligence", "natural language processing",
    "project management", "communication", "problem solving",
    "rest api", "rest apis", "aws", "azure", "git"
]


def month_num(name):
    return months[name[:3].lower()]


def find_date(text):
    return re.search(
        r"\b\d{2}-\d{4}\s*[-–—]\s*\d{2}-\d{4}\b",
        text
    )


def normalize_dates(text):
    now = datetime.now().strftime("%m-%Y")

    month = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)"
    )

    # Month-name ranges: Jan 2022 - Dec 2024
    text = re.sub(
        rf"\b{month}\s+(\d{{4}})\s*[-–—]\s*"
        rf"{month}\s+(\d{{4}})\b",
        lambda m: (
            f"{month_num(m.group(1)):02d}-{m.group(2)} – "
            f"{month_num(m.group(3)):02d}-{m.group(4)}"
        ),
        text,
        flags=re.IGNORECASE
    )

    # Month-year to Present/Current/Now/Ongoing
    text = re.sub(
        rf"\b{month}\s+(\d{{4}})\s*[-–—]?\s*"
        r"(present|current|now|ongoing)\b",
        lambda m: (
            f"{month_num(m.group(1)):02d}-{m.group(2)} – {now}"
        ),
        text,
        flags=re.IGNORECASE
    )

    
    text = re.sub(
        r"\b(\d{4})\s*[-–—]\s*(present|current|now|ongoing)\b",
        lambda m: f"01-{m.group(1)} – {now}",
        text,
        flags=re.IGNORECASE
    )

    # Year-only ranges: 2022 - 2024
    text = re.sub(
        r"\b(\d{4})\s*[-–—]\s*(\d{4})\b",
        lambda m: f"01-{m.group(1)} – 12-{m.group(2)}",
        text
    )

    return text


experience_headings = {
    "experience", "work experience", "professional experience",
    "work history", "employment history", "previous experience"
}

skills_headings = {
    "skills", "key skills", "core skills", "technical skills",
    "professional skills", "soft skills"
}

other_headings = {
    "education", "interests", "interest", "qualifications",
    "certificates", "certifications", "objectives", "summary",
    "projects", "references", "achievements", "languages"
}

all_headings = experience_headings | skills_headings | other_headings


def clean_heading(text):
    return re.sub(r"[:\s]+$", "", text.strip().lower())



def has_word(skill, text):
    return re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text) is not None


def extract_information(text):
    text = text.split("\n")
    text = text[:20]
    # FIX 4 (UnboundLocalError)
    # PROBLEM: these were only assigned when found, so a missing email,
    #          github etc. crashed at the return statement.
    # FIXED:   give every field a default value.
    email = ""
    name = ""
    linkedin_profile = ""
    github = ""
    url = ""
    for line in text:
        # FIX 5 (invalid regex)
        # PROBLEM: r"[A-Z]+\s+{1,3}" raised "multiple repeat".
        # FIXED:   r"[A-Z]+\s{1,3}"
        if re.search(r"[A-Z]+\s{1,3}", line) and not re.search(EMAIL_PATTERN, line) and not re.search(LINKEDIN_PATTERN, line):
            name = line
        if re.search(EMAIL_PATTERN, line):
            email = line
        if re.search(LINKEDIN_PATTERN, line):
            linkedin_profile = line
        if re.search(GITHUB_PATTERN, line):
            github = line
        # FIX 6 (typo)
        # PROBLEM: re.searc(...) -> AttributeError
        # FIXED:   re.search(...)
        if re.search(URL_PATTERN, line):
            url = line
    return {"Name": name,
            "Email": email,
            "Linkedin": linkedin_profile,
            "Github Profile": github,
            "URL": url}


# Detects sections like Experience, Skills etc. from the extracted text and
# stores each section's lines in a dictionary
def detect_sections(extracted_content):
    sections = {
        "skills": [],
        "experience": [],
        "other": []
    }
    
    current_section = "other"

    text = extracted_content["text"]
    lines = text.split("\n")
    for line in lines:
        heading = clean_heading(line)

        if heading in experience_headings:
            current_section = "experience"
            continue

        if heading in skills_headings:
            current_section = "skills"
            continue

        if heading in other_headings:
            current_section = "other"
            continue

        sections[current_section].append(line)

    return sections



def extract_skills(extracted_content, sections):
    resume_skills_text = " ".join(sections["skills"]).lower()

    jd_keywords, _, jd_skill_lines = extracted_content["job_description"]
    jd_skills_text = " ".join(jd_skill_lines).lower()
    # JD skills = target skills that appear in the JD skills section;
    # if the JD has no skills section, fall back to all JD keywords.
    job_skills = [s.lower() for s in jd_keywords if has_word(s.lower(), jd_skills_text)]
    if not job_skills:
        job_skills = [s.lower() for s in jd_keywords]

    if not job_skills:
        return {"match_percentage": 0, "weighted_score": 0, "missing_skills": []}

    matched_skill = [skill for skill in job_skills if has_word(skill, resume_skills_text)]
    match_percentage = len(matched_skill) / len(job_skills) * 100
    weighted_score = match_percentage * skills_weight / 100
    missing_skills = [skill for skill in job_skills if skill not in matched_skill]
    return {"match_percentage": match_percentage,
            "weighted_score": weighted_score,
            "missing_skills": missing_skills}


def get_entries(experience_section, text):
    
    if isinstance(experience_section, list):
        experience_section = "\n".join(experience_section)

    

    job_pattern = re.compile(
        r"(?:\w+\s){0,2}"
        r"(?:Engineer|Developer|Manager|Analyst|Designer|Intern|Consultant|"
        r"Director|Lead|Officer|Specialist|Administrator|Architect)\b",
        re.IGNORECASE)

    company_pattern = re.compile(r"^[A-Z][\w&.']*(?:\s+[A-Z][\w&.']*){0,3}$")

   
    location_patterns = re.compile(
        r'(?:Location|Address|Based in|City)[:\s]+([A-Za-z\s,]+)|'
        r'([A-Za-z]+,\s*[A-Z]{2})\s*\d{5}|'  # City, ST ZIP
        r'([A-Za-z]+,\s*[A-Za-z\s]+,\s*[A-Za-z]+)'  # City, State, Country
    )

    def get_location(header):
        m = location_patterns.search(header)
        if not m:
            return None
        return next((g.strip() for g in m.groups() if g), m.group())

    entries = []
    text = experience_section if experience_section else text
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    pending = []        
    current = None

    for line in lines:

        is_bullet = line.lower().startswith(ACTION_VERBS) or detect_bullet_lines(line)

        
        date = find_date(line)
        if date:
            current = {
                "job_title": None,
                "company_name": None,
                "location": None,
               
                "date": date.group(),
                "responsibilities": [],
            }
            entries.append(current)

            
            for header in pending + [line]:
                title_match = job_pattern.search(header)
                if current["job_title"] is None and title_match:
                    current["job_title"] = title_match.group()
                # FIX 17: a line like "Python Developer" also matched
                #         company_pattern, so the title was stored as the
                #         company. Skip lines that contain a job title.
                if current["company_name"] is None and not title_match:
                    m = company_pattern.search(header)
                    if m:
                        current["company_name"] = m.group()
                if current["location"] is None:
                    current["location"] = get_location(header)

            pending = []
            continue

         # First Entry
        if current is None:
            pending.append(line)
            continue

        # BULLET: responsibility of the current entry 
        if is_bullet:
            current["responsibilities"].append(line.lstrip("•-*●○ ").strip())
            continue

        # Plain line, current entry has no bullets yet:
        if not current["responsibilities"]:
            title_match = job_pattern.search(line)
            if current["job_title"] is None and title_match:
                current["job_title"] = title_match.group()
            if current["company_name"] is None and not title_match:
                m = company_pattern.search(line)
                if m:
                    current["company_name"] = m.group()
            continue

        # Plain line after bullets: next job's header OR wrapped bullet ----
        if job_pattern.search(line) or company_pattern.search(line):
            pending.append(line)                       # belongs to the NEXT date
        else:
            current["responsibilities"][-1] += " " + line   # wrapped bullet text

    return entries


def score_job(job):
    job_points = 0
    company_points = 0
    date_points = 0
    responsibility_points = 0

    if job.get("job_title"):
        job_points = 10

    if job.get("company_name"):
        company_points = 10

    if job.get("date"):
        date_points = 20

    if job.get("responsibilities"):
        responsibility_points = 60

    total_score = (
        job_points
        + company_points
        + date_points
        + responsibility_points
    )

    return total_score



def calculate_similarity_score(extracted_content, experience_section):
    experience = extracted_content["job_description"][1].get("experience")

    entries = get_entries(experience_section, extracted_content["text"])
    resume_experience = "\n".join(
        resp for entry in entries for resp in entry["responsibilities"]
    )

    if not (experience and resume_experience):
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        jd_vector = vectorizer.fit_transform([experience])
        resume_vector = vectorizer.transform([resume_experience])
    except ValueError:  # empty vocabulary
        return 0.0
    score = cosine_similarity(jd_vector, resume_vector)[0][0]
    return float(score) * experience_weight



def calculate_similarity_score_keywords(extracted_content):
    jd_keywords = extracted_content["job_description"][0]
    if not jd_keywords:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        jd_vector = vectorizer.fit_transform([" ".join(sorted(jd_keywords))])
        resume_vector = vectorizer.transform([extracted_content["text"]])
    except ValueError:
        return 0.0
    score = cosine_similarity(jd_vector, resume_vector)[0][0]
    return float(score) * keywords_weight

# Called to idenify responsibilities in experience section

def detect_bullet_lines(line):
    stripped = line.strip()
    
    if stripped.startswith(('•', '-', '*', '●', '○')) or re.match(r"\d+[.)]\s", stripped):
        return True
    return False



def read_job_description(job_description, target_skills=target_skills):
    
    with open(job_description, "r", encoding="utf-8") as file:
        file_content = file.read().lower()

    # Skills
    keywords = set()
    for skill in target_skills:
        #Prevents case insensitive matching
        if re.search(rf"\b{re.escape(skill.lower())}\b", file_content):
            keywords.add(skill)

    # Experience section
    exp_headers = ("experience", "required experience",
                   "minimum professional experience", "professional experience")
    other_headers = ("objectives", "about us",
                     "company culture", "working conditions", "nice to have")
    skills_headers = ("skills", "target skills", "required skills", "key skills", "core skills")

    # Detects header for required sections
    def is_header(line, headers):
        return line.startswith(headers) and len(line) <= 40

    skills_line = []
    exp_lines = []
    in_experience = False
    in_skills = False

    for line in file_content.split("\n"):
        line = line.strip()
        if not line:
            continue

        
        if is_header(line, exp_headers):
            in_experience = True
            in_skills = False
            continue

        if is_header(line, skills_headers):
            in_skills = True
            in_experience = False
            continue

        if is_header(line, other_headers):
            in_experience = False
            in_skills = False      # left the section
            continue

        if in_experience:
            exp_lines.append(line)
        if in_skills:
            skills_line.append(line)

    experience = {"experience": "\n".join(exp_lines)}
    return keywords, experience, skills_line


def calculate_keyword_overlap(job_keywords, resume_text):
    resume_text = resume_text.lower()
    # uses predefined skills list as a keywords
    matched = [k for k in job_keywords if re.search(rf"\b{re.escape(k.lower())}\b", resume_text)]
    missing = [k for k in job_keywords if k not in matched]

    score = (len(matched) / len(job_keywords)) * keywords_weight if job_keywords else 0
    return {"match_percentage": score, "missing_keywords": missing}


def read_docx(file_path, job_description):
    document = Document(file_path)
    text = ""

    for para in document.paragraphs:
        
        if not para.text.strip():
            continue
        text += para.text.strip() + "\n"

    text = normalize_dates(text)

    analyze_job_description = read_job_description(job_description, target_skills)

    return {"text": text, "job_description": analyze_job_description}


def read_pdf(file_path, job_description):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        
        page_text = page.extract_text()
        if not page_text:
            continue
        text += page_text + "\n"
    text = normalize_dates(text)
    
    if not text.strip():
        return None

    analyze_job_description = read_job_description(job_description, target_skills)
    return {"text": text, "job_description": analyze_job_description}



def process_resume(extracted_content):
    sections = detect_sections(extracted_content)
    experience_section = sections["experience"]

    entries = get_entries(experience_section, extracted_content["text"])

    skills_result = extract_skills(extracted_content, sections)
    experience_score = calculate_similarity_score(extracted_content, experience_section)
    keywords_result = calculate_keyword_overlap(
        extracted_content["job_description"][0], extracted_content["text"]
    )

    total_score = (
        skills_result["weighted_score"]
        + experience_score
        + keywords_result["match_percentage"]
    )
# Returns individual score for Skills, Experience and keywords and a combined score of resume
    return {
        "entries": entries,
        "skills_score": skills_result["weighted_score"],
        "missing_skills": skills_result["missing_skills"],
        "experience_score": experience_score,
        "keywords_score": keywords_result["match_percentage"],
        "missing_keywords": keywords_result["missing_keywords"],
        "total_score": total_score,
    }


# FIX 30 (missing orchestration)
# PROBLEM: read_folder was commented out, and the bugs in it were: readers
#          called with 1 argument, unsupported files passed on as None, no
#          per-file try/except, and only the last score returned.
# FIXED:   restored, one result per file, bad files are skipped.
def read_folder(folder_path, job_description):
    scores = []
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print("Invalid Folder Path")
        return scores

    for file in folder_path.iterdir():
        suffix = file.suffix.lower()
        try:
            if suffix == ".docx":
                extracted_content = read_docx(file, job_description)
            elif suffix == ".pdf":
                extracted_content = read_pdf(file, job_description)
            else:
                continue  # unsupported file type

            if not extracted_content:
                print(f"Skipped {file.name}: no text could be extracted")
                continue

            result = process_resume(extracted_content)
            result["file"] = file.name
            scores.append(result)
            print(f"\nResume: {file.name}")
            print(f"Total score: {result['total_score']:.2f}")
            print(f"Missing skills: {result['missing_skills']}")
            print(f"Missing keywords: {result['missing_keywords']}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return scores



def main():
    folder_path = input("Enter your folder path: ").strip()
    job_description = input("Enter job description file path (.txt): ").strip()
    if not os.path.isfile(job_description):
        print("Job description file not found")
        return
    read_folder(folder_path, job_description)


if __name__ == "__main__":
    main()