"""
Role determination logic based on career history
"""
from typing import Tuple, Optional
from datetime import datetime
from . import schemas
from . import llm


PROMPT_DETERMINE_CURRENT_ROLE = """
You are an expert in academic and industrial career trajectory analysis.

TASK:
Determine the CURRENT role and affiliation of a researcher based solely on their provided career history.

CURRENT YEAR: {current_year}
(Use this year as reference when deciding whether roles are current, past, or future.)

INPUT DATA:
Author Name: {author_name}

Career & Education History:
{career_education_text}

Industrial Experience:
{industrial_experience_text}

------------------------------------------------------------
ROLE CATEGORIES (choose EXACTLY ONE):
1. PhD Student – currently pursuing a PhD degree
2. Master Student – currently pursuing a Master's degree
3. Undergraduate Student – currently pursuing an undergraduate/Bachelor's degree
4. Professor – university faculty (Assistant/Associate/Full Professor, Lecturer, Faculty)
5. Postdoc – postdoctoral researcher or fellow
6. Industrial Researcher – researcher/scientist/engineer at a company
7. Institution Researcher – researcher at a non-company, non-university research institute
8. Unknown – current role cannot be reliably determined from the given information
------------------------------------------------------------

CURRENT ROLE IDENTIFICATION RULES:

1. For each role entry, extract if possible:
   - start_year
   - end_year
   - whether the text explicitly contains “Present”, “Current”, or “Now”.

2. A role is CURRENT if and only if at least ONE holds:
   - The duration explicitly contains “Present”, “Current”, or “Now”.
   - There is NO end year (e.g., “2022–”, “since 2020”) AND start_year (if given) ≤ {current_year}.
   - Both start_year and end_year exist and satisfy:
       start_year ≤ {current_year} ≤ end_year.

3. A role is PAST if:
   - end_year exists AND end_year < {current_year}.

4. A role is FUTURE if:
   - start_year exists AND start_year > {current_year}.
   - FUTURE roles MUST NOT be treated as current.

5. IMPORTANT:
   - Use numeric comparison with {current_year} when years are available.
   - Do NOT assume a role is current based on title or expertise alone.
   - If you cannot reliably identify any CURRENT role, you MUST output role_category = "Unknown".

------------------------------------------------------------
MULTIPLE CURRENT ROLES — SELECTION STRATEGY:

6. When more than one role is CURRENT, you MUST select the final current role in TWO stages:

   STEP 1 — BUILD CANDIDATE SET WITH “PRESENT” PRIORITY (but preserving students vs internships):

   Let:
   - CURRENT_ALL = all roles that are CURRENT by the rules above.
   - For each role, determine:
       * is_student_role: PhD/Master/Undergraduate student enrollment.
       * is_intern_role: title contains “Intern”, “Research Intern”, or “Student Intern”.
       * has_present_flag: text explicitly contains “Present”, “Current”, or “Now”.

   Define:
   - PRESENT_NON_INTERN = roles in CURRENT_ALL where:
       has_present_flag == True AND is_intern_role == False
       (This includes Professor, Postdoc, Industrial, Institution roles, and students with “Present”, but EXCLUDES internships.)
   - PRESENT_INTERN = roles in CURRENT_ALL where:
       has_present_flag == True AND is_intern_role == True

   Candidate set construction:
   - If PRESENT_NON_INTERN is NOT empty:
       * The candidate set is PRESENT_NON_INTERN ONLY.
       * All roles that are CURRENT only by year range, and all internships, are ignored in this step.
       (This ensures that clearly marked “Present” non-intern roles such as Professor/Postdoc/Researcher
        can override overlapping past student degrees.)
   - Else (PRESENT_NON_INTERN is empty, i.e., any “Present” roles are internships only, or there is no “Present” at all):
       * The candidate set is CURRENT_ALL.
       (In this case, student enrollment may override internships.)

   STEP 2 — APPLY CATEGORY PRIORITY WITHIN THE CANDIDATE SET:

   6.2 Within this candidate set:
       - First apply the STRICT PRIORITY RULES (students vs professors vs industry vs institutes) **only inside this set**.
       - If there is still more than one option, choose the one with the most recent START YEAR.

7. If NO role satisfies the CURRENT criteria:
   - Set role_category = "Unknown"
   - Set role_text = "Unknown"
   - Set affiliation = "Unknown".

------------------------------------------------------------
STRICT PRIORITY RULES (applied ONLY inside the candidate set from rule 6):

1. Active student enrollment (PhD / Master / Undergraduate) overrides:
   - Internships
   - Industrial positions
   - Visiting roles
   - Temporary appointments
   - Research intern / student intern roles

   (For example, if both a PhD Student and a Research Intern are in the candidate set,
    you must choose the PhD Student.)

2. Academic positions (Professor, Postdoc) apply ONLY when:
   - There is NO student enrollment in the candidate set.

3. Industrial Researcher applies when:
   - There is NO student enrollment, AND
   - The selected role is at a COMPANY (including internships).

4. Institution Researcher applies when:
   - There is NO student enrollment, AND
   - The selected role is at a university-affiliated institute, national lab, or research institute (including internships).

------------------------------------------------------------
CRITICAL RULE — CURRENT INTERNSHIPS:
If the ONLY role in the candidate set (after applying the time rules and rule 6) is an internship
(e.g., “Intern”, “Research Intern”, “Student Intern”):

- Treat this internship as the TRUE current role.
- Classify by organization type:
    * Company → Industrial Researcher
    * University or standalone research institute → Institution Researcher
- The role_text should be “Research Intern”, “Intern”, or a similar phrase.

------------------------------------------------------------
CRITICAL RULE — UNDERGRADUATE STUDENTS:
- If history shows ONLY undergraduate enrollment with NO explicit evidence of a Master’s or PhD,
  and the undergraduate enrollment is CURRENT by the time rules,
  the role MUST be “Undergraduate Student”.
- DO NOT infer future graduate enrollment.

------------------------------------------------------------
CATEGORY MAPPING RULES:

PhD Student:
- Explicit markers: “PhD”, “Ph.D.”, “Doctoral”, “Doctorate”.

Master Student:
- Explicit: “Master”, “M.S.”, “M.Sc.”, “M.A.”.

Graduate Student (ambiguous):
- Interpret level (PhD vs Master) ONLY if explicitly clarified somewhere.

Undergraduate Student:
- Indicators: “undergraduate”, “Bachelor”, “B.S.”, “B.A.”, “undergrad”.
- If this is the ONLY enrollment stage and CURRENT → choose Undergraduate Student.

Professor:
- “Professor”, “Assistant Professor”, “Associate Professor”, “Lecturer”, “Faculty”.

Postdoc:
- “Postdoc”, “Postdoctoral”, “Research Fellow”, “Postdoctoral Fellow”.

Industrial Researcher:
- Research/science/engineering roles at COMPANIES (Google, Meta, Amazon, Microsoft, Apple, etc.).
- Internships at companies → Industrial Researcher.

Institution Researcher:
- Research roles at research institutes or university-affiliated institutes.
- Internships at such institutes → Institution Researcher.

NOTE:
If the organization name contains a university, treat it as academic (NOT Institution Researcher).
If it is a standalone research institute or national lab, classify as Institution Researcher.

------------------------------------------------------------
AFFILIATION & ROLE TEXT:

- affiliation:
  * Use the exact organization name from the selected entry WITHOUT modification.
  * If missing or role_category is “Unknown”, return "Unknown".

- role_text:
  * Short natural phrase describing the role (e.g., “PhD student”, “Research Intern”, “Assistant Professor”).
  * If role_category is “Unknown” → role_text = "Unknown".

------------------------------------------------------------
EXPLICIT EXAMPLES (ASSUME current_year = 2025):

EXAMPLE 1 — PhD that has already ended:
PhD student @ University A [2018–2022]
→ end_year 2022 < 2025 → PAST. If no other current role → "Unknown".

EXAMPLE 2 — PhD still in progress:
PhD student @ University B [2022–2027]
→ 2022 ≤ 2025 ≤ 2027 → CURRENT → PhD Student @ University B.

EXAMPLE 3 — PhD with end year + Postdoc with Present:
Postdoc @ CMU [2025–Present]
PhD student @ Princeton University [2017–2025]
→ Both CURRENT by time; Postdoc has "Present" and is NOT internship.
→ Candidate set contains only Postdoc → Final role: Postdoc @ CMU.

EXAMPLE 4 — Student + Research Intern (student should win):
PhD student @ University C [2022–2027]
Research Intern @ Company X [2024–Present]
→ Both CURRENT; only Present role is an internship.
→ PRESENT_NON_INTERN is empty → candidate set = BOTH roles.
→ Student overrides internship → Final role: PhD Student @ University C.

EXAMPLE 5 — Postdoc with Present:
PhD student @ Princeton University [2017–2022]
Postdoc @ CMU [2023–Present]
→ PhD PAST, Postdoc CURRENT → Final role: Postdoc @ CMU.

EXAMPLE 6 — Future role:
Assistant Professor @ University D [2027– ]
→ start_year 2027 > 2025 → FUTURE, NOT current.
→ If no other current role → "Unknown".

------------------------------------------------------------
OUTPUT FORMAT (JSON only):
{{
  "role_category": "<one of the 8 categories EXACTLY as listed: 'PhD Student', 'Master Student', 'Undergraduate Student', 'Professor', 'Postdoc', 'Industrial Researcher', 'Institution Researcher', or 'Unknown'>",
  "role_text": "<natural language role or 'Unknown'>",
  "affiliation": "<exact institution/company name or 'Unknown'>",
  "explanation": "<1–2 sentences explaining the reasoning>"
}}

CRITICAL: The role_category field MUST be EXACTLY one of these 8 strings (case-sensitive):
- "PhD Student"
- "Master Student"  
- "Undergraduate Student"
- "Professor"
- "Postdoc"
- "Industrial Researcher"
- "Institution Researcher"
- "Unknown"

DO NOT use variations like "Postdoc only", "PhD", "Ph.D. Student", etc. Use the EXACT strings listed above.
"""

def determine_current_role_from_profile(
    enhanced_profile: schemas.EnhancedAuthorProfile,
    api_key: Optional[str] = None
) -> Tuple[str, str, str, str]:
    """
    根据 EnhancedAuthorProfile 的 career_education_history 和 industrial_experience 决定当前 role
    
    Args:
        enhanced_profile: 完整的 EnhancedAuthorProfile
        api_key: LLM API key
        
    Returns:
        (role_category, role_text, affiliation, explanation)
        - role_category: 7个类别之一 (PhD Student, Master Student, Undergraduate Student, Professor, Postdoc, Industrial Researcher, Institution Researcher)
        - role_text: 自然语言描述 (e.g., "PhD student", "Undergraduate student")
        - affiliation: 机构名称
        - explanation: 选择理由
    """
    author_name = enhanced_profile.introduction.name or "Unknown"
    
    # 构建 career_education_text
    career_edu_lines = []
    if enhanced_profile.career_education_history:
        for idx, item in enumerate(enhanced_profile.career_education_history, 1):
            degree_or_pos = item.degree_or_position or "N/A"
            institution = item.institution or "N/A"
            duration = item.duration or "N/A"
            department = item.department or ""
            field = item.field or ""
            
            line = f"[{idx}] {degree_or_pos} at {institution}"
            if duration:
                line += f" ({duration})"
            if department:
                line += f", {department}"
            if field:
                line += f", Field: {field}"
            career_edu_lines.append(line)
    
    career_education_text = "\n".join(career_edu_lines) if career_edu_lines else "No career/education history available"
    
    # 构建 industrial_experience_text
    industrial_lines = []
    if enhanced_profile.industrial_experience:
        for idx, item in enumerate(enhanced_profile.industrial_experience, 1):
            position = item.position or "N/A"
            organization = item.organization or "N/A"
            duration = item.duration or "N/A"
            
            line = f"[{idx}] {position} at {organization}"
            if duration:
                line += f" ({duration})"
            industrial_lines.append(line)
    
    industrial_experience_text = "\n".join(industrial_lines) if industrial_lines else "No industrial experience available"
    
    # 如果两者都为空，返回默认值
    if not career_edu_lines and not industrial_lines:
        print(f"[Role Determination] No career history available for {author_name}, using default")
        return ("Unknown", "", "", "No career history available")
    
    # 获取当前年份
    current_year = datetime.now().year
    
    # 调用 LLM
    try:
        prompt = PROMPT_DETERMINE_CURRENT_ROLE.format(
            current_year=current_year,
            author_name=author_name,
            career_education_text=career_education_text,
            industrial_experience_text=industrial_experience_text
        )
        
        llm_instance = llm.get_llm("role_determination", temperature=0.0, api_key=api_key)
        response = llm.safe_structured(llm_instance, prompt, schemas.LLMRoleDeterminationSpec)
        
        if response and hasattr(response, 'role_category'):
            role_category = response.role_category or "Unknown"
            role_text = response.role_text or ""
            affiliation = response.affiliation or ""
            explanation = response.explanation or ""
            
            # ✅ 验证 role_category 是否为有效值，如果不是则设为 "Unknown"
            valid_categories = {
                "PhD Student", "Master Student", "Undergraduate Student",
                "Professor", "Postdoc", "Industrial Researcher", 
                "Institution Researcher", "Unknown"
            }
            if role_category not in valid_categories:
                print(f"[Role Determination] ⚠️ Invalid role_category '{role_category}' for {author_name}, converting to 'Unknown'")
                explanation = f"Invalid category '{role_category}' was returned. " + (explanation or "")
                role_category = "Unknown"
            
            print(f"[Role Determination] {author_name} → {role_category}: {role_text} at {affiliation}")
            print(f"  Reason: {explanation}")
            
            return (role_category, role_text, affiliation, explanation)
        else:
            print(f"[Role Determination] LLM response format error for {author_name}")
            return ("Unknown", "", "", "LLM response format error")
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"[Role Determination] Error for {author_name}: {error_type}: {error_msg}")
        if "role_category" in error_msg or "ValidationError" in error_type:
            print(f"[Role Determination] This appears to be a JSON parsing/validation error.")
            print(f"[Role Determination] The LLM may have returned malformed JSON. Using fallback.")
        return ("Unknown", "", "", f"Error: {error_msg}")
