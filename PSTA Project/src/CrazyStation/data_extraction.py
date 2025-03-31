import re
import datetime
from dateutil.parser import parse as date_parse
import logging
from pathlib import Path

# Import original helper functions
from resources.helper import (
    extract_work_experience as basic_extract_work_experience,
    extract_education_experience as basic_extract_education,
    extract_language_knowledge as basic_extract_language
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
GERMAN_MONTHS = {
    'januar': '01', 'februar': '02', 'märz': '03', 'april': '04',
    'mai': '05', 'juni': '06', 'juli': '07', 'august': '08',
    'september': '09', 'oktober': '10', 'november': '11', 'dezember': '12'
}

SKILL_CATEGORIES = {
    "programming": ["python", "java", "javascript", "c++", "c#", "sql", "php", "html", "css"],
    "data_science": ["machine learning", "data analysis", "statistics", "pandas", "numpy", 
                     "deep learning", "neural networks", "tensorflow", "pytorch", "big data"],
    "project_management": ["scrum", "agile", "kanban", "waterfall", "jira", "ms project", 
                          "projektmanagement", "prince2", "pmi", "pmp"],
    "languages": ["deutsch", "englisch", "französisch", "spanisch", "italienisch", 
                  "russisch", "chinesisch", "japanisch"],
    "soft_skills": ["teamwork", "communication", "leadership", "problem solving", 
                    "time management", "creativity", "adaptability", "teamarbeit", 
                    "kommunikation", "führung", "problemlösung", "zeitmanagement"]
}

def normalize_date(date_str):
    """Normalize date strings to a standard format (DD.MM.YYYY)"""
    if not date_str:
        return None
        
    # Convert to lowercase and handle special cases
    date_lower = date_str.lower()
    if date_lower in ['heute', 'present', 'aktuell', 'now', 'current']:
        return datetime.datetime.now().strftime('%d.%m.%Y')
        
    # Try different date parsing strategies
    try:
        # Try standard parsing
        parsed_date = date_parse(date_str, fuzzy=True)
        return parsed_date.strftime('%d.%m.%Y')
    except Exception:
        # Check for German month names
        for month_name, month_num in GERMAN_MONTHS.items():
            if month_name in date_lower:
                # Extract year
                year_match = re.search(r'(\d{4})', date_lower)
                if year_match:
                    year = year_match.group(1)
                    # Default to 1st day of month if day not specified
                    return f"01.{month_num}.{year}"
        
        # Just year
        if re.match(r'^\d{4}$', date_str):
            return f"01.01.{date_str}"
            
        # Return original if we couldn't parse
        return date_str

def extract_skills_from_text(text):
    """Extract skills from text using pattern matching"""
    skills = []
    text_lower = text.lower()
    
    # Extract skills from predefined categories
    for category, category_skills in SKILL_CATEGORIES.items():
        for skill in category_skills:
            if skill in text_lower:
                skills.append(skill)
    
    # Look for skill sections
    skill_section_indicators = [
        "kenntnisse", "fähigkeiten", "skills", "kompetenzen", 
        "competencies", "expertise", "qualifikationen", "abilities"
    ]
    
    for indicator in skill_section_indicators:
        if indicator in text_lower:
            # Find the skill section
            pattern = rf"{indicator}[:\s]+((?:[^.;]*[,.])*)"
            matches = re.search(pattern, text_lower)
            if matches:
                skill_section = matches.group(1)
                # Split by commas or similar separators
                potential_skills = re.split(r'[,;.]', skill_section)
                for skill in potential_skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2 and skill not in skills:
                        skills.append(skill)
    
    return skills

def extract_work_experience(text):
    """
    Extract work experience with enhanced skill detection
    
    Args:
        text (str): Text content from resume
        
    Returns:
        list: List of work experience entries with skills
    """
    # Start with basic extraction
    work_experiences = basic_extract_work_experience(text)
    
    # Enhance with skills
    for exp in work_experiences:
        # Create a context string from job details
        context = f"{exp.get('job_title', '')} {exp.get('company_name', '')} {exp.get('industry', '')}"
        
        # Extract skills
        exp['skills'] = extract_skills_from_text(context)
        
        # Normalize dates
        if 'start_date' in exp:
            exp['start_date'] = normalize_date(exp['start_date'])
        if 'end_date' in exp:
            exp['end_date'] = normalize_date(exp['end_date'])
    
    return work_experiences

def extract_education(text):
    """
    Extract education with enhanced date normalization
    
    Args:
        text (str): Text content from resume
        
    Returns:
        list: List of education entries
    """
    # Start with basic extraction
    education_experiences = basic_extract_education(text)
    
    # Enhance with normalized dates
    for edu in education_experiences:
        if 'start_date' in edu:
            edu['start_date'] = normalize_date(edu['start_date'])
        if 'end_date' in edu:
            edu['end_date'] = normalize_date(edu['end_date'])
    
    return education_experiences

def extract_language_knowledge(text):
    """
    Extract language knowledge with proficiency scoring
    
    Args:
        text (str): Text content from resume
        
    Returns:
        list: List of language knowledge entries with scores
    """
    # Start with basic extraction
    languages = basic_extract_language(text)
    
    # Add proficiency scores
    for lang in languages:
        level = lang.get('level', '')
        
        # Assign score based on proficiency level
        if 'Muttersprache' in level:
            lang['score'] = 5
        elif 'Verhandlungssicher' in level:
            lang['score'] = 4
        elif 'Konversationsfähig' in level:
            lang['score'] = 3
        elif 'Grundkenntnisse' in level:
            lang['score'] = 2
        else:
            lang['score'] = 0
    
    return languages

def calculate_experience_years(work_experiences):
    """Calculate total years of work experience"""
    total_years = 0
    current_date = datetime.datetime.now()
    
    for exp in work_experiences:
        try:
            # Parse dates
            if 'start_date' not in exp or not exp['start_date']:
                continue
                
            try:
                start_date = datetime.datetime.strptime(exp['start_date'], '%d.%m.%Y')
            except ValueError:
                # Try alternative format
                continue
                
            if 'end_date' not in exp or not exp['end_date'] or exp['end_date'] == 'TT.MM.JJJJ':
                end_date = current_date
            else:
                try:
                    end_date = datetime.datetime.strptime(exp['end_date'], '%d.%m.%Y')
                    # Ensure end date is not in the future
                    if end_date > current_date:
                        end_date = current_date
                except ValueError:
                    # If parsing fails, use current date
                    end_date = current_date
            
            # Calculate duration
            duration = end_date - start_date
            years = duration.days / 365.25
            
            # Add to total (ignore negative or extremely large values)
            if 0 <= years <= 50:  # Sanity check
                total_years += years
        except Exception as e:
            logger.warning(f"Error calculating experience duration: {str(e)}")
                
    return round(total_years, 1)

def determine_highest_education_level(education_experiences):
    """Determine highest education level from education experiences"""
    education_ranks = {
        "promotion": 8, "doktor": 8, "phd": 8, "doctorate": 8,
        "master": 7, "mba": 7, "magister": 7, "diplom": 6, 
        "bachelor": 5, "bsc": 5, "ba": 5,
        "fachhochschule": 4, "ausbildung": 3, "berufsausbildung": 3,
        "abitur": 2, "fachabitur": 2, "hochschulreife": 2,
        "mittlere reife": 1, "hauptschulabschluss": 0
    }
    
    highest_rank = 0
    highest_education = None
    
    for edu in education_experiences:
        if 'degree' in edu and edu['degree']:
            degree_lower = edu['degree'].lower()
            
            # Check against our ranking system
            for key, rank in education_ranks.items():
                if key in degree_lower and rank > highest_rank:
                    highest_rank = rank
                    highest_education = edu.copy()
    
    # Map rank to a standardized level
    if highest_education:
        if highest_rank >= 8:
            highest_education['level'] = "Doctorate"
            highest_education['level_score'] = 5
        elif highest_rank >= 6:
            highest_education['level'] = "Master's Degree"
            highest_education['level_score'] = 4
        elif highest_rank >= 5:
            highest_education['level'] = "Bachelor's Degree"
            highest_education['level_score'] = 3
        elif highest_rank >= 3:
            highest_education['level'] = "Vocational Training"
            highest_education['level_score'] = 2
        elif highest_rank >= 2:
            highest_education['level'] = "High School"
            highest_education['level_score'] = 1
        else:
            highest_education['level'] = "Basic Education"
            highest_education['level_score'] = 0
                
    return highest_education

def extract_comprehensive_profile(text):
    """
    Extract comprehensive profile from resume text
    
    Args:
        text (str): Text content from resume
        
    Returns:
        dict: Comprehensive profile with all extracted information
    """
    # Extract all components
    work_experience = extract_work_experience(text)
    education = extract_education(text)
    languages = extract_language_knowledge(text)
    skills = extract_skills_from_text(text)
    
    # Calculate derived metrics
    total_years_experience = calculate_experience_years(work_experience)
    highest_education = determine_highest_education_level(education)
    
    # Determine skill categories
    skill_categories = {}
    for category, category_skills in SKILL_CATEGORIES.items():
        category_skills_found = [skill for skill in skills if skill in category_skills]
        skill_categories[category] = len(category_skills_found)
    
    # Create complete profile
    profile = {
        'work_experience': work_experience,
        'education': education,
        'languages': languages,
        'skills': skills,
        'total_years_experience': total_years_experience,
        'highest_education': highest_education,
        'skill_categories': skill_categories,
        'education_level_score': highest_education['level_score'] if highest_education else 0
    }
    
    return profile

def match_candidate_to_job(candidate_profile, job_requirements):
    """
    Calculate match score between candidate profile and job requirements
    
    Args:
        candidate_profile (dict): Comprehensive candidate profile
        job_requirements (dict): Job requirements with weighted skills
        
    Returns:
        dict: Match scores and details
    """
    match_result = {
        'overall_match': 0,
        'skill_matches': {},
        'missing_skills': [],
        'experience_match': 0,
        'education_match': 0,
        'language_match': 0
    }
    
    # Match skills
    total_skill_weight = 0
    skill_match_score = 0
    
    if 'subsets' in job_requirements:
        for subset_name, subset_info in job_requirements['subsets'].items():
            weight = subset_info.get('weight', 1)
            total_skill_weight += weight
            
            # Extract keywords for this subset
            keywords = subset_info.get('keywords', [])
            
            # Calculate matches for this subset
            subset_matches = []
            subset_missing = []
            
            for keyword in keywords:
                # Check if keyword is in candidate skills
                if any(keyword.lower() in skill.lower() for skill in candidate_profile.get('skills', [])):
                    subset_matches.append(keyword)
                else:
                    # Check in work experience descriptions
                    keyword_in_experience = False
                    for exp in candidate_profile.get('work_experience', []):
                        if 'skills' in exp and any(keyword.lower() in skill.lower() for skill in exp['skills']):
                            subset_matches.append(keyword)
                            keyword_in_experience = True
                            break
                    
                    if not keyword_in_experience:
                        subset_missing.append(keyword)
            
            # Calculate score for this subset
            if keywords:
                subset_score = (len(subset_matches) / len(keywords)) * weight
                skill_match_score += subset_score
                
                match_result['skill_matches'][subset_name] = {
                    'matched': subset_matches,
                    'missing': subset_missing,
                    'score': subset_score / weight,  # Normalize to 0-1 range
                    'weight': weight
                }
    
    # Calculate overall skill match percentage
    if total_skill_weight > 0:
        match_result['skill_match_percentage'] = (skill_match_score / total_skill_weight) * 100
    else:
        match_result['skill_match_percentage'] = 0
    
    # Experience match - if job requires minimum years
    if 'minimum_experience' in job_requirements:
        min_experience = job_requirements['minimum_experience']
        candidate_experience = candidate_profile.get('total_years_experience', 0)
        
        if candidate_experience >= min_experience:
            match_result['experience_match'] = 1.0
        else:
            match_result['experience_match'] = candidate_experience / max(1, min_experience)
    else:
        match_result['experience_match'] = 1.0  # No specific requirement
    
    # Education match - if job requires minimum education level
    if 'minimum_education' in job_requirements:
        required_level = job_requirements['minimum_education']
        candidate_level = candidate_profile.get('education_level_score', 0)
        
        education_levels = {
            'high_school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'doctorate': 5
        }
        
        required_score = education_levels.get(required_level, 0)
        
        if candidate_level >= required_score:
            match_result['education_match'] = 1.0
        else:
            match_result['education_match'] = candidate_level / max(1, required_score)
    else:
        match_result['education_match'] = 1.0  # No specific requirement
    
    # Language match - if job requires specific languages
    if 'required_languages' in job_requirements:
        language_matches = 0
        required_languages = job_requirements['required_languages']
        
        for req_lang in required_languages:
            lang_name = req_lang['language']
            min_level = req_lang.get('min_level', 0)
            
            # Find this language in candidate languages
            for cand_lang in candidate_profile.get('languages', []):
                if cand_lang['language'].lower() == lang_name.lower():
                    if cand_lang.get('score', 0) >= min_level:
                        language_matches += 1
                    break
        
        if required_languages:
            match_result['language_match'] = language_matches / len(required_languages)
        else:
            match_result['language_match'] = 1.0
    else:
        match_result['language_match'] = 1.0  # No specific requirement
    
    # Calculate overall match
    match_components = [
        match_result['skill_match_percentage'] / 100 * 0.6,  # Skills are 60% of overall match
        match_result['experience_match'] * 0.2,              # Experience is 20%
        match_result['education_match'] * 0.1,               # Education is 10%
        match_result['language_match'] * 0.1                 # Language is 10%
    ]
    
    match_result['overall_match'] = sum(match_components) * 100  # Convert to percentage
    match_result['match_components'] = {
        'skills': match_result['skill_match_percentage'] / 100 * 0.6 * 100,
        'experience': match_result['experience_match'] * 0.2 * 100,
        'education': match_result['education_match'] * 0.1 * 100,
        'language': match_result['language_match'] * 0.1 * 100
    }
    
    return match_result