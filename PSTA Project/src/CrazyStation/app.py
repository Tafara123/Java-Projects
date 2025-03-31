import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import time
import re
from datetime import datetime

# Import original helper functions
from resources.helper import (
    get_pdf_files, get_application_id, get_candidate_application_pdf_data, 
    get_bmw_file_data, read_pdf_contents, extract_work_experience, 
    extract_education_experience, extract_language_knowledge
)

# Setup page config
st.set_page_config(
    page_title="HR Candidate Scoring System",
    page_icon="👨‍💼",
    layout="wide"
)

# Ensure directories exist
for directory in ["data/151304", "models", "config", "temp"]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize session state
if "job_profiles" not in st.session_state:
    # Load job profiles
    profiles_path = Path("config/job_profiles.json")
    if profiles_path.exists():
        with open(profiles_path, 'r') as f:
            st.session_state.job_profiles = json.load(f)
    else:
        # Create default job profiles
        st.session_state.job_profiles = {
            "Communication Specialist": {
                "description": "Communication specialist for technical documentation",
                "subsets": {
                    "Knowledge (Communication Science)": {
                        "weight": 5,
                        "keywords": [
                            "communication science", "media studies", "journalism", 
                            "public relations", "advertising", "Kommunikation"
                        ]
                    },
                    "Knowledge (Project Management)": {
                        "weight": 3,
                        "keywords": [
                            "project management", "complex processes", "agile", 
                            "scrum", "kanban", "stakeholder management"
                        ]
                    },
                    "Knowledge (Python)": {
                        "weight": 3,
                        "keywords": [
                            "Python", "pandas", "data analysis", "matplotlib", 
                            "numpy", "scikit-learn", "jupyter"
                        ]
                    }
                }
            }
        }
        # Save default profiles
        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profiles_path, 'w') as f:
            json.dump(st.session_state.job_profiles, f, indent=2)

if "selected_candidates" not in st.session_state:
    st.session_state.selected_candidates = []
if "current_job_profile" not in st.session_state:
    st.session_state.current_job_profile = None
if "scoring_results" not in st.session_state:
    st.session_state.scoring_results = None
if "tab_index" not in st.session_state:
    st.session_state.tab_index = 0

# Define helper functions
def list_candidates():
    """List all available candidates"""
    try:
        # Get all PDF files
        pdf_files = get_pdf_files("data/151304")
        
        # Extract unique candidate IDs
        candidate_ids = list(set([get_application_id(application) for application in pdf_files]))
        
        # Get basic info for each candidate
        candidates = []
        for candidate_id in candidate_ids:
            try:
                bmw_file, candidate_files = get_candidate_application_pdf_data(candidate_id, "data/151304")
                if bmw_file:
                    basic_data = get_bmw_file_data(bmw_file)
                    if basic_data and 'Candidate Name' in basic_data:
                        candidates.append({
                            'id': candidate_id,
                            'name': basic_data['Candidate Name'],
                            'email': basic_data.get('Email', ''),
                            'country': basic_data.get('Country', '').replace('des Wohnsitzes', ''),
                            'file_count': len(candidate_files) + 1
                        })
            except Exception as e:
                st.error(f"Error getting info for candidate {candidate_id}: {str(e)}")
                
        return candidates
        
    except Exception as e:
        st.error(f"Error listing candidates: {str(e)}")
        return []

def save_job_profile(name, profile_data):
    """Save a job profile"""
    st.session_state.job_profiles[name] = profile_data
    
    # Save to file
    profiles_path = Path("config/job_profiles.json")
    with open(profiles_path, 'w') as f:
        json.dump(st.session_state.job_profiles, f, indent=2)
    
    return True

def extract_skills_from_text(text):
    """Extract skills from text using pattern matching"""
    skills = []
    text_lower = text.lower()
    
    # Define skill categories
    skill_categories = {
        "programming": ["python", "java", "javascript", "c++", "sql", "php", "html", "css"],
        "data_science": ["machine learning", "data analysis", "statistics", "pandas", "numpy"],
        "project_management": ["scrum", "agile", "kanban", "jira", "projektmanagement"],
        "languages": ["deutsch", "englisch", "französisch", "spanisch", "italienisch"],
        "soft_skills": ["teamwork", "communication", "leadership", "problem solving"]
    }
    
    # Extract skills from categories
    for category, category_skills in skill_categories.items():
        for skill in category_skills:
            if skill in text_lower:
                skills.append(skill)
    
    # Look for skill sections
    skill_section_indicators = ["kenntnisse", "fähigkeiten", "skills", "kompetenzen"]
    
    for indicator in skill_section_indicators:
        if indicator in text_lower:
            pattern = rf"{indicator}[:\s]+((?:[^.;]*[,.])*)"
            matches = re.search(pattern, text_lower)
            if matches:
                skill_section = matches.group(1)
                potential_skills = re.split(r'[,;.]', skill_section)
                for skill in potential_skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2 and skill not in skills:
                        skills.append(skill)
    
    return skills

def get_candidate_profile(candidate_id):
    """Get detailed profile for a candidate"""
    try:
        # Get candidate files
        bmw_file, candidate_files = get_candidate_application_pdf_data(candidate_id, "data/151304")
        
        if not bmw_file:
            st.warning(f"No BMW file found for candidate {candidate_id}")
            return None
            
        # Get basic candidate data
        candidate_data = get_bmw_file_data(bmw_file)
        
        # Read all candidate content
        candidate_content = ""
        for file in candidate_files:
            try:
                content = read_pdf_contents(pdf_file_path=f"data/151304/{file}")
                candidate_content += content
            except Exception as e:
                st.error(f"Error reading file {file}: {str(e)}")
                
        # Extract work experience
        work_experience = extract_work_experience(candidate_content)
        
        # Add skills to work experience
        for exp in work_experience:
            job_desc = f"{exp.get('job_title', '')} {exp.get('company_name', '')} {exp.get('industry', '')}"
            exp['skills'] = extract_skills_from_text(job_desc)
        
        # Extract education
        education = extract_education_experience(candidate_content)
        
        # Extract languages
        languages = extract_language_knowledge(candidate_content)
        
        # Extract overall skills
        skills = extract_skills_from_text(candidate_content)
        
        # Calculate total experience years
        total_years = 0
        for exp in work_experience:
            try:
                if 'start_date' in exp and 'end_date' in exp:
                    start_date = datetime.strptime(exp['start_date'], '%d.%m.%Y')
                    if exp['end_date'] == 'TT.MM.JJJJ':
                        end_date = datetime.now()
                    else:
                        end_date = datetime.strptime(exp['end_date'], '%d.%m.%Y')
                    
                    years = (end_date - start_date).days / 365.25
                    if 0 <= years <= 50:  # Sanity check
                        total_years += years
            except:
                continue
        
        # Create profile
        profile = {
            'work_experience': work_experience,
            'education': education,
            'languages': languages,
            'skills': skills,
            'total_years_experience': round(total_years, 1)
        }
        
        # Combine with basic data
        combined_data = {**candidate_data, **profile}
        
        return combined_data
        
    except Exception as e:
        st.error(f"Error getting candidate profile: {str(e)}")
        return None

def match_candidate_to_job(candidate_profile, job_requirements):
    """Match candidate profile to job requirements"""
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
    
    # Experience match - default to 100% if no minimum specified
    match_result['experience_match'] = 1.0
    
    # Education match - default to 100% if no minimum specified
    match_result['education_match'] = 1.0
    
    # Language match - if candidate has German and English, give high score
    languages = [lang['language'] for lang in candidate_profile.get('languages', [])]
    if 'Englisch' in languages and 'Deutsch' in languages:
        match_result['language_match'] = 1.0
    elif 'Englisch' in languages or 'Deutsch' in languages:
        match_result['language_match'] = 0.5
    else:
        match_result['language_match'] = 0.0
    
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

def score_candidate(candidate_id, job_profile_name):
    """Score a candidate for a specific job profile"""
    # Get candidate profile
    candidate_profile = get_candidate_profile(candidate_id)
    
    if not candidate_profile:
        st.warning(f"Could not get profile for candidate {candidate_id}")
        return None
        
    # Get job profile
    job_profile = st.session_state.job_profiles.get(job_profile_name)
    
    if not job_profile:
        st.warning(f"Job profile not found: {job_profile_name}")
        return None
        
    # Calculate match score
    match_result = match_candidate_to_job(candidate_profile, job_profile)
    
    # Combine candidate data with match results
    result = {
        'candidate_id': candidate_id,
        'candidate_name': candidate_profile.get('Candidate Name', 'Unknown'),
        'job_profile': job_profile_name,
        'overall_match': match_result['overall_match'],
        'skill_match': match_result['skill_match_percentage'],
        'experience_match': match_result['experience_match'] * 100,
        'education_match': match_result['education_match'] * 100,
        'language_match': match_result['language_match'] * 100,
        'profile': candidate_profile,
        'match_details': match_result
    }
    
    return result

def score_candidates(candidate_ids, job_profile_name, progress_callback=None):
    """Score multiple candidates for a specific job profile"""
    results = []
    total_candidates = len(candidate_ids)
    
    # Get job profile
    job_profile = st.session_state.job_profiles.get(job_profile_name)
    
    if not job_profile:
        st.warning(f"Job profile not found: {job_profile_name}")
        return pd.DataFrame(), []
        
    for i, candidate_id in enumerate(candidate_ids):
        # Update progress if callback provided
        if progress_callback:
            progress_callback(int((i / total_candidates) * 100))
            
        # Score candidate
        try:
            result = score_candidate(candidate_id, job_profile_name)
            if result:
                results.append(result)
        except Exception as e:
            st.error(f"Error scoring candidate {candidate_id}: {str(e)}")
            continue
    
    # Sort results by overall match (descending)
    results.sort(key=lambda x: x['overall_match'], reverse=True)
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame([
            {
                'Candidate ID': r['candidate_id'],
                'Candidate Name': r['candidate_name'],
                'Overall Match (%)': round(r['overall_match'], 1),
                'Skill Match (%)': round(r['skill_match'], 1),
                'Experience Match (%)': round(r['experience_match'], 1),
                'Education Match (%)': round(r['education_match'], 1),
                'Language Match (%)': round(r['language_match'], 1),
                'Years Experience': r['profile'].get('total_years_experience', 0),
            }
            for r in results
        ])
        return df, results
    else:
        return pd.DataFrame(), []

def timestamp_now():
    """Get current timestamp as a string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def export_results_to_excel(df, file_path):
    """Export results to Excel"""
    df.to_excel(file_path, index=False)
    return file_path

# Create sidebar
with st.sidebar:
    st.title("HR Candidate Scoring")
    st.markdown("---")
    
    # Job profile selection
    st.subheader("1. Select Job Profile")
    profile_names = list(st.session_state.job_profiles.keys())
    
    if profile_names:
        selected_profile = st.selectbox(
            "Choose a job profile",
            options=profile_names,
            index=0 if st.session_state.current_job_profile is None else 
                  profile_names.index(st.session_state.current_job_profile)
        )
        
        if selected_profile != st.session_state.current_job_profile:
            st.session_state.current_job_profile = selected_profile
            st.session_state.selected_candidates = []
            st.session_state.tab_index = 0
    else:
        st.warning("No job profiles found. Please create one.")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("2. Navigation")
    if st.button("🔍 Candidate Selection"):
        st.session_state.tab_index = 0
    if st.button("📊 Scoring Results"):
        st.session_state.tab_index = 1
    if st.button("⚙️ Job Profiles"):
        st.session_state.tab_index = 2
    if st.button("📋 Analytics"):
        st.session_state.tab_index = 3

# Create main content
if st.session_state.current_job_profile is None:
    st.title("Welcome to HR Candidate Scoring System")
    st.markdown("""
    ## Intelligent Candidate Evaluation
    
    This system helps HR professionals evaluate candidates efficiently and objectively.
    
    **Get started by selecting a job profile from the sidebar.**
    """)
else:
    # Create tabs based on the selected tab index
    tabs = st.tabs(["Candidate Selection", "Scoring Results", "Job Profiles", "Analytics"])
    tab = tabs[st.session_state.tab_index]
    
    with tabs[0]:  # Candidate Selection
        st.header("Select Candidates")
        
        # Display job profile info
        job_profile = st.session_state.job_profiles[st.session_state.current_job_profile]
        st.subheader(f"Job Profile: {st.session_state.current_job_profile}")
        st.markdown(f"**Description**: {job_profile.get('description', 'No description')}")
        
        # Get available candidates
        candidates = list_candidates()
        
        if not candidates:
            st.warning("No candidates found. Please check your data directory.")
        else:
            # Display candidate selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create a DataFrame for display
                candidates_df = pd.DataFrame(candidates)
                
                # Create a multiselect with current selection
                selected_ids = st.multiselect(
                    "Select candidates to score",
                    options=candidates_df['id'].tolist(),
                    default=st.session_state.selected_candidates,
                    format_func=lambda x: next((f"{c['name']} (ID: {c['id']})" for c in candidates if c['id'] == x), x)
                )
                
                # Update session state
                st.session_state.selected_candidates = selected_ids
                
                # Display selected count
                st.markdown(f"**{len(selected_ids)}** candidates selected")
                
                # Score button
                if st.button("Score Selected Candidates", type="primary", disabled=len(selected_ids) == 0):
                    with st.spinner(f"Scoring {len(selected_ids)} candidates..."):
                        progress_bar = st.progress(0)
                        
                        df, raw_results = score_candidates(
                            selected_ids, 
                            st.session_state.current_job_profile,
                            progress_callback=lambda p: progress_bar.progress(p/100)
                        )
                        
                        # Store results in session state
                        st.session_state.scoring_results = {
                            'df': df,
                            'raw_results': raw_results,
                            'timestamp': timestamp_now()
                        }
                        
                        if len(df) > 0:
                            # Export to Excel
                            excel_path = f"temp/candidate_scoring_{timestamp_now()}.xlsx"
                            export_results_to_excel(df, excel_path)
                            st.session_state.excel_path = excel_path
                            
                            # Switch to results tab
                            st.session_state.tab_index = 1
                            st.experimental_rerun()
                        else:
                            st.error("No candidates could be scored. Please check the logs.")
            
            with col2:
                st.markdown("### Available Candidates")
                
                # Show candidate list (paginated)
                page_size = 10
                page = st.number_input("Page", min_value=1, max_value=max(1, (len(candidates_df) + page_size - 1) // page_size), value=1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(candidates_df))
                
                for i in range(start_idx, end_idx):
                    candidate = candidates_df.iloc[i]
                    st.markdown(f"**{candidate['name']}**")
                    st.caption(f"ID: {candidate['id']} | Files: {candidate.get('file_count', 0)}")
                    
                    if candidate['id'] in selected_ids:
                        if st.button("Remove", key=f"remove_{candidate['id']}"):
                            selected_ids.remove(candidate['id'])
                            st.session_state.selected_candidates = selected_ids
                            st.experimental_rerun()
                    else:
                        if st.button("Add", key=f"add_{candidate['id']}"):
                            selected_ids.append(candidate['id'])
                            st.session_state.selected_candidates = selected_ids
                            st.experimental_rerun()
    
    with tabs[1]:  # Scoring Results
        st.header("Scoring Results")
        
        # Check if we have results
        if st.session_state.scoring_results is None:
            st.info("No scoring results available. Please select and score candidates first.")
        else:
            results = st.session_state.scoring_results
            df = results['df']
            raw_results = results['raw_results']
            
            if len(df) == 0:
                st.warning("No scored candidates found.")
            else:
                # Display filtering options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filter by threshold
                    threshold = st.slider(
                        "Show candidates with overall match ≥",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=5
                    )
                    
                    # Apply filter
                    if threshold > 0:
                        filtered_df = df[df['Overall Match (%)'] >= threshold]
                    else:
                        filtered_df = df
                
                with col2:
                    # Download Excel button
                    if 'excel_path' in st.session_state and os.path.exists(st.session_state.excel_path):
                        with open(st.session_state.excel_path, "rb") as file:
                            st.download_button(
                                label="Download Excel Report",
                                data=file,
                                file_name=f"candidate_scoring_{st.session_state.current_job_profile}_{timestamp_now()}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                # Display results
                st.subheader(f"Ranked Candidates ({len(filtered_df)} candidates)")
                
                # Create a copy for display
                display_df = filtered_df.copy()
                
                # Format percentage columns
                for col in display_df.columns:
                    if '(%)' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
                
                # Show results table
                st.dataframe(
                    display_df.sort_values('Overall Match (%)', ascending=False),
                    use_container_width=True
                )
                
                # Show visualizations
                if len(filtered_df) > 0:
                    st.subheader("Visualizations")
                    
                    # Create columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Candidate Ranking
                        fig = px.bar(
                            filtered_df.sort_values('Overall Match (%)', ascending=False).head(10),
                            y='Candidate Name',
                            x='Overall Match (%)',
                            title='Top 10 Candidates by Match Score',
                            labels={'Overall Match (%)': 'Match Score (%)'},
                            color='Overall Match (%)',
                            color_continuous_scale=px.colors.sequential.Viridis,
                            orientation='h'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Match Distribution
                        fig = px.histogram(
                            filtered_df,
                            x='Overall Match (%)',
                            nbins=10,
                            title='Distribution of Match Scores',
                            labels={'Overall Match (%)': 'Match Score (%)'},
                            color_discrete_sequence=['#3D85C6']
                        )
                        
                        # Add average line
                        avg_match = filtered_df['Overall Match (%)'].mean()
                        fig.add_vline(
                            x=avg_match,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Avg: {avg_match:.1f}%"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # Individual candidate analysis
                    st.subheader("Individual Candidate Analysis")
                    
                    candidate_select = st.selectbox(
                        "Select a candidate for detailed analysis",
                        options=filtered_df['Candidate ID'].tolist(),
                        format_func=lambda x: next((f"{r['candidate_name']}" for r in raw_results if r['candidate_id'] == x), x)
                    )
                    
                    # Get selected candidate data
                    selected_result = next((r for r in raw_results if r['candidate_id'] == candidate_select), None)
                    
                    if selected_result:
                        # Display match components
                        st.subheader(f"Match Analysis: {selected_result['candidate_name']}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Overall Match", f"{selected_result['overall_match']:.1f}%")
                        with col2:
                            st.metric("Skill Match", f"{selected_result['skill_match']:.1f}%")
                        with col3:
                            st.metric("Experience", f"{selected_result['experience_match']:.1f}%")
                        with col4:
                            st.metric("Language", f"{selected_result['language_match']:.1f}%")
                        
                        # Skill matches
                        st.subheader("Skill Matches")
                        
                        skill_matches = selected_result['match_details'].get('skill_matches', {})
                        if skill_matches:
                            # Create DataFrame
                            skill_df = pd.DataFrame([
                                {
                                    'Skill Category': name,
                                    'Match Score': details['score'] * 100,
                                    'Matched Keywords': len(details['matched']),
                                    'Missing Keywords': len(details['missing'])
                                }
                                for name, details in skill_matches.items()
                            ])
                            
                            # Create bar chart
                            fig = px.bar(
                                skill_df.sort_values('Match Score', ascending=False),
                                x='Skill Category',
                                y='Match Score',
                                color='Match Score',
                                text='Match Score',
                                labels={'Match Score': 'Match (%)'},
                                title='Skill Category Match Scores',
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig.update_layout(yaxis_range=[0, 100])
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display matched and missing keywords
                            for category, details in skill_matches.items():
                                with st.expander(f"{category} - {details['score']*100:.1f}%"):
                                    st.markdown("**Matched Keywords:**")
                                    st.markdown(", ".join(details['matched']) if details['matched'] else "None")
                                    
                                    st.markdown("**Missing Keywords:**")
                                    st.markdown(", ".join(details['missing']) if details['missing'] else "None")
                        else:
                            st.info("No detailed skill match information available.")
                        
                        # Profile overview
                        with st.expander("Profile Overview"):
                            profile = selected_result['profile']
                            
                            # Basic info
                            st.markdown("#### Basic Information")
                            st.markdown(f"**Name:** {profile.get('Candidate Name', 'N/A')}")
                            st.markdown(f"**Email:** {profile.get('Email', 'N/A')}")
                            st.markdown(f"**Phone:** {profile.get('Phone', 'N/A')}")
                            st.markdown(f"**Location:** {profile.get('Country', 'N/A')}, {profile.get('City', 'N/A')}")
                            
                            # Experience
                            st.markdown("#### Experience")
                            st.markdown(f"**Total Years:** {profile.get('total_years_experience', 0)}")
                            
                            # Work Experience
                            st.markdown("#### Experience")
                            st.markdown(f"**Total Years:** {profile.get('total_years_experience', 0)}")
                            
                            # Work Experience
                            if 'work_experience' in profile and profile['work_experience']:
                                for i, exp in enumerate(profile['work_experience']):
                                    st.markdown(f"**{exp.get('job_title', 'Role')}** at {exp.get('company_name', 'Company')}")
                                    st.markdown(f"{exp.get('start_date', '')} to {exp.get('end_date', '')}")
                                    if 'skills' in exp and exp['skills']:
                                        st.markdown(f"Skills: {', '.join(exp['skills'])}")
                            
                            # Education
                            st.markdown("#### Education")
                            if 'education' in profile and profile['education']:
                                for i, edu in enumerate(profile['education']):
                                    st.markdown(f"**{edu.get('degree', 'Degree')}** - {edu.get('institute_name', 'Institution')}")
                                    st.markdown(f"{edu.get('start_date', '')} to {edu.get('end_date', '')}")
                                    if 'degree_main_topic' in edu:
                                        st.markdown(f"Field: {edu.get('degree_main_topic', '')}")
                            
                            # Languages
                            st.markdown("#### Languages")
                            if 'languages' in profile and profile['languages']:
                                for lang in profile['languages']:
                                    st.markdown(f"**{lang.get('language', '')}**: {lang.get('level', '')}")
                            
                            # Skills
                            st.markdown("#### Skills")
                            if 'skills' in profile and profile['skills']:
                                st.markdown(", ".join(profile['skills']))
    
    with tabs[2]:  # Job Profiles
        st.header("Job Profile Management")
        
        # Create tabs for viewing and editing
        profile_tabs = st.tabs(["View Profiles", "Create/Edit Profile"])
        
        with profile_tabs[0]:  # View Profiles
            st.subheader("Available Job Profiles")
            
            # Display all profiles
            for profile_name, profile_data in st.session_state.job_profiles.items():
                with st.expander(profile_name):
                    st.markdown(f"**Description:** {profile_data.get('description', 'No description')}")
                    
                    # Display skill categories
                    if 'subsets' in profile_data:
                        st.markdown("**Skill Categories:**")
                        
                        for subset_name, subset_info in profile_data['subsets'].items():
                            st.markdown(f"* **{subset_name}** (Weight: {subset_info.get('weight', 1)})")
                            st.markdown(f"  Keywords: {', '.join(subset_info.get('keywords', []))}")
                    
                    # Actions
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("Select", key=f"select_{profile_name}"):
                            st.session_state.current_job_profile = profile_name
                            st.experimental_rerun()
                    with col2:
                        if st.button("Edit", key=f"edit_{profile_name}"):
                            st.session_state.profile_to_edit = profile_name
                            st.experimental_rerun()
                    with col3:
                        if profile_name == st.session_state.current_job_profile:
                            st.info("Currently selected")
        
        with profile_tabs[1]:  # Create/Edit Profile
            # Check if we're editing an existing profile
            is_editing = False
            profile_data = {
                "description": "",
                "subsets": {
                    "Knowledge": {
                        "weight": 5,
                        "keywords": []
                    }
                }
            }
            
            if "profile_to_edit" in st.session_state and st.session_state.profile_to_edit:
                is_editing = True
                profile_name = st.session_state.profile_to_edit
                profile_data = st.session_state.job_profiles.get(profile_name, profile_data)
                st.subheader(f"Edit Profile: {profile_name}")
            else:
                st.subheader("Create New Profile")
                profile_name = st.text_input("Profile Name")
            
            # Profile description
            description = st.text_area("Description", value=profile_data.get("description", ""))
            
            # Skill categories (subsets)
            st.markdown("### Skill Categories")
            st.caption("Define categories of skills with keywords and weights")
            
            # Get existing subsets or start with one
            subsets = profile_data.get("subsets", {})
            if not subsets:
                subsets = {"Knowledge": {"weight": 5, "keywords": []}}
            
            # Allow dynamic addition of subsets
            new_subsets = {}
            for subset_name, subset_info in subsets.items():
                with st.expander(f"{subset_name}", expanded=True):
                    # Subset name can be changed
                    new_name = st.text_input("Category Name", value=subset_name, key=f"name_{subset_name}")
                    
                    # Subset weight
                    weight = st.slider(
                        "Weight (Importance)",
                        min_value=1,
                        max_value=10,
                        value=subset_info.get("weight", 5),
                        key=f"weight_{subset_name}"
                    )
                    
                    # Keywords
                    keywords_text = st.text_area(
                        "Keywords (one per line or comma-separated)",
                        value="\n".join(subset_info.get("keywords", [])),
                        key=f"keywords_{subset_name}",
                        height=100
                    )
                    
                    # Parse keywords
                    if "," in keywords_text:
                        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
                    else:
                        keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]
                    
                    # Store in new subsets
                    new_subsets[new_name] = {
                        "weight": weight,
                        "keywords": keywords
                    }
                    
                    # Delete button
                    if st.button("Delete Category", key=f"delete_{subset_name}"):
                        # Don't include this subset
                        continue
            
            # Add new category button
            if st.button("Add Category"):
                new_category_name = f"Category {len(new_subsets) + 1}"
                new_subsets[new_category_name] = {
                    "weight": 5,
                    "keywords": []
                }
            
            # Save button
            save_col1, save_col2 = st.columns([1, 3])
            with save_col1:
                if st.button("Save Profile", type="primary", disabled=not profile_name):
                    # Create updated profile
                    updated_profile = {
                        "description": description,
                        "subsets": new_subsets
                    }
                    
                    # Save profile
                    save_job_profile(profile_name, updated_profile)
                    
                    # Update current profile if we're creating a new one
                    if not is_editing:
                        st.session_state.current_job_profile = profile_name
                    
                    # Clear edit state
                    if "profile_to_edit" in st.session_state:
                        del st.session_state.profile_to_edit
                    
                    st.success(f"Profile '{profile_name}' saved successfully!")
                    time.sleep(1)
                    st.experimental_rerun()
            
            with save_col2:
                if is_editing and st.button("Cancel Editing"):
                    if "profile_to_edit" in st.session_state:
                        del st.session_state.profile_to_edit
                    st.experimental_rerun()
    
    with tabs[3]:  # Analytics
        st.header("Analytics & Insights")
        
        # Check if we have results
        if st.session_state.scoring_results is None:
            st.info("No scoring results available. Please score candidates first.")
        else:
            results = st.session_state.scoring_results
            df = results['df']
            raw_results = results['raw_results']
            
            if len(df) == 0:
                st.warning("No scored candidates found.")
            else:
                # Display analytics dashboard
                st.subheader("Candidate Pool Analytics")
                
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_match = df['Overall Match (%)'].mean()
                    st.metric("Average Match Score", f"{avg_match:.1f}%")
                
                with col2:
                    if 'Years Experience' in df.columns:
                        avg_exp = df['Years Experience'].mean()
                        st.metric("Average Experience", f"{avg_exp:.1f} years")
                
                with col3:
                    high_matches = len(df[df['Overall Match (%)'] >= 75])
                    st.metric("High Match Candidates", f"{high_matches}/{len(df)}")
                
                # Match distribution chart
                st.subheader("Match Score Distribution")
                
                # Create bins
                bins = [0, 25, 50, 75, 100]
                labels = ['Poor (0-25%)', 'Fair (25-50%)', 'Good (50-75%)', 'Excellent (75-100%)']
                df['Match Category'] = pd.cut(df['Overall Match (%)'], bins=bins, labels=labels, right=False)
                
                # Count candidates in each bin
                category_counts = df['Match Category'].value_counts().sort_index()
                
                # Create bar chart
                fig = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    labels={'x': 'Match Category', 'y': 'Number of Candidates'},
                    title='Candidate Distribution by Match Category',
                    color=category_counts.values,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text=category_counts.values
                )
                fig.update_traces(textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Skill gap analysis
                st.subheader("Skill Gap Analysis")
                
                # Extract skill columns
                skill_columns = [col for col in df.columns if 'Match (%)' in col and col != 'Overall Match (%)']
                
                if skill_columns:
                    # Calculate average for each skill
                    skill_averages = {}
                    for col in skill_columns:
                        skill_name = col.replace(' Match (%)', '')
                        skill_averages[skill_name] = df[col].mean()
                    
                    # Create DataFrame for chart
                    skill_df = pd.DataFrame({
                        'Skill Category': list(skill_averages.keys()),
                        'Average Match (%)': list(skill_averages.values())
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        skill_df.sort_values('Average Match (%)', ascending=False),
                        x='Skill Category',
                        y='Average Match (%)',
                        title='Average Match Score by Skill Category',
                        color='Average Match (%)',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        text='Average Match (%)'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(yaxis_range=[0, 100])
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify skill gaps
                    min_skill = skill_df.loc[skill_df['Average Match (%)'].idxmin()]
                    
                    st.info(f"The biggest skill gap is in **{min_skill['Skill Category']}** with an average match of only {min_skill['Average Match (%)']:.1f}%.")
                    
                    # Get job profile keywords for this skill
                    job_profile = st.session_state.job_profiles.get(st.session_state.current_job_profile, {})
                    skill_subsets = job_profile.get('subsets', {})
                    
                    if min_skill['Skill Category'] in skill_subsets:
                        keywords = skill_subsets[min_skill['Skill Category']].get('keywords', [])
                        if keywords:
                            st.markdown(f"Keywords in this category: {', '.join(keywords)}")
                
                # Top candidates summary
                st.subheader("Top Candidates Summary")
                
                # Get top 5 candidates
                top_candidates = df.sort_values('Overall Match (%)', ascending=False).head(5)
                
                if len(top_candidates) > 0:
                    # Format DataFrame for display
                    display_df = top_candidates.copy()
                    for col in display_df.columns:
                        if '(%)' in col:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
                    
                    # Show table
                    st.dataframe(display_df[['Candidate Name', 'Overall Match (%)', 'Years Experience']], use_container_width=True)
                    
                    # Add recommendation
                    top_candidate = df.sort_values('Overall Match (%)', ascending=False).iloc[0]
                    top_match = top_candidate['Overall Match (%)']
                    
                    if top_match >= 75:
                        st.success(f"**Recommendation:** {top_candidate['Candidate Name']} is an excellent match for this position with a match score of {top_match:.1f}%.")
                    elif top_match >= 60:
                        st.info(f"**Recommendation:** {top_candidate['Candidate Name']} is a good match for this position with a match score of {top_match:.1f}%, but may require some additional training.")
                    else:
                        st.warning("**Recommendation:** None of the candidates are strong matches for this position. Consider expanding the search or adjusting the job requirements.")

# Run the app
if __name__ == "__main__":
    pass  # Streamlit runs the script automatically