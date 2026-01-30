import streamlit as st
import pandas as pd
import os
import warnings
# Suppress deprecation warnings from google.generativeai
warnings.simplefilter(action='ignore', category=FutureWarning)

from dotenv import load_dotenv
from pypdf import PdfReader
from matcher import ResumeMatcher
from scraper import scrape_job_boards, scrape_social_posts

# Load environment variables
load_dotenv()

st.set_page_config(page_title="JobSniper AI", layout="wide", page_icon="🎯")

# Initialize Session State
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'suggested_roles' not in st.session_state:
    st.session_state.suggested_roles = []
if 'job_results' not in st.session_state:
    st.session_state.job_results = pd.DataFrame()

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return ""

def main():
    st.title("🎯 JobSniper AI")
    st.markdown("Automated Agentic Job Finder & Matcher")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        
        # API Key handling
        env_api_key = os.getenv("GEMINI_API_KEY")
        api_key = st.text_input("Gemini API Key", value=env_api_key if env_api_key else "", type="password")
        
        st.divider()
        
        # Job Search Params
        job_roles_input = st.text_input("Job Roles (comma-separated)", value="Machine Learning Engineer, Generative AI Engineer, Data Scientist, MLOps Engineer, Software Engineer")
        location = st.text_input("Location", value="Remote")
        
        # Enhanced slider (User requested more results)
        num_jobs = st.slider("Jobs to Scrape (per role)", 2, 500, 20)
        hours_old = st.number_input("Jobs posted within (hours)", min_value=1, value=72, help="Filter jobs by how long ago they were posted.")
        fetch_full_desc = st.toggle("Fetch Full LinkedIn Description", value=True, help="Slower but gets more details for LinkedIn jobs.")
        remove_duplicates = st.toggle("Remove Duplicates", value=True, help="Removes jobs with same Title, Company, and Location.")
        
        # Experience Settings
        if 'resume_years' not in st.session_state:
            st.session_state.resume_years = 0
            
        resume_years = st.number_input("Your Experience (Years)", min_value=0, max_value=50, value=st.session_state.resume_years)
        st.session_state.resume_years = resume_years
        
        enable_strict_filter = st.toggle("Strict Experience Filter", value=True, help="Filters out jobs requiring > (Your Years + 1)")
        enable_deep_analysis = st.toggle("Enable Deep AI Analysis (Layer 2)", value=False)
        st.caption("Layer 2 uses Gemini Flash (Slow due to rate limits)")

    # --- Main Content ---
    
    # 1. Resume Upload
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    
    if uploaded_file:
        if st.session_state.resume_text == "":
            with st.spinner("Parsing resume..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.resume_text = text
                st.success("Resume parsed!")

    st.divider()

    st.divider()

    # 3. Job Search & Matching
    st.subheader("2. Search & Match")
    
    if st.button("Find High-Fit Jobs"):
        if not job_roles_input:
            st.error("Please enter at least one job role.")
            return

        status_text = st.empty()
        status_text.text("Scraping job boards...")
        
        roles = [r.strip() for r in job_roles_input.split(',') if r.strip()]
        
        all_dfs = []
        progress_bar = st.progress(0)
        
        # Scrape loop
        try:
            for i, role in enumerate(roles):
                status_text.text(f"Scraping for '{role}'...")
                
                df_boards = scrape_job_boards(role, location, num_jobs, hours_old, linkedin_fetch_description=fetch_full_desc)
                df_social = scrape_social_posts(role, location)
                
                if not df_boards.empty:
                    df_boards['Search_Term'] = role
                    all_dfs.append(df_boards)
                if not df_social.empty:
                    df_social['Search_Term'] = role
                    all_dfs.append(df_social)
                
                progress_bar.progress((i + 1) / len(roles))

            # Combine
            if not all_dfs:
                st.warning("No jobs found. Try broadening your criteria.")
                st.session_state.job_results = pd.DataFrame()
                return

            df = pd.concat(all_dfs, ignore_index=True)
            status_text.text("Processing results...")
            
            # Remove Duplicates
            if remove_duplicates:
                before_dedup = len(df)
                df = df.drop_duplicates(subset=['title', 'company', 'location'], keep='first')
                after_dedup = len(df)
                if before_dedup > after_dedup:
                    st.toast(f"Removed {before_dedup - after_dedup} duplicate jobs.")
            
            st.write(f"Found {len(df)} unique jobs. Running Layer 1 (Vector Match)...")
            
            # Layer 1: Local Vector Match + Guardrails
            matcher = ResumeMatcher()
            df['description'] = df['description'].fillna('')
            
            # Extract experience for ALL jobs first (for visibility and filtering)
            df['Min_Years_Req'] = df['description'].apply(lambda x: matcher.extract_job_experience_requirement(x))
            
            # 1.1 Experience Filter (Guardrail)
            if enable_strict_filter:
                initial_count = len(df)
                # Filter logic using the pre-calculated column
                # Gap is 1 year: Req <= User + 1
                # Also pass if Req == 0 (not found)
                df = df[ (df['Min_Years_Req'] == 0) | (df['Min_Years_Req'] <= (st.session_state.resume_years + 1)) ]
                
                filtered_count = len(df)
                if initial_count > filtered_count:
                    st.caption(f"Filtered out {initial_count - filtered_count} jobs due to experience mismatch.")

            # 1.2 Vector Score
            df['Vector_Score'] = df['description'].apply(
                lambda x: matcher.get_embedding_score(st.session_state.resume_text, x)
            )
            
            # Sort by Vector Score
            df = df.sort_values(by='Vector_Score', ascending=False)
            
            # Layer 2: Deep Analysis (Optional)
            if enable_deep_analysis and api_key:
                status_text.text("Running Layer 2: Gemini Analysis on Top 20 jobs...")
                progress_bar = st.progress(0)
                
                # Initialize new columns
                df['AI_Match_Score'] = 0
                df['Missing_Skills'] = ""
                df['Reasoning'] = ""
                
                # Analyze only top 20
                subset = df.head(20)
                total_to_analyze = len(subset)
                
                for i, (index, row) in enumerate(subset.iterrows()):
                    res = matcher.get_gemini_analysis(st.session_state.resume_text, row['description'], api_key)
                    
                    df.at[index, 'AI_Match_Score'] = res.get('match_percentage', 0)
                    df.at[index, 'Missing_Skills'] = ", ".join(res.get('missing_skills', []))
                    df.at[index, 'Reasoning'] = res.get('reasoning', '')
                    
                    progress_bar.progress((i + 1) / total_to_analyze)
                
                # Sort by AI Score first, then Vector Score
                df = df.sort_values(by=['AI_Match_Score', 'Vector_Score'], ascending=False)
            
            st.session_state.job_results = df
            status_text.text("Done!")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 4. Display Results
    if not st.session_state.job_results.empty:
        st.subheader("Top Opportunities")
        
        display_cols = ['company', 'title', 'location', 'Vector_Score', 'Min_Years_Req', 'job_url']
        if 'AI_Match_Score' in st.session_state.job_results.columns:
            display_cols.insert(0, 'AI_Match_Score')
            display_cols.append('Missing_Skills')
        
        if 'Min_Years_Req' not in st.session_state.job_results.columns:
             # Backward compatibility / Safety: Recalculate if missing
             if 'description' in st.session_state.job_results.columns:
                 matcher = ResumeMatcher()
                 st.session_state.job_results['Min_Years_Req'] = st.session_state.job_results['description'].apply(lambda x: matcher.extract_job_experience_requirement(x) if x else 0)
             else:
                 st.session_state.job_results['Min_Years_Req'] = 0

        st.dataframe(
            st.session_state.job_results[display_cols],
            column_config={
                "job_url": st.column_config.LinkColumn("Link"),
                "Vector_Score": st.column_config.ProgressColumn("Vector Score", format="%d%%", min_value=0, max_value=100),
                "Min_Years_Req": st.column_config.NumberColumn("Min Exp (Yrs)", format="%d"),
                "AI_Match_Score": st.column_config.NumberColumn("Gemini Score", format="%d%%")
            },
            hide_index=True
        )
        
        # Export
        csv = st.session_state.job_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "jobs_with_scores.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()
