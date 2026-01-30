import json
import time
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

class ResumeMatcher:
    def __init__(self):
        # Load local embedding model (fast, runs on CPU)
        try:
            self.model = SentenceTransformer('all-mpnet-base-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None

    def suggest_roles(self, resume_text, api_key):
        """
        Uses Gemini to analyze the resume and infer the top 3 best-fit job titles AND years of experience.
        Returns a dict: {'roles': [], 'years_of_experience': int}
        """
        default_return = {'roles': ["Python Developer", "Data Analyst", "Software Engineer"], 'years_of_experience': 0}
        
        if not api_key:
            return default_return

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze the following resume text.
            1. Suggest the top 3 best-fit job titles.
            2. Extract the TOTAL years of relevant professional experience (estimated integer).
            
            Return ONLY a JSON object with keys: "roles" (list of strings) and "years_of_experience" (integer).
            Do not include any markdown formatting or extra text.
            
            Resume Text:
            {resume_text[:4000]}
            """
            
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Clean up potential markdown code blocks
            if text.startswith("```"):
                text = text.strip("`json \n")
            
            data = json.loads(text)
            return data
        except Exception as e:
            print(f"Error in suggest_roles: {e}")
            return default_return

    def extract_job_experience_requirement(self, job_description):
        """
        Extracts minimum years of experience required from the job description using Regex.
        Returns int (0 if not found).
        """
        try:
            # Patterns looking for strict experience requirements
            # We prioritize patterns that explicitly mention "experience" or "work" to avoid company age etc.
            # Handles: "5 years experience", "5yrs exp", "5+ years", "at least 5 years"
            # We allow optional words: "of", "relevant", "professional", "work", "industry"
            middleware = r'(?:\s*(?:of|relevant|professional|work|industry)\s*)*'
            
            patterns = [
                r'(\d+)\+?\s*(?:years?|yrs?)' + middleware + r'(?:experience|exp)?',
                r'(\d+)\s*-\s*\d+\s*(?:years?|yrs?)' + middleware + r'(?:experience|exp)?',
                r'at least\s*(\d+)\s*(?:years?|yrs?)',
                r'minimum\s*(\d+)\s*(?:years?|yrs?)',
                r'more than\s*(\d+)\s*(?:years?|yrs?)',
                r'(\d+)\+?\s*(?:years?|yrs?)\s+', # aggressive: "5+ years" followed by space (likely start of sentence)
            ]
            
            min_years = 0
            
            # Simple heuristic: scan for patterns and take the one that looks like a requirement
            # We look for small numbers (e.g. 1-15) usually associated with "experience"
            matches = []
            for pat in patterns:
                found = re.findall(pat, job_description, re.IGNORECASE)
                for f in found:
                    try:
                        val = int(f)
                        if 0 < val < 20: # Sanity check
                            matches.append(val)
                    except:
                        pass
            
            if matches:
                 # If multiple matches (e.g. "5 years Python", "3 years SQL"), we take the MAX.
                 # Rationale: If a job requires 5 years of ANYTHING, and you have 1 year, 
                 # you likely won't pass the strict filter. 
                 # This acts as a safer guardrail for "High Fit".
                 return max(matches) 
            return 0
        except Exception:
            return 0

    def is_experience_match(self, resume_years, job_description):
        """
        Returns True if the job's required experience is within the acceptable gap.
        Rule: Job Required <= Resume Years + 1.
        """
        if resume_years == 0:
             return True # If we couldn't parse resume years, don't filter.
             
        job_req = self.extract_job_experience_requirement(job_description)
        if job_req == 0:
            return True # If we couldn't parse job req, don't filter.
            
        # user said: "upto one year experience gap is okay"
        # Meaning: If I have 3 years, I can apply for jobs requiring 4 years.
        # So Req <= MyYears + 1
        return job_req <= (resume_years + 1)

    def get_embedding_score(self, resume_text, job_description):
        """
        Uses SentenceTransformer to get a quick 0-100% score.
        """
        if not self.model or not job_description:
            return 0.0

        try:
            # Compute embeddings
            resume_emb = self.model.encode(resume_text, convert_to_tensor=True)
            job_emb = self.model.encode(job_description, convert_to_tensor=True)

            # Compute cosine similarity
            score = util.cos_sim(resume_emb, job_emb).item()
            return round(score * 100, 2)
        except Exception as e:
            print(f"Error in get_embedding_score: {e}")
            return 0.0

    def get_gemini_analysis(self, resume_text, job_description, api_key):
        """
        Uses Gemini to analyze the resume vs job description.
        Returns a dictionary with match_percentage, missing_skills, reasoning.
        """
        if not api_key:
            return {"match_percentage": 0, "missing_skills": [], "reasoning": "API Key missing"}

        # Rate Limiting: 4 second delay to stay under 15 RPM
        time.sleep(4)

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            prompt = f"""
            Act as a recruiter. Compare this resume to the job description below.
            Return a valid JSON object with the following keys:
            - "match_percentage": integer between 0 and 100
            - "missing_skills": list of strings
            - "reasoning": short summary string

            Resume:
            {resume_text[:3000]}

            Job Description:
            {job_description[:3000]}
            """

            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up potential markdown
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            return json.loads(text)
        except Exception as e:
            print(f"Error in get_gemini_analysis: {e}")
            return {"match_percentage": 0, "missing_skills": ["Error analyzing"], "reasoning": str(e)}
