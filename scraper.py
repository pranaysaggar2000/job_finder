import pandas as pd
from jobspy import scrape_jobs
from duckduckgo_search import DDGS

import requests
from bs4 import BeautifulSoup

def fetch_description_from_url(url):
    """
    Fallback: naive fetch of text from URL if description is missing.
    """
    if not url:
        return ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text (simple fallback)
            # We try to target common job board containers if possible, else body
            for tag in ['script', 'style', 'nav', 'header', 'footer']:
                 for s in soup(tag):
                     s.decompose()
            
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"Fallback fetch failed for {url}: {e}")
    return ""

def scrape_job_boards(role, location, num_jobs=10, hours_old=72, linkedin_fetch_description=True):
    """
    Scrapes jobs using python-jobspy.
    """
    try:
        # Note: 'country_indeed' is required for indeed/glassdoor.
        # 'linkedin_fetch_description' gets full description but is slower.
        jobs = scrape_jobs(
            site_name=["indeed", "glassdoor", "zip_recruiter", "linkedin"],
            search_term=role,
            location=location,
            results_wanted=num_jobs,
            hours_old=hours_old,
            country_indeed='USA',
            linkedin_fetch_description=linkedin_fetch_description
        )
        if jobs is not None and not jobs.empty:
            jobs['Source'] = 'JobBoard'
            if 'description' not in jobs.columns:
                jobs['description'] = ""
            
            # Fill missing/empty descriptions
            # We iterate to avoid making 500 requests at once if not needed.
            # Only fetch if description is missing or very short (< 50 chars)
            for index, row in jobs.iterrows():
                desc = str(row.get('description', ''))
                if not desc or len(desc) < 50:
                    url = row.get('job_url')
                    if url:
                        print(f"Fetching fallback description for: {row.get('title')}...")
                        fallback_text = fetch_description_from_url(url)
                        if fallback_text:
                            jobs.at[index, 'description'] = fallback_text[:5000] # truncate to avoid huge memory usage
                            
            jobs['description'] = jobs['description'].fillna("")
            return jobs
        return pd.DataFrame()
    except Exception as e:
        print(f"Error scraping job boards: {e}")
        return pd.DataFrame()

def scrape_social_posts(role, location):
    """
    Uses DuckDuckGo to search for LinkedIn posts.
    dork: site:linkedin.com/posts ("hiring" OR "looking for") "{role}"
    """
    results_list = []
    try:
        query = f'site:linkedin.com/posts ("hiring" OR "looking for") "{role}" "{location}"'
        
        with DDGS() as ddgs:
            # limit results to modest number to avoid blocks
            results = list(ddgs.text(query, max_results=10))
            
        for res in results:
            results_list.append({
                'title': res.get('title', 'Social Post'),
                'company': 'LinkedIn Social',
                'job_url': res.get('href'),
                'description': res.get('body', ''),
                'location': location,
                'Source': 'Social'
            })
            
        return pd.DataFrame(results_list)
    except Exception as e:
        print(f"Error scraping social posts: {e}")
        return pd.DataFrame()
