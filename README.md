# JobSniper AI

Automated job finder using local AI matching and Gemini Flash analysis.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
1.  **Get an API Key**: obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
2.  **Environment Variables**:
    - Rename `.env.example` to `.env` (create one if it doesn't exist).
    - Add: `GEMINI_API_KEY=your_key_here`
    - **Important**: Never commit your `.env` file to version control. The `.gitignore` file is set up to prevent this.

## Technologies
- **Streamlit**: UI
- **Python-JobSpy**: Job Board Scraping
- **DuckDuckGo Search**: Social Media Scraping
- **Sentence-Transformers**: Local Vector Embedding
- **Google Gemini**: Resume Analysis & Skill Matching

## Usage

Run the application:
```bash
streamlit run app.py
```
