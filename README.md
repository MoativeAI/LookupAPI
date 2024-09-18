# Company Lookup API

This FastAPI-based API provides company information lookup services. It offers two endpoints for retrieving company details based on a company name.

## Features

- Retrieve comprehensive company information including social media links, contact details, and business analysis.
- Get primary company information for quick lookups.
- Uses AI-powered tools for web scraping and analysis.
- Integrates with NAICS (North American Industry Classification System) for industry classification.

## Prerequisites

- Python 3.7+
- FastAPI
- Pydantic
- python-dotenv
- LangChain
- OpenAI's GPT models
- Google Custom Search API

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   GOOGLE_API_KEY=<your-google-api-key>
   GOOGLE_CSE_ID=<your-google-custom-search-engine-id>
   OPENAI_API_KEY=<your-openai-api-key>
   ```

## Usage

To begin, run docloader.py to generate the vector database for the RAG module.

To run the API:

```
uvicorn main:app --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8001`.

### Endpoints

1. Comprehensive Company Lookup:
   ```
   GET /lookup/company/{company_name}
   ```
   This endpoint provides detailed information about a company.

2. Primary Company Lookup:
   ```
   GET /lookup/company/primary/{company_name}
   ```
   This endpoint provides basic information about a company for quick lookups.

## API Response Models

### CompanyInfo

- Company_URL: str
- Company_LinkedIn_URL: str
- Company_Facebook_URL: str
- Company_Twitter_URL: str
- Company_Phone: str
- Company_Address: str
- Meta_Description: str
- Overview: str
- USP: str
- Target_Audience: str
- Conclusion: str
- NAICS_Code: str
- Title: str
- Description: str
- Common_Labels: str
- execution_time: float

### PrimaryCompanyInfo

- Company_URL: str
- Meta_Description: str
- Overview: str
- Industry_Type: str
- execution_time: float

## Notes

- The API uses OpenAI's GPT models for analysis. Ensure you have sufficient API credits.
- Google Custom Search API is used for URL retrieval. Make sure you're within the usage limits.
- NAICS classification is done using a custom RAG (Retrieval-Augmented Generation) system.