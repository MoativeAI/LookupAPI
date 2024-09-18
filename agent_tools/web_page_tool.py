from langchain.tools import BaseTool
from bs4 import BeautifulSoup
import cloudscraper
import time
from pydantic import Field

class WebPageTool(BaseTool):
    name: str = Field(default="get_webpage")
    description: str = Field(default="Useful for when you need to get the content from a specific webpage")

    def _run(self, webpage: str):
        # The rest of your _run method remains unchanged
        scraper = cloudscraper.create_scraper(browser='chrome')

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = scraper.get(webpage)
                response.raise_for_status()
                html_content = response.text
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Failed to fetch the webpage after {max_retries} attempts. Error: {str(e)}"
                time.sleep(5)

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text(separator=' ', strip=True)
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")