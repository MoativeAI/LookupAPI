from langchain.tools import BaseTool
import cloudscraper
from bs4 import BeautifulSoup
import time
import certifi
from pydantic import Field

class MetaDescriptionTool(BaseTool):
    name: str = Field(default="meta_description_tool")
    description: str = Field(default="Extracts the meta description from a given URL.")

    def _run(self, url: str) -> str:
        start_time = time.time()
        scraper = cloudscraper.create_scraper(browser='chrome')

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = scraper.get(url, verify=certifi.where())
                response.raise_for_status()  # Raises an HTTPError for bad responses

                soup = BeautifulSoup(response.text, 'html.parser')
                meta_description = soup.find('meta', attrs={'name': 'description'})

                if meta_description and 'content' in meta_description.attrs:
                    return meta_description['content']
                else:
                    # If no meta description, try to get the first paragraph or title
                    first_p = soup.find('p')
                    title = soup.find('title')

                    if first_p:
                        return first_p.get_text(strip=True)[:200] + "..."  # Truncate if too long
                    elif title:
                        return title.get_text(strip=True)
                    else:
                        return "No meta description or suitable alternative content found."

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error fetching meta description after {max_retries} attempts: {str(e)}"
                time.sleep(5)  # Wait for 5 seconds before retrying

    async def _arun(self, url: str) -> str:
        raise NotImplementedError("This tool does not support async")