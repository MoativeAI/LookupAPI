from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


def get_embedding_function():
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    return embeddings