import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from wiley_tdm import TDMClient
from google.adk.agents import Agent



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
wiley_client_token = os.getenv("Wiley-TDM-Client-Token")
tdm = TDMClient()


def get_wiley_fulltext() -> dict:
    """Fetches full texts from Wiley using the TDM API for predefined DOIs.

    Returns:
        dict: status and list of downloaded DOIs or error message.
    """
    dois = [
        '10.1002/ams2.70086',
        '10.1002/ams2.70082',
    ]

    downloaded = []
    for doi in dois:
        try:
            response = tdm.download_pdf(doi)  # Assuming tdm is already defined elsewhere
            print("Downloaded", doi)
            downloaded.append(doi)
        except Exception as e:
            error_msg = f"Error fetching full text for DOI {doi}: {e}"
            print(error_msg)
            return {
                "status": "error",
                "error_message": error_msg
            }

    return {
        "status": "success",
        "downloaded_dois": downloaded
    }




root_agent = Agent(
    name="wiley_tdm_agent",
    model="gemini-2.0-flash",
    description="Agent to fetch full texts from Wiley using DOIs via the TDM API.",
    instruction="You are a helpful agent that can fetch full texts of scientific papers given predefined DOIs using Wiley's TDM API.",
    tools=[get_wiley_fulltext],
)