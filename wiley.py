import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from wiley_tdm import TDMClient



# system_prompt = """
# You are a helpful AI assistant for scholarly search. 
# You DO NOT fetch data from APIs directly — that is handled by the Python code. 
# Instead, your responsibilities are:

# 1. After the system fetches results from Wiley API (DOIs or metadata), translate the doi into a request to the Crossref API search endpoint.
# 4. If the system provides you with Wiley API output for a DOI, help summarize or highlight key 
#    points from the full text.
# 5. If no full text is available, explain this to the user in a supportive and encouraging way.

# Important guidelines:
# - Do not fabricate DOIs, metadata, or full text. Always rely on the data provided by the system.
# - Keep explanations user-friendly and not overly technical.
# - Encourage the user to further explore the topic if access is limited.
# """

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
wiley_client_token = os.getenv("Wiley-TDM-Client-Token")
tdm = TDMClient()

def get_doi_from_crossref(query: str) -> str | None:
    url = (
        f"https://api.crossref.org/works"
        f"?query={query}&rows=1&filter=publisher-name:Wiley"
    )
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        print(data)
        items = data.get("message", {}).get("items", [])
        if items:
            print(items[0].get("DOI"))
            return items[0].get("DOI")
    return None

# def check_wiley_fulltext(doi: str, token: str) -> str:
#     url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
#     print(f"Checking Wiley full text for URL: {url}")
#     headers = {"Wiley-TDM-Client-Token" : token}
#     resp = requests.get(url, headers=headers)

#     if resp.status_code == 200:
#         return resp.text  # full text (XML usually)
#     elif resp.status_code == 404:
#         return "No full text available for this DOI in Wiley."
#     else:
#         return f"Wiley API error: {resp.status_code}"

def get_wiley_fulltext(doi: str) -> str:
    """Fetch the full text from Wiley using the TDM API."""
    try:
        response = tdm.download_pdf(doi)
        print("Downloaded", doi)
    except Exception as e:
        print(f"Error fetching Wiley full text: {e}")
        return "Error fetching full text."

def ask_wiley_llm(query: str) -> str:
    """Send a query to Gemini and return the response text."""
    messages = [
        types.Content(role="user", parts=[types.Part(text=query)]),
    ]
    doi = get_doi_from_crossref(query)
    if doi:
        wiley_fulltext = get_wiley_fulltext(doi)
        if wiley_fulltext:
            messages.append(types.Content(role="user", parts=[types.Part(text=wiley_fulltext)]))
    return messages
    # response = client.models.generate_content(
    #     model='gemini-2.0-flash-001',
    #     contents=messages,
    #     config=types.GenerateContentConfig(system_instruction=system_prompt),
    # )

    # return response.text


def run():

    print("🔎 Wiley Search Agent (type 'exit' to quit)\n")
    
   
    while True:
        user_input = input("\nYour query: ")
        if user_input.lower().strip() == "exit":
            print("See you next time!")
            break

        doi = get_doi_from_crossref(user_input)
        if doi:
            wiley_fulltext = get_wiley_fulltext(doi)
        print("\n--- Results ---")
        print("Done")



if __name__ == "__main__":
    run()
