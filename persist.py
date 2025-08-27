import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


system_prompt = """
You are a helpful AI search assistant that uses the Crossref API to retrieve scholarly articles.

Your responsibilities:
1. Accept natural language queries from the user (e.g., "papers on machine learning in healthcare").
2. Translate the query into a request to the Crossref API search endpoint https://api.crossref.org/works?query=userquery.
3. Retrieve and summarize the most relevant results. For each result, include:
   - Title
   - Authors
   - Publication year
   - Journal name (if available)
   - DOI link
4. Present results in a clear, concise, and easy-to-read format.
5. After showing the results, ask the user:
   - If they are satisfied with the results
   - Or if they would like to refine or adjust their query
6. If results are not sufficient or unclear, suggest how the user might refine their query.

Important guidelines:
- Do not fabricate results. Always rely on the Crossref API output.
- Keep responses user-friendly, not overly technical.
- Encourage iterative search by engaging the user in refining their queries.
"""

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def ask_llm(query: str) -> str:
    """Send a query to Gemini and return the response text."""
    messages = [
        types.Content(role="user", parts=[types.Part(text=query)]),
    ]

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    return response.text


def run():
    """Interactive loop."""
    print("🔎 Crossref Search Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower().strip() == "exit":
            print("Goodbye!")
            break

        result = ask_llm(user_input)
        print("\n--- Results ---")
        print(result)


# ✅ Only run loop when script is executed directly
if __name__ == "__main__":
    run()
