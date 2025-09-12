import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

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

# Store conversation history
messages = []

print("🔎 Crossref Search Agent (type 'exit' to quit)")
while True:
    user_input = input("\nYour query: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye 👋")
        break

    # Add user message to conversation history
    messages.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

    # Send to LLM
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    # Show response
    print("\nAssistant:\n", response.text)

    # Add assistant response to history
    messages.append(types.Content(role="model", parts=[types.Part(text=response.text)]))

    # Optional: token usage
    print(
        f"[tokens used: prompt={response.usage_metadata.prompt_token_count}, "
        f"response={response.usage_metadata.candidates_token_count}]"
    )
