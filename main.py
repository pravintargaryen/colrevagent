import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.get_files_info import schema_get_files_info
from functions.get_files_content import get_file_content, schema_get_file_content
from functions.write_file  import schema_write_file
from functions.run_python_file import schema_run_python_file
from call_function import call_function

if len(sys.argv) < 2:
    print("Usage: python script.py <your prompt>")
    sys.exit(1)

verbose = False
user_prompt = " ".join(sys.argv[1:])



system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""

messages = [
    types.Content(role="user", parts=[types.Part(text=user_prompt)]),
]

available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_get_files_content,
        schema_write_file,
        schema_run_python_file
    ]
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents=messages, 
    config=types.GenerateContentConfig(tools=[available_functions], system_instruction=system_prompt),
)

function_call_found = False
for candidate in response.candidates:
    for part in candidate.content.parts:
        if hasattr(part, "function_call") and part.function_call:
            function_call_found = True
            print(f"Calling function: {part.function_call.name}({part.function_call.args})")
             # Actually call the function via Gemini API
            function_call_result = client.models.call_function(
                name=part.function_call.name,
                arguments=part.function_call.args
            )
            # Ensure function_response exists
            if (
                not function_call_result.parts
                or not hasattr(function_call_result.parts[0], "function_response")
                or not hasattr(function_call_result.parts[0].function_response, "response")
                or function_call_result.parts[0].function_response.response is None
            ):
                raise RuntimeError("Fatal: Function call did not return a valid function_response.response")

            # Print function output if verbose
            if verbose:
                print(f"-> {function_call_result.parts[0].function_response.response}")

if not function_call_found:
    print(response.text)


if verbose:
    print('prompt_tokens', response.usage_metadata.prompt_token_count)
    print('response_tokens', response.usage_metadata.candidates_token_count)
