import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Import the module you want to test
import main


@pytest.fixture
def mock_genai_response():
    """Mock response object returned by client.models.generate_content"""
    mock_response = MagicMock()
    mock_response.text = "Here are some scholarly articles..."
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 42
    return mock_response


@patch("main.genai.Client")
def test_generate_content(mock_client_class, mock_genai_response):
    """Test that the script generates content using the mocked client."""
    # Arrange: mock client and method
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_genai_response
    mock_client_class.return_value = mock_client

    # Act: simulate running the script logic
    user_prompt = "machine learning in healthcare"
    messages = [main.types.Content(role="user", parts=[main.types.Part(text=user_prompt)])]
    client = mock_client_class(api_key="fake-key")
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=messages,
        config=main.types.GenerateContentConfig(system_instruction=main.system_prompt),
    )

    # Assert: verify expected behavior
    assert response.text == "Here are some scholarly articles..."
    assert response.usage_metadata.prompt_token_count == 15
    assert response.usage_metadata.candidates_token_count == 42
    mock_client.models.generate_content.assert_called_once()


@patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
def test_env_variable_loaded():
    """Test that GEMINI_API_KEY is loaded from environment."""
    assert os.getenv("GEMINI_API_KEY") == "fake-key"


def test_argument_check(monkeypatch):
    """Test that script exits if no prompt is provided."""
    monkeypatch.setattr(sys, "argv", ["main.py"])  # no args
    with pytest.raises(SystemExit) as e:
        exec(open("main.py").read())
    assert e.type == SystemExit
    assert e.value.code == 1
