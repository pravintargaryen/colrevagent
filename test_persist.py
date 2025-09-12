import os
import pytest
from unittest.mock import patch, MagicMock
import persist  


@pytest.fixture
def mock_genai_response():
    """Mock response object returned by client.models.generate_content"""
    mock_response = MagicMock()
    mock_response.text = "Here are some scholarly articles..."
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 25
    return mock_response


@patch("persist.genai.Client")
def test_generate_content_once(mock_client_class, mock_genai_response):
    """Test that a user query generates a valid response with mocked client"""
    # Arrange
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_genai_response
    mock_client_class.return_value = mock_client

    # Simulate messages history
    messages = [
        persist.types.Content(role="user", parts=[persist.types.Part(text="Find AI papers")])
    ]

    # Act
    client = mock_client_class(api_key="fake-key")
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
        config=persist.types.GenerateContentConfig(system_instruction=persist.system_prompt),
    )

    # Assert
    assert response.text == "Here are some scholarly articles..."
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 25
    mock_client.models.generate_content.assert_called_once()


@patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
def test_env_variable_loaded():
    """Check that environment variable GEMINI_API_KEY loads correctly"""
    assert os.getenv("GEMINI_API_KEY") == "fake-key"


