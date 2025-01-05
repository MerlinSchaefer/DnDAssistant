from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.llm_qa import LLMJudge
from src.llm.deployments import AvailableChatModels


@pytest.fixture
def mock_llm_judge():
    """Fixture to create a mock LLMJudge instance with a mocked LLM."""
    judge = LLMJudge(model_name=AvailableChatModels.LLAMA_3_2)
    judge.llm = MagicMock()  # mock the LLM to avoid actual API calls
    # TODO: implement integration tests for the LLMJudge class
    return judge


def test_generate_response(mock_llm_judge):
    """Test generating a response from the LLM."""
    mock_llm_judge.llm.invoke.return_value.content = "This is a mock response."
    response = mock_llm_judge.generate_response("What is useEffect?")
    assert response == "This is a mock response."


def test_calculate_match_score(mock_llm_judge):
    """Test the embedding-based match score calculation."""
    ground_truth = "The quick brown fox jumps over the lazy dog."
    model_response = "A quick brown fox jumped over a lazy dog."
    score = mock_llm_judge.calculate_match_score(ground_truth, model_response)
    assert 0.9 < score <= 1.0  # Expect high similarity for semantically similar sentences


def test_evaluate_response(mock_llm_judge):
    """Test evaluating a response with the LLM-as-a-judge."""
    mock_llm_judge.llm.invoke.return_value.content = """
    1. Overall Score: 8/10
    2. Sub-scores:
       - Relevance: 9/10
       - Accuracy: 8/10
       - Completeness: 7/10
       - Ground Truth Match: 8/10
    3. Justification: The response is accurate and relevant but could include more details.
    """
    input_text = "What is useEffect?"
    ground_truth = "A React hook for performing side effects."
    model_response = "useEffect is used for side effects in React."
    response = mock_llm_judge.evaluate_response(input_text, ground_truth, model_response)
    assert "Overall Score: 8/10" in response


def test_parse_llm_response(mock_llm_judge):
    """Test parsing a structured response from the LLM."""
    structured_response = """
    1. Overall Score: 8/10
    2. Sub-scores:
       - Relevance: 9/10
       - Accuracy: 8/10
       - Completeness: 7/10
       - Ground Truth Match: 8/10
    3. Justification: The response is accurate and relevant but could include more details.
    """
    parsed = mock_llm_judge.parse_llm_response(structured_response)
    assert parsed["overall_score"] == 8.0
    assert parsed["relevance"] == 9.0
    assert parsed["accuracy"] == 8.0
    assert parsed["completeness"] == 7.0
    assert parsed["ground_truth_match"] == 8.0
    assert "accurate and relevant" in parsed["justification"]


def test_parse_llm_response_missing_fields(mock_llm_judge):
    """Test parsing with missing fields in the response."""
    structured_response = """
    1. Overall Score: 8/10
    3. Justification: Missing sub-scores but overall good response.
    """
    parsed = mock_llm_judge.parse_llm_response(structured_response)
    assert parsed["overall_score"] == 8.0
    assert np.isnan(parsed["relevance"])
    assert np.isnan(parsed["accuracy"])
    assert np.isnan(parsed["completeness"])
    assert np.isnan(parsed["ground_truth_match"])
    assert "overall good response" in parsed["justification"]
