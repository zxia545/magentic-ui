"""Tests for Talent Report Agent."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from autogen_core.models import UserMessage, SystemMessage
from autogen_agentchat.messages import TextMessage

from magentic_ui.agents.talent_report import TalentReportAgent
from magentic_ui.agents.talent_report._talent_report_agent import TalentProfile


@pytest.fixture
def mock_model_client():
    """Create a mock model client."""
    client = Mock()
    client.create = AsyncMock()
    return client


@pytest.fixture
def talent_agent(mock_model_client):
    """Create a TalentReportAgent instance."""
    return TalentReportAgent(
        name="test_agent",
        model_client=mock_model_client,
        max_search_iterations=2
    )


@pytest.mark.asyncio
async def test_agent_initialization(talent_agent):
    """Test that the agent initializes correctly."""
    assert talent_agent.name == "test_agent"
    assert talent_agent._max_search_iterations == 2
    assert talent_agent._current_profile is None


@pytest.mark.asyncio
async def test_conduct_initial_search(talent_agent, mock_model_client):
    """Test initial search functionality."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = "Test search results about the talent"
    mock_model_client.create.return_value = mock_response
    
    results = await talent_agent._conduct_initial_search("Test Talent")
    
    assert len(results) > 0
    assert "Test search results" in results[0]
    assert mock_model_client.create.called


@pytest.mark.asyncio
async def test_analyze_information_gaps(talent_agent, mock_model_client):
    """Test gap analysis functionality."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = """
    - Missing publication information
    - Need current affiliation
    * Educational background unclear
    """
    mock_model_client.create.return_value = mock_response
    
    search_results = ["Some basic information about the talent"]
    gaps = await talent_agent._analyze_information_gaps("Test Talent", search_results)
    
    assert len(gaps) > 0
    assert any("publication" in gap.lower() for gap in gaps)


@pytest.mark.asyncio
async def test_structure_information(talent_agent, mock_model_client):
    """Test information structuring."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = """
    Name: Test Talent
    Position: Professor at Test University
    Expertise: Machine Learning, AI
    """
    mock_model_client.create.return_value = mock_response
    
    search_results = ["Result 1", "Result 2"]
    structured = await talent_agent._structure_information("Test Talent", search_results)
    
    assert "name" in structured
    assert structured["name"] == "Test Talent"
    assert "raw_content" in structured
    assert structured["search_results_count"] == 2


@pytest.mark.asyncio
async def test_generate_scores(talent_agent, mock_model_client):
    """Test score generation."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = """
    Research Impact: 9/10 - Extensive publications
    Expertise Level: 8/10 - Deep knowledge in field
    """
    mock_model_client.create.return_value = mock_response
    
    structured_info = {
        "name": "Test Talent",
        "raw_content": "Professional information"
    }
    scores = await talent_agent._generate_scores(structured_info)
    
    assert "research_impact" in scores
    assert "expertise_level" in scores
    assert "_raw_scoring_output" in scores


@pytest.mark.asyncio
async def test_generate_markdown_report(talent_agent, mock_model_client):
    """Test markdown report generation."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = """
# Test Talent

## Overview
A leading researcher in the field.

## Scoring Summary
| Dimension | Score |
|-----------|-------|
| Research Impact | 9/10 |
"""
    mock_model_client.create.return_value = mock_response
    
    structured_info = {"raw_content": "Info"}
    scores = {"_raw_scoring_output": "Scores"}
    
    report = await talent_agent._generate_markdown_report(
        "Test Talent", structured_info, scores
    )
    
    assert "# Test Talent" in report
    assert "## Overview" in report


@pytest.mark.asyncio
async def test_on_messages_success(talent_agent, mock_model_client):
    """Test the main message handling with successful flow."""
    # Mock all LLM responses
    mock_response = Mock()
    mock_response.content = "Mocked content"
    mock_model_client.create.return_value = mock_response
    
    messages = [TextMessage(content="Test Talent", source="user")]
    
    response = await talent_agent.on_messages(messages)
    
    assert response.chat_message is not None
    assert isinstance(response.chat_message.content, str)
    
    # Check that profile was created
    profile = talent_agent.get_current_profile()
    assert profile is not None
    assert profile.name == "Test Talent"


@pytest.mark.asyncio
async def test_on_messages_non_text_message(talent_agent):
    """Test handling of non-text messages."""
    from autogen_agentchat.messages import StopMessage
    
    messages = [StopMessage(content="stop", source="user")]
    response = await talent_agent.on_messages(messages)
    
    assert "provide a talent name" in response.chat_message.content.lower()


@pytest.mark.asyncio
async def test_on_messages_error_handling(talent_agent, mock_model_client):
    """Test error handling in message processing."""
    # Make the LLM client raise an exception
    mock_model_client.create.side_effect = Exception("API Error")
    
    messages = [TextMessage(content="Test Talent", source="user")]
    response = await talent_agent.on_messages(messages)
    
    assert "error" in response.chat_message.content.lower()


@pytest.mark.asyncio
async def test_get_current_profile_none(talent_agent):
    """Test getting profile when none exists."""
    profile = talent_agent.get_current_profile()
    assert profile is None


@pytest.mark.asyncio
async def test_reset(talent_agent):
    """Test agent reset."""
    # Set a profile
    talent_agent._current_profile = TalentProfile(
        name="Test",
        raw_search_results=[],
        structured_info={},
        scores={},
        markdown_report="",
        gaps_identified=[],
        timestamp="2024-01-01"
    )
    
    await talent_agent.on_reset()
    
    assert talent_agent._current_profile is None


@pytest.mark.asyncio
async def test_conducted_targeted_search(talent_agent, mock_model_client):
    """Test targeted search for filling gaps."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = "Additional information found"
    mock_model_client.create.return_value = mock_response
    
    gaps = ["Missing publication info", "Need current position"]
    results = await talent_agent._conduct_targeted_search(gaps)
    
    assert len(results) == 2
    assert all("Gap:" in result for result in results)


@pytest.mark.asyncio
async def test_max_iterations_limit(talent_agent, mock_model_client):
    """Test that search iterations are limited."""
    # Mock responses that always return gaps
    search_response = Mock()
    search_response.content = "Basic info"
    
    gap_response = Mock()
    gap_response.content = "- Gap 1\n- Gap 2"
    
    def mock_create(messages):
        # Return gaps for gap analysis, basic info for searches
        if "gap" in str(messages).lower():
            return gap_response
        return search_response
    
    mock_model_client.create.side_effect = mock_create
    
    messages = [TextMessage(content="Test Talent", source="user")]
    
    # This should complete without infinite loop
    response = await talent_agent.on_messages(messages)
    
    assert response.chat_message is not None
    # Verify iterations were limited (max_search_iterations=2)
    assert mock_model_client.create.call_count <= 20  # Reasonable upper bound


def test_produced_message_types(talent_agent):
    """Test that agent declares correct message types."""
    types = talent_agent.produced_message_types
    assert TextMessage in types
    assert len(types) >= 1
