"""Talent Report Agent implementation."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass
from loguru import logger

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
    SystemMessage,
    AssistantMessage,
)
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseChatMessage,
    TextMessage,
    StopMessage,
)
from autogen_agentchat.state import BaseState

from ._prompts import (
    TALENT_SEARCH_SYSTEM_PROMPT,
    TALENT_SCORING_PROMPT,
    MARKDOWN_GENERATION_PROMPT,
    INFORMATION_GAP_ANALYSIS_PROMPT,
)


@dataclass
class TalentProfile:
    """Data class for talent profile information."""
    
    name: str
    raw_search_results: List[str]
    structured_info: Dict[str, Any]
    scores: Dict[str, Dict[str, Any]]
    markdown_report: str
    gaps_identified: List[str]
    timestamp: str


class TalentReportAgent(BaseChatAgent):
    """Agent for researching talents and generating comprehensive markdown reports."""

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        search_tools: Optional[List[Any]] = None,
        max_search_iterations: int = 3,
        description: str = "An agent that researches talents and generates detailed profile reports.",
        state: Optional[BaseState] = None,
    ):
        """
        Initialize the Talent Report Agent.

        Args:
            name: Agent name
            model_client: LLM client for the agent
            search_tools: Optional list of search tools (e.g., web search, database queries)
            max_search_iterations: Maximum number of search iterations to fill information gaps
            description: Agent description
            state: Optional state object
        """
        super().__init__(name=name, description=description, state=state)
        self._model_client = model_client
        self._search_tools = search_tools or []
        self._max_search_iterations = max_search_iterations
        self._current_profile: Optional[TalentProfile] = None

    @property
    def produced_message_types(self) -> List[type[BaseChatMessage]]:
        """Message types produced by this agent."""
        return [TextMessage, StopMessage]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Response:
        """
        Handle incoming messages and generate talent reports.

        Args:
            messages: Sequence of messages to process
            cancellation_token: Optional cancellation token

        Returns:
            Response containing the generated report
        """
        # Extract the talent name from the last user message
        last_message = messages[-1]
        if not isinstance(last_message, TextMessage):
            return Response(
                chat_message=TextMessage(
                    content="Please provide a talent name to research.",
                    source=self.name,
                )
            )

        talent_query = last_message.content
        logger.info(f"Starting talent research for: {talent_query}")

        try:
            # Step 1: Initial search and information gathering
            search_results = await self._conduct_initial_search(talent_query)
            
            # Step 2: Analyze information gaps
            gaps = await self._analyze_information_gaps(talent_query, search_results)
            
            # Step 3: Conduct additional searches if needed
            iteration_count = 0
            while gaps and iteration_count < self._max_search_iterations:
                logger.info(f"Filling information gaps (iteration {iteration_count + 1})")
                additional_results = await self._conduct_targeted_search(gaps)
                search_results.extend(additional_results)
                gaps = await self._analyze_information_gaps(talent_query, search_results)
                iteration_count += 1
            
            # Step 4: Structure the information
            structured_info = await self._structure_information(talent_query, search_results)
            
            # Step 5: Generate scores
            scores = await self._generate_scores(structured_info)
            
            # Step 6: Generate markdown report
            markdown_report = await self._generate_markdown_report(
                talent_query, structured_info, scores
            )
            
            # Store the profile
            self._current_profile = TalentProfile(
                name=talent_query,
                raw_search_results=search_results,
                structured_info=structured_info,
                scores=scores,
                markdown_report=markdown_report,
                gaps_identified=gaps,
                timestamp=datetime.now().isoformat(),
            )
            
            logger.info(f"Talent report generated successfully for: {talent_query}")
            
            return Response(
                chat_message=TextMessage(
                    content=markdown_report,
                    source=self.name,
                )
            )

        except Exception as e:
            logger.error(f"Error generating talent report: {str(e)}")
            return Response(
                chat_message=TextMessage(
                    content=f"Error generating report: {str(e)}",
                    source=self.name,
                )
            )

    async def _conduct_initial_search(self, talent_name: str) -> List[str]:
        """
        Conduct initial comprehensive search for the talent.

        Args:
            talent_name: Name of the talent to search

        Returns:
            List of search results as strings
        """
        logger.info(f"Conducting initial search for: {talent_name}")
        
        # Prepare search prompt
        search_prompt = f"""Search for comprehensive information about: {talent_name}

Include:
- Professional background and current position
- Research areas and expertise
- Publications and contributions
- Academic credentials
- Contact information and online profiles

Provide detailed, factual information from reliable sources."""

        messages = [
            SystemMessage(content=TALENT_SEARCH_SYSTEM_PROMPT),
            UserMessage(content=search_prompt, source="user"),
        ]

        # Use LLM to help structure the search (in a real implementation, 
        # this would interact with actual search tools)
        response = await self._model_client.create(messages=messages)
        
        search_results = []
        if response.content:
            search_results.append(str(response.content))
        
        # If search tools are available, use them
        for tool in self._search_tools:
            try:
                tool_result = await self._execute_search_tool(tool, talent_name)
                if tool_result:
                    search_results.append(tool_result)
            except Exception as e:
                logger.warning(f"Search tool {tool} failed: {str(e)}")
        
        return search_results

    async def _execute_search_tool(self, tool: Any, query: str) -> Optional[str]:
        """
        Execute a search tool with the given query.

        Args:
            tool: Search tool to execute
            query: Search query

        Returns:
            Search result as string or None
        """
        # This is a placeholder for actual tool execution
        # In a real implementation, this would interface with web search APIs,
        # database queries, etc.
        logger.debug(f"Executing search tool: {tool} with query: {query}")
        return None

    async def _analyze_information_gaps(
        self, talent_name: str, search_results: List[str]
    ) -> List[str]:
        """
        Analyze collected information to identify gaps.

        Args:
            talent_name: Name of the talent
            search_results: Current search results

        Returns:
            List of identified information gaps
        """
        if not search_results:
            return ["Basic professional information", "Research areas", "Publications"]
        
        combined_info = "\n\n".join(search_results)
        
        messages = [
            SystemMessage(content=INFORMATION_GAP_ANALYSIS_PROMPT),
            UserMessage(
                content=f"Talent: {talent_name}\n\nCollected Information:\n{combined_info[:4000]}",
                source="user",
            ),
        ]

        response = await self._model_client.create(messages=messages)
        
        # Parse gaps from response (simplified parsing)
        gaps = []
        if response.content:
            content = str(response.content)
            # Extract bullet points or numbered items as gaps
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith(("-", "*", "•")) or (
                    len(line) > 0 and line[0].isdigit() and "." in line[:3]
                ):
                    gap = line.lstrip("-*•0123456789. ")
                    if gap:
                        gaps.append(gap)
        
        return gaps[:5]  # Limit to top 5 gaps

    async def _conduct_targeted_search(self, gaps: List[str]) -> List[str]:
        """
        Conduct targeted searches to fill specific information gaps.

        Args:
            gaps: List of information gaps to address

        Returns:
            List of additional search results
        """
        results = []
        
        for gap in gaps:
            search_query = f"Find information about: {gap}"
            
            messages = [
                SystemMessage(content=TALENT_SEARCH_SYSTEM_PROMPT),
                UserMessage(content=search_query, source="user"),
            ]

            try:
                response = await self._model_client.create(messages=messages)
                if response.content:
                    results.append(f"Gap: {gap}\nInformation: {response.content}")
            except Exception as e:
                logger.warning(f"Failed to fill gap '{gap}': {str(e)}")
        
        return results

    async def _structure_information(
        self, talent_name: str, search_results: List[str]
    ) -> Dict[str, Any]:
        """
        Structure the collected information into organized categories.

        Args:
            talent_name: Name of the talent
            search_results: All collected search results

        Returns:
            Structured information dictionary
        """
        combined_info = "\n\n".join(search_results)
        
        structure_prompt = f"""Structure the following information about {talent_name} into organized categories:

{combined_info[:6000]}

Organize into:
- name: Full name
- current_position: Current role and organization
- previous_roles: List of previous positions
- education: Academic background
- expertise: Research areas and domains
- publications: Notable works
- achievements: Awards and recognition
- links: Professional profiles and websites
- notes: Additional context

Provide structured data in a clear format."""

        messages = [
            SystemMessage(content="You are an expert at organizing professional information."),
            UserMessage(content=structure_prompt, source="user"),
        ]

        response = await self._model_client.create(messages=messages)
        
        # In a real implementation, this would parse structured output
        # For now, return a simplified structure
        structured = {
            "name": talent_name,
            "raw_content": str(response.content) if response.content else "",
            "search_results_count": len(search_results),
        }
        
        return structured

    async def _generate_scores(self, structured_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate scores for various dimensions of the talent.

        Args:
            structured_info: Structured talent information

        Returns:
            Dictionary of scores with justifications
        """
        scoring_prompt = f"""Based on this information, provide scores:

{structured_info.get('raw_content', '')[:4000]}

{TALENT_SCORING_PROMPT}"""

        messages = [
            SystemMessage(content="You are an expert talent evaluator."),
            UserMessage(content=scoring_prompt, source="user"),
        ]

        response = await self._model_client.create(messages=messages)
        
        # Parse scores (simplified)
        scores = {
            "research_impact": {"score": 0, "justification": ""},
            "expertise_level": {"score": 0, "justification": ""},
            "professional_standing": {"score": 0, "justification": ""},
            "innovation": {"score": 0, "justification": ""},
            "collaboration": {"score": 0, "justification": ""},
        }
        
        if response.content:
            scores["_raw_scoring_output"] = str(response.content)
        
        return scores

    async def _generate_markdown_report(
        self,
        talent_name: str,
        structured_info: Dict[str, Any],
        scores: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Generate the final markdown report.

        Args:
            talent_name: Name of the talent
            structured_info: Structured information
            scores: Scoring information

        Returns:
            Markdown formatted report
        """
        report_prompt = f"""Generate a comprehensive markdown report for {talent_name}:

Information:
{structured_info.get('raw_content', '')[:5000]}

Scores:
{scores.get('_raw_scoring_output', '')}

{MARKDOWN_GENERATION_PROMPT}

Current date: {datetime.now().strftime('%Y-%m-%d')}"""

        messages = [
            SystemMessage(content="You are an expert at creating professional reports."),
            UserMessage(content=report_prompt, source="user"),
        ]

        response = await self._model_client.create(messages=messages)
        
        return str(response.content) if response.content else "# Report Generation Failed"

    def get_current_profile(self) -> Optional[TalentProfile]:
        """Get the most recently generated profile."""
        return self._current_profile

    async def on_reset(self, cancellation_token: Optional[CancellationToken] = None) -> None:
        """Reset the agent state."""
        self._current_profile = None
