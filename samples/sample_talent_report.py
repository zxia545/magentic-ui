#!/usr/bin/env python3
"""
Sample script to demonstrate the Talent Report Agent.

This script shows how to use the TalentReportAgent to research a talent
and generate a comprehensive markdown profile report with scoring.
"""

import argparse
import asyncio
import os
from pathlib import Path

from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import UserProxyAgent

from magentic_ui.agents import TalentReportAgent

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main function to run the Talent Report Agent.

    Parses command line arguments, initializes the agent,
    and runs the talent research workflow.
    """
    parser = argparse.ArgumentParser(
        description="""
        Run the Talent Report Agent to research talents and generate profile reports.
        
        The agent will:
        1. Search for comprehensive information about the specified talent
        2. Identify and fill information gaps through iterative searches
        3. Structure the collected information
        4. Generate scores across multiple dimensions
        5. Produce a comprehensive markdown report
        
        Example usage:
            python sample_talent_report.py --talent "Andrew Ng"
            python sample_talent_report.py --interactive
        """
    )
    parser.add_argument(
        "--talent",
        type=str,
        help="Name of the talent to research (if not provided, runs in interactive mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of search iterations to fill information gaps (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file path to save the generated report",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with user proxy",
    )
    
    args = parser.parse_args()

    # Initialize the model client
    logger.info(f"Initializing model client with model: {args.model}")
    model_client = OpenAIChatCompletionClient(model=args.model)

    # Initialize the Talent Report Agent
    talent_agent = TalentReportAgent(
        name="talent_reporter",
        model_client=model_client,
        max_search_iterations=args.max_iterations,
    )

    if args.interactive or not args.talent:
        # Interactive mode with user proxy
        logger.info("Running in interactive mode")
        
        termination = TextMentionTermination("TERMINATE")
        user_proxy = UserProxyAgent(name="user_proxy")

        team = RoundRobinGroupChat(
            participants=[talent_agent, user_proxy],
            max_turns=20,
            termination_condition=termination,
        )

        print("\n" + "=" * 70)
        print("Talent Report Agent - Interactive Mode")
        print("=" * 70)
        print("\nEnter a talent name to research, or type 'TERMINATE' to exit.")
        print("Example: 'Research Andrew Ng' or 'Generate a report for Yann LeCun'\n")

        user_message = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter talent name to research: "
        )

        stream = team.run_stream(task=user_message)
        await Console(stream)

    else:
        # Direct mode - research the specified talent
        logger.info(f"Researching talent: {args.talent}")
        
        print("\n" + "=" * 70)
        print(f"Generating Talent Report for: {args.talent}")
        print("=" * 70)
        print()

        # Create a simple message to trigger the agent
        from autogen_agentchat.messages import TextMessage
        
        messages = [TextMessage(content=args.talent, source="user")]
        
        # Get the response
        response = await talent_agent.on_messages(messages)
        
        # Display the report
        print(response.chat_message.content)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(response.chat_message.content)
            logger.info(f"Report saved to: {output_path}")
        
        # Also get the structured profile data
        profile = talent_agent.get_current_profile()
        if profile:
            logger.info(f"\nProfile generated at: {profile.timestamp}")
            logger.info(f"Search results collected: {len(profile.raw_search_results)}")
            if profile.gaps_identified:
                logger.info(f"Information gaps identified: {len(profile.gaps_identified)}")


if __name__ == "__main__":
    asyncio.run(main())
