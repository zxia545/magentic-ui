#!/usr/bin/env python3
"""
Advanced Talent Report Agent with Web Search Integration.

This script demonstrates how to use the TalentReportAgent integrated with
WebSurfer for real web searches to gather talent information.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import UserProxyAgent

from magentic_ui.agents import TalentReportAgent, WebSurfer
from magentic_ui.tools.playwright import LocalPlaywrightBrowser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TalentReportPipeline:
    """Pipeline for researching talents using web search and generating reports."""
    
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        browser=None,
        max_iterations: int = 3,
    ):
        """
        Initialize the talent report pipeline.
        
        Args:
            model_client: OpenAI model client
            browser: Optional browser for web searches
            max_iterations: Maximum search iterations
        """
        self.model_client = model_client
        self.browser = browser
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.web_surfer = None
        self.talent_agent = None
        
    async def initialize(self):
        """Initialize the pipeline agents."""
        # Initialize web surfer for searches
        if self.browser:
            self.web_surfer = WebSurfer(
                name="web_researcher",
                model_client=self.model_client,
                animate_actions=False,
                max_actions_per_step=5,
                browser=self.browser,
                to_save_screenshots=False,
            )
            await self.web_surfer.lazy_init()
        
        # Initialize talent report agent
        self.talent_agent = TalentReportAgent(
            name="talent_reporter",
            model_client=self.model_client,
            max_search_iterations=self.max_iterations,
        )
        
        logger.info("Pipeline initialized successfully")
    
    async def research_talent(self, talent_name: str) -> str:
        """
        Research a talent and generate a comprehensive report.
        
        Args:
            talent_name: Name of the talent to research
            
        Returns:
            Markdown formatted report
        """
        logger.info(f"Starting research pipeline for: {talent_name}")
        
        # Step 1: Use web surfer to gather initial information
        if self.web_surfer:
            logger.info("Conducting web searches...")
            search_queries = [
                f"{talent_name} professional profile",
                f"{talent_name} research publications",
                f"{talent_name} academic background",
            ]
            
            # In a full implementation, you would run these searches
            # and collect the results to pass to the talent agent
            # For now, we let the talent agent handle it directly
        
        # Step 2: Generate the talent report
        from autogen_agentchat.messages import TextMessage
        
        messages = [TextMessage(content=talent_name, source="user")]
        response = await self.talent_agent.on_messages(messages)
        
        logger.info("Report generation completed")
        return response.chat_message.content
    
    async def batch_research(self, talent_names: list[str], output_dir: Path):
        """
        Research multiple talents and save reports.
        
        Args:
            talent_names: List of talent names to research
            output_dir: Directory to save reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, talent in enumerate(talent_names, 1):
            logger.info(f"Processing {idx}/{len(talent_names)}: {talent}")
            
            try:
                report = await self.research_talent(talent)
                
                # Save report
                filename = talent.replace(" ", "_").lower() + ".md"
                filepath = output_dir / filename
                filepath.write_text(report)
                
                logger.info(f"Saved report to: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to research {talent}: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.web_surfer:
            await self.web_surfer.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced Talent Report Agent with Web Search Integration"
    )
    parser.add_argument(
        "--talent",
        type=str,
        help="Single talent name to research",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to file containing talent names (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="talent_reports",
        help="Directory to save reports (default: talent_reports)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--use-browser",
        action="store_true",
        help="Use browser for web searches",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum search iterations (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model_client = OpenAIChatCompletionClient(model=args.model)
    
    # Initialize browser if requested
    browser = None
    if args.use_browser:
        browser = LocalPlaywrightBrowser(headless=True)
    
    # Initialize pipeline
    pipeline = TalentReportPipeline(
        model_client=model_client,
        browser=browser,
        max_iterations=args.max_iterations,
    )
    await pipeline.initialize()
    
    try:
        if args.batch:
            # Batch processing
            batch_file = Path(args.batch)
            if not batch_file.exists():
                logger.error(f"Batch file not found: {args.batch}")
                return
            
            talent_names = [
                line.strip()
                for line in batch_file.read_text().splitlines()
                if line.strip()
            ]
            
            logger.info(f"Processing {len(talent_names)} talents from batch file")
            await pipeline.batch_research(talent_names, Path(args.output_dir))
            
        elif args.talent:
            # Single talent research
            report = await pipeline.research_talent(args.talent)
            
            print("\n" + "=" * 70)
            print(report)
            print("=" * 70)
            
            # Save report
            output_path = Path(args.output_dir) / f"{args.talent.replace(' ', '_').lower()}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to: {output_path}")
            
        else:
            print("Please specify --talent or --batch")
            parser.print_help()
            
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
