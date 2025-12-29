# Talent Report Agent - Quick Start Guide

## Installation

Ensure dependencies are installed:
```bash
uv sync
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

### 1. Command Line (Easiest)

**Research a single talent:**
```bash
python samples/sample_talent_report.py --talent "Andrew Ng"
```

**Save output to file:**
```bash
python samples/sample_talent_report.py --talent "Yann LeCun" --output reports/yann_lecun.md
```

**Interactive mode:**
```bash
python samples/sample_talent_report.py --interactive
```

### 2. Batch Processing

Create a file `talents.txt` with one name per line:
```
Andrew Ng
Yann LeCun
Geoffrey Hinton
Yoshua Bengio
```

Run batch processing:
```bash
python samples/sample_talent_report_pipeline.py --batch talents.txt --output-dir reports/
```

### 3. Python API

**Minimal example:**
```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from magentic_ui.agents import TalentReportAgent
from autogen_agentchat.messages import TextMessage

async def main():
    # Initialize
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = TalentReportAgent(
        name="talent_reporter",
        model_client=model_client
    )
    
    # Generate report
    messages = [TextMessage(content="Andrew Ng", source="user")]
    response = await agent.on_messages(messages)
    
    # Print report
    print(response.chat_message.content)
    
    # Access structured data
    profile = agent.get_current_profile()
    print(f"\nGenerated at: {profile.timestamp}")
    print(f"Search iterations: {len(profile.raw_search_results)}")

asyncio.run(main())
```

**With team workflow:**
```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from magentic_ui.agents import TalentReportAgent

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create team
    talent_agent = TalentReportAgent(
        name="researcher",
        model_client=model_client
    )
    user_proxy = UserProxyAgent(name="user")
    
    team = RoundRobinGroupChat(
        participants=[talent_agent, user_proxy],
        max_turns=10,
        termination_condition=TextMentionTermination("TERMINATE")
    )
    
    # Run
    result = await team.run(task="Research Andrew Ng and provide a detailed profile")
    print(result)

asyncio.run(main())
```

## Configuration Options

### Agent Parameters

```python
agent = TalentReportAgent(
    name="talent_reporter",           # Agent name
    model_client=model_client,         # LLM client
    search_tools=[],                   # Optional search tools
    max_search_iterations=3,           # Max gap-filling iterations
    description="Custom description",  # Agent description
)
```

### Model Selection

```python
# GPT-4o (recommended)
from autogen_ext.models.openai import OpenAIChatCompletionClient
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# GPT-4 Turbo
model_client = OpenAIChatCompletionClient(model="gpt-4-turbo")

# Custom endpoint
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    base_url="https://your-endpoint.com/v1"
)
```

## Output Format

The agent generates markdown reports with this structure:

```markdown
# [Full Name]

## Overview
2-3 sentence summary

## Professional Background
- **Current Position**: ...
- **Previous Roles**: ...
- **Education**: ...

## Expertise & Research Areas
Domain expertise list

## Key Achievements
- Publications
- Awards
- Contributions

## Scoring Summary
| Dimension             | Score | Justification |
|-----------------------|-------|---------------|
| Research Impact       | 9/10  | ...          |
| Expertise Level       | 8/10  | ...          |
| Professional Standing | 9/10  | ...          |
| Innovation           | 8/10  | ...          |
| Collaboration        | 7/10  | ...          |

## Professional Links
- Website: ...
- Google Scholar: ...
- LinkedIn: ...

## Notes
Additional observations

---
*Report generated on [Date]*
```

## Accessing Profile Data

```python
profile = agent.get_current_profile()

# Available attributes:
profile.name                  # str: Talent name
profile.timestamp            # str: Generation timestamp
profile.raw_search_results   # List[str]: All search results
profile.structured_info      # Dict: Organized information
profile.scores               # Dict: Scores with justifications
profile.markdown_report      # str: Full markdown report
profile.gaps_identified      # List[str]: Information gaps found
```

## Advanced Usage

### Integration with WebSurfer

```python
from magentic_ui.agents import TalentReportAgent, WebSurfer
from magentic_ui.tools.playwright import LocalPlaywrightBrowser

# Initialize browser
browser = LocalPlaywrightBrowser(headless=True)

# Initialize web surfer
web_surfer = WebSurfer(
    name="web_researcher",
    model_client=model_client,
    browser=browser
)
await web_surfer.lazy_init()

# Create talent agent with search tools
talent_agent = TalentReportAgent(
    name="talent_reporter",
    model_client=model_client,
    search_tools=[web_surfer]
)

# Use normally...

# Cleanup
await web_surfer.close()
```

### Custom Search Tools

```python
class CustomSearchTool:
    async def search(self, query: str) -> str:
        # Your search implementation
        # E.g., call Google Scholar API, ORCID API, etc.
        return search_results

search_tool = CustomSearchTool()
agent = TalentReportAgent(
    name="reporter",
    model_client=model_client,
    search_tools=[search_tool]
)
```

### Batch Processing with Progress

```python
import asyncio
from pathlib import Path

async def batch_research(talent_names, output_dir):
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = TalentReportAgent(name="researcher", model_client=model_client)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for idx, talent in enumerate(talent_names, 1):
        print(f"Processing {idx}/{len(talent_names)}: {talent}")
        
        messages = [TextMessage(content=talent, source="user")]
        response = await agent.on_messages(messages)
        
        filename = talent.replace(" ", "_").lower() + ".md"
        filepath = output_dir / filename
        filepath.write_text(response.chat_message.content)
        
        print(f"✓ Saved: {filepath}")

talents = ["Andrew Ng", "Yann LeCun", "Geoffrey Hinton"]
asyncio.run(batch_research(talents, "reports"))
```

## Customization

### Custom Prompts

Edit `src/magentic_ui/agents/talent_report/_prompts.py` to customize:
- `TALENT_SEARCH_SYSTEM_PROMPT` - Initial search behavior
- `TALENT_SCORING_PROMPT` - Scoring criteria
- `MARKDOWN_GENERATION_PROMPT` - Report structure
- `INFORMATION_GAP_ANALYSIS_PROMPT` - Gap detection

### Custom Report Format

Modify `MARKDOWN_GENERATION_PROMPT` to change output format:
```python
MARKDOWN_GENERATION_PROMPT = """
Generate a report with this custom structure:

# {name}

## Custom Section 1
...

## Custom Section 2
...
"""
```

## Troubleshooting

### API Key Issues
```bash
export OPENAI_API_KEY="your-key"
# Or create .env file with OPENAI_API_KEY=your-key
```

### Import Errors
```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Rate Limits
Reduce `max_search_iterations`:
```python
agent = TalentReportAgent(
    name="researcher",
    model_client=model_client,
    max_search_iterations=1  # Fewer API calls
)
```

## Validation

Check implementation:
```bash
python validate_talent_report.py
```

Run tests:
```bash
pytest tests/test_talent_report_agent.py -v
```

## Support

- Documentation: `src/magentic_ui/agents/talent_report/README.md`
- Examples: `samples/sample_talent_report*.py`
- Tests: `tests/test_talent_report_agent.py`

---

**Quick Reference Complete** ✅
