# Talent Report Agent

An intelligent agent for researching talents and generating comprehensive markdown profile reports with scoring.

## Overview

The Talent Report Agent automates the process of researching individuals (academics, researchers, professionals) and generating structured markdown reports that include:

- Professional background and affiliations
- Research areas and expertise
- Key achievements and publications
- Multi-dimensional scoring (Research Impact, Expertise, etc.)
- Professional links and references

## Features

### Core Capabilities

1. **Iterative Information Gathering**: Conducts initial searches and identifies information gaps, then performs targeted follow-up searches
2. **Gap Analysis**: Automatically detects missing critical information and searches to fill those gaps
3. **Structured Organization**: Converts raw search results into organized, structured information
4. **Multi-Dimensional Scoring**: Evaluates talents across 5 dimensions:
   - Research Impact (0-10)
   - Expertise Level (0-10)
   - Professional Standing (0-10)
   - Innovation (0-10)
   - Collaboration (0-10)
5. **Markdown Report Generation**: Produces clean, professional markdown reports
6. **Extensible Search Tools**: Can integrate with various search backends (web search, databases, APIs)

### Workflow

```
User Input (Talent Name)
    ↓
Initial Search & Information Gathering
    ↓
Gap Analysis
    ↓
Iterative Gap Filling (max iterations)
    ↓
Information Structuring
    ↓
Score Generation
    ↓
Markdown Report Generation
    ↓
Final Report
```

## Usage

### Basic Usage

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from magentic_ui.agents import TalentReportAgent
from autogen_agentchat.messages import TextMessage

# Initialize
model_client = OpenAIChatCompletionClient(model="gpt-4o")
agent = TalentReportAgent(
    name="talent_reporter",
    model_client=model_client,
    max_search_iterations=3
)

# Generate report
messages = [TextMessage(content="Andrew Ng", source="user")]
response = await agent.on_messages(messages)
print(response.chat_message.content)

# Access structured profile data
profile = agent.get_current_profile()
```

### Command Line Usage

**Simple mode:**
```bash
python samples/sample_talent_report.py --talent "Andrew Ng"
```

**Interactive mode:**
```bash
python samples/sample_talent_report.py --interactive
```

**With output file:**
```bash
python samples/sample_talent_report.py --talent "Yann LeCun" --output reports/yann_lecun.md
```

**Batch processing:**
```bash
python samples/sample_talent_report_pipeline.py --batch talents.txt --output-dir reports/
```

### Advanced: Integration with Web Search

```python
from magentic_ui.agents import TalentReportAgent, WebSurfer
from magentic_ui.tools.playwright import LocalPlaywrightBrowser

# Initialize browser and web surfer
browser = LocalPlaywrightBrowser(headless=True)
web_surfer = WebSurfer(
    name="web_researcher",
    model_client=model_client,
    browser=browser
)
await web_surfer.lazy_init()

# Initialize talent agent with search tools
talent_agent = TalentReportAgent(
    name="talent_reporter",
    model_client=model_client,
    search_tools=[web_surfer]  # Pass search capabilities
)

# Use as normal
# ...

# Cleanup
await web_surfer.close()
```

## Configuration

### Agent Parameters

- `name` (str): Agent name
- `model_client` (ChatCompletionClient): LLM client for the agent
- `search_tools` (Optional[List]): List of search tools to use for information gathering
- `max_search_iterations` (int, default=3): Maximum number of iterations to fill information gaps
- `description` (str): Agent description
- `state` (Optional[BaseState]): Optional state object

### Environment Variables

The agent uses OpenAI's API through the model client. Ensure you have:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or configure alternative endpoints via the model client.

## Output Format

The generated markdown reports follow this structure:

```markdown
# [Full Name]

## Overview
Brief summary of the talent and their focus area.

## Professional Background
- **Current Position**: ...
- **Previous Roles**: ...
- **Education**: ...

## Expertise & Research Areas
List of domains and expertise.

## Key Achievements
- Publications
- Awards
- Contributions

## Scoring Summary
| Dimension | Score | Justification |
|-----------|-------|---------------|
| Research Impact | X/10 | ... |
| Expertise Level | X/10 | ... |
| Professional Standing | X/10 | ... |
| Innovation | X/10 | ... |
| Collaboration | X/10 | ... |

## Professional Links
- Website: [URL]
- Google Scholar: [URL]
- LinkedIn: [URL]

## Notes
Additional context and observations.

---
*Report generated on [Date]*
```

## Accessing Structured Data

The agent maintains a structured profile that can be accessed programmatically:

```python
profile = agent.get_current_profile()

# Access profile attributes
print(f"Name: {profile.name}")
print(f"Timestamp: {profile.timestamp}")
print(f"Search results: {len(profile.raw_search_results)}")
print(f"Scores: {profile.scores}")
print(f"Gaps identified: {profile.gaps_identified}")
print(f"Structured info: {profile.structured_info}")
```

## Integration Examples

### Team-Based Workflow

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination

user_proxy = UserProxyAgent(name="user")
talent_agent = TalentReportAgent(name="researcher", model_client=model_client)

team = RoundRobinGroupChat(
    participants=[talent_agent, user_proxy],
    max_turns=10,
    termination_condition=TextMentionTermination("TERMINATE")
)

await team.run(task="Research Geoff Hinton")
```

### Batch Processing

```python
talents = ["Andrew Ng", "Yann LeCun", "Geoffrey Hinton", "Yoshua Bengio"]

for talent in talents:
    messages = [TextMessage(content=talent, source="user")]
    response = await agent.on_messages(messages)
    
    # Save to file
    filename = f"reports/{talent.replace(' ', '_').lower()}.md"
    Path(filename).write_text(response.chat_message.content)
```

## Customization

### Custom Prompts

You can customize the prompts by modifying the prompt templates in `_prompts.py`:

- `TALENT_SEARCH_SYSTEM_PROMPT`: Initial search behavior
- `TALENT_SCORING_PROMPT`: Scoring criteria and format
- `MARKDOWN_GENERATION_PROMPT`: Report structure and style
- `INFORMATION_GAP_ANALYSIS_PROMPT`: Gap detection logic

### Custom Search Tools

Implement custom search tools and pass them to the agent:

```python
class CustomSearchTool:
    async def search(self, query: str) -> str:
        # Your search implementation
        return results

search_tool = CustomSearchTool()
agent = TalentReportAgent(
    name="reporter",
    model_client=model_client,
    search_tools=[search_tool]
)
```

## Files

- `src/magentic_ui/agents/talent_report/_talent_report_agent.py`: Main agent implementation
- `src/magentic_ui/agents/talent_report/_prompts.py`: Prompt templates
- `src/magentic_ui/agents/talent_report/__init__.py`: Package exports
- `samples/sample_talent_report.py`: Basic usage example
- `samples/sample_talent_report_pipeline.py`: Advanced pipeline example

## Limitations & Future Work

**Current Limitations:**
- Search tool integration is a placeholder (needs real web search/API integration)
- Scoring is LLM-generated (could use metrics-based scoring)
- No caching of search results

**Planned Enhancements:**
- Integration with Google Scholar API
- Integration with ORCID and other academic databases
- Citation count and h-index extraction
- Co-author network analysis
- Temporal trend analysis (career progression)
- Comparison mode (compare multiple talents)
- Export to other formats (PDF, JSON)

## License

Same as magentic-ui project.
