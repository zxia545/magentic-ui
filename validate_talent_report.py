#!/usr/bin/env python3
"""
Simple validation script to check the Talent Report Agent implementation.
This doesn't require a full environment setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_structure():
    """Validate the basic structure of the implementation."""
    print("=" * 70)
    print("Talent Report Agent - Implementation Validation")
    print("=" * 70)
    
    checks = []
    
    # Check 1: Files exist
    base_path = Path(__file__).parent / "src" / "magentic_ui" / "agents" / "talent_report"
    
    files_to_check = [
        "__init__.py",
        "_talent_report_agent.py",
        "_prompts.py",
        "README.md",
    ]
    
    for filename in files_to_check:
        filepath = base_path / filename
        if filepath.exists():
            checks.append(("✓", f"File exists: {filename}"))
        else:
            checks.append(("✗", f"File missing: {filename}"))
    
    # Check 2: Check prompts
    prompts_file = base_path / "_prompts.py"
    if prompts_file.exists():
        content = prompts_file.read_text()
        required_prompts = [
            "TALENT_SEARCH_SYSTEM_PROMPT",
            "TALENT_SCORING_PROMPT",
            "MARKDOWN_GENERATION_PROMPT",
            "INFORMATION_GAP_ANALYSIS_PROMPT",
        ]
        for prompt_name in required_prompts:
            if prompt_name in content:
                checks.append(("✓", f"Prompt defined: {prompt_name}"))
            else:
                checks.append(("✗", f"Prompt missing: {prompt_name}"))
    
    # Check 3: Check main agent file
    agent_file = base_path / "_talent_report_agent.py"
    if agent_file.exists():
        content = agent_file.read_text()
        required_components = [
            ("class TalentReportAgent", "Main agent class"),
            ("class TalentProfile", "Profile dataclass"),
            ("async def on_messages", "Message handler"),
            ("async def _conduct_initial_search", "Initial search method"),
            ("async def _analyze_information_gaps", "Gap analysis method"),
            ("async def _generate_markdown_report", "Report generation method"),
        ]
        for component, description in required_components:
            if component in content:
                checks.append(("✓", f"Component found: {description}"))
            else:
                checks.append(("✗", f"Component missing: {description}"))
    
    # Check 4: Check samples
    samples_path = Path(__file__).parent / "samples"
    sample_files = [
        "sample_talent_report.py",
        "sample_talent_report_pipeline.py",
    ]
    for sample_file in sample_files:
        filepath = samples_path / sample_file
        if filepath.exists():
            checks.append(("✓", f"Sample exists: {sample_file}"))
        else:
            checks.append(("✗", f"Sample missing: {sample_file}"))
    
    # Check 5: Check main agents __init__ export
    agents_init = Path(__file__).parent / "src" / "magentic_ui" / "agents" / "__init__.py"
    if agents_init.exists():
        content = agents_init.read_text()
        if "TalentReportAgent" in content:
            checks.append(("✓", "Agent exported in agents/__init__.py"))
        else:
            checks.append(("✗", "Agent not exported in agents/__init__.py"))
    
    # Print results
    print("\nValidation Results:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for status, message in checks:
        print(f"{status} {message}")
        if status == "✓":
            passed += 1
        else:
            failed += 1
    
    print("-" * 70)
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All validation checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: uv sync")
        print("2. Run tests: pytest tests/test_talent_report_agent.py")
        print("3. Try the sample: python samples/sample_talent_report.py --talent 'Andrew Ng'")
    else:
        print("\n⚠️  Some validation checks failed. Please review the implementation.")
    
    return failed == 0


def show_usage():
    """Show usage examples."""
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    
    examples = [
        ("Basic usage", "python samples/sample_talent_report.py --talent 'Andrew Ng'"),
        ("Interactive mode", "python samples/sample_talent_report.py --interactive"),
        ("Save to file", "python samples/sample_talent_report.py --talent 'Yann LeCun' --output report.md"),
        ("Batch processing", "python samples/sample_talent_report_pipeline.py --batch talents.txt"),
    ]
    
    for title, command in examples:
        print(f"\n{title}:")
        print(f"  $ {command}")


def show_code_example():
    """Show a code example."""
    print("\n" + "=" * 70)
    print("Code Example")
    print("=" * 70)
    
    code = '''
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

# Print and access results
print(response.chat_message.content)
profile = agent.get_current_profile()
'''
    
    print(code)


if __name__ == "__main__":
    success = validate_structure()
    
    if success:
        show_usage()
        show_code_example()
    
    sys.exit(0 if success else 1)
