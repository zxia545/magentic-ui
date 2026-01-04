"""Log formatter utilities for better autogen message visualization."""

import json
from typing import Any, Dict
from loguru import logger


def format_autogen_message(record: Dict[str, Any]) -> str:
    """Format autogen_core messages in a more readable way.
    
    Args:
        record: The log record dictionary
        
    Returns:
        Formatted message string
    """
    message = record["message"]
    
    # Check if this is an autogen_core message
    if "autogen_core" not in record.get("name", ""):
        return message
    
    # Try to extract and format structured data
    try:
        # Publishing message
        if "Publishing message of type" in message:
            parts = message.split(": ", 1)
            if len(parts) == 2:
                header = parts[0]
                data = parts[1]
                try:
                    # Try to parse as Python dict string
                    import ast
                    data_dict = ast.literal_eval(data)
                    formatted_data = json.dumps(data_dict, indent=2)
                    return f"{header}:\n{formatted_data}"
                except:
                    pass
        
        # Calling message handler
        if "Calling message handler" in message:
            return f"📩 {message}"
        
        # Sending message
        if "Sending message" in message:
            return f"📤 {message}"
            
    except Exception:
        pass
    
    return message


def setup_autogen_message_logger(log_file: str, debug: bool = False) -> None:
    """Set up a dedicated logger for autogen messages with better formatting.
    
    Args:
        log_file: Path to the log file
        debug: Whether to enable debug mode
    """
    # Create a separate log file just for autogen messages
    autogen_log_file = log_file.replace(".log", "_autogen_messages.log")
    
    # Add a filter to only log autogen messages
    def autogen_filter(record):
        return "autogen" in record["name"].lower()
    
    # Add handler with custom formatting for autogen messages
    logger.add(
        autogen_log_file,
        level="DEBUG" if debug else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        rotation="100 MB",
        retention="7 days",
        filter=autogen_filter,
    )
    
    logger.info(f"Autogen messages will be logged to: {autogen_log_file}")


def create_autogen_message_html_viewer(log_file: str, output_html: str) -> None:
    """Create an HTML viewer for autogen messages.
    
    Args:
        log_file: Path to the autogen messages log file
        output_html: Path to output HTML file
    """
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>AutoGen Message Viewer</title>
    <style>
        body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
        .message {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; background: #2d2d2d; }}
        .timestamp {{ color: #4ec9b0; }}
        .level {{ color: #ce9178; font-weight: bold; }}
        .info {{ border-left-color: #4ec9b0; }}
        .warning {{ border-left-color: #dcdcaa; }}
        .error {{ border-left-color: #f48771; }}
        .content {{ white-space: pre-wrap; word-wrap: break-word; }}
        h1 {{ color: #569cd6; }}
    </style>
</head>
<body>
    <h1>AutoGen Message Timeline</h1>
    <div id="messages">
        {messages}
    </div>
</body>
</html>"""
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        messages_html = []
        for line in lines:
            # Parse log line
            parts = line.split(" | ", 3)
            if len(parts) >= 4:
                timestamp, level, location, content = parts
                level_class = level.strip().lower()
                
                message_html = f"""
                <div class="message {level_class}">
                    <span class="timestamp">{timestamp}</span>
                    <span class="level">[{level.strip()}]</span>
                    <div class="content">{content}</div>
                </div>
                """
                messages_html.append(message_html)
        
        html_content = html_template.format(messages="\n".join(messages_html))
        
        with open(output_html, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML viewer created at: {output_html}")
        
    except Exception as e:
        logger.error(f"Failed to create HTML viewer: {e}")
