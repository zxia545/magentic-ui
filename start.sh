#!/usr/bin/env bash
set -euo pipefail

# Configure your OpenAI-compatible endpoint via env vars.
# Recommended usage (keeps secrets out of the repo):
#   OPENAI_API_KEY='...' OPENAI_MODEL='gpt-4o-mini' OPENAI_BASE_URL='https://uu.ci/v1' bash start.sh

# OpenAI-compatible proxies often expose non-OpenAI model names.
# autogen-ext requires `model_info` for unknown model names.
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-chat_2025-10-03}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-"http://localhost:8029/v1"}"

# Keep secrets out of the repo: do NOT default the API key.
export OPENAI_API_KEY="${OPENAI_API_KEY:-xxxxxx}"

# Minimal capability hints needed by Magentic-UI agents.
# CRITICAL: Must be enabled for tool/function calling to work!
export OPENAI_MODEL_INFO="${OPENAI_MODEL_INFO:-{\"vision\":false,\"function_calling\":true,\"json_output\":true,\"family\":\"gpt-4o\",\"structured_output\":true,\"multiple_system_messages\":true}}"

# For gateways like aiping.cn that require routing hints in the request body.
# You can override by setting OPENAI_EXTRA_BODY to your own JSON.
# export OPENAI_EXTRA_BODY="${OPENAI_EXTRA_BODY:-{\"provider\":{\"only\":[],\"order\":[],\"sort\":\"latency\",\"input_price_range\":[],\"output_price_range\":[],\"input_length_range\":[],\"throughput_range\":[],\"latency_range\":[]}}}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
	echo "ERROR: OPENAI_API_KEY is not set." >&2
	echo "Set it like: OPENAI_API_KEY='...' bash start.sh" >&2
	exit 1
fi

source .venv/bin/activate

magentic-ui --port 8083