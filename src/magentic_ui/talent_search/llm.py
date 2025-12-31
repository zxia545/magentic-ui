from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from . import config, schemas


_DEFAULT_MODEL_CLIENT: ChatCompletionClient | None = None


def set_default_model_client(client: ChatCompletionClient) -> None:
    global _DEFAULT_MODEL_CLIENT
    _DEFAULT_MODEL_CLIENT = client


def _get_default_model_client() -> ChatCompletionClient:
    if _DEFAULT_MODEL_CLIENT is not None:
        return _DEFAULT_MODEL_CLIENT
    # Fallback to a minimal OpenAI-compatible config.
    cfg = {
        "provider": "OpenAIChatCompletionClient",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
        },
        "max_retries": 5,
    }
    return ChatCompletionClient.load_component(cfg)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


@dataclass
class LLMResponse:
    content: str
    text: str


class SimpleLLM:
    def __init__(self, model_client: ChatCompletionClient, temperature: float, max_tokens: int):
        self._model_client = model_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> "SimpleLLM":
        return self

    def invoke(self, prompt: Any) -> LLMResponse:
        messages = _to_messages(prompt)
        async def _call():
            try:
                result = await self._model_client.create(
                    messages=messages,
                    extra_create_args={
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                    },
                )
            except Exception:
                result = await self._model_client.create(messages=messages)
            return result

        result = _run_async(_call())
        content = result.content if isinstance(result.content, str) else str(result.content)
        return LLMResponse(content=content, text=content)


def _to_messages(prompt: Any) -> list[SystemMessage | UserMessage]:
    if isinstance(prompt, str):
        return [UserMessage(content=prompt, source="user")]

    if isinstance(prompt, Sequence):
        messages: list[SystemMessage | UserMessage] = []
        for msg in prompt:
            role = "user"
            content = ""
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
                role = msg.get("role", "user")
                source = msg.get("source") or "user"
            else:
                content = str(getattr(msg, "content", msg))
                name = msg.__class__.__name__.lower()
                if "system" in name:
                    role = "system"
                source = getattr(msg, "source", None) or "user"
            if role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(UserMessage(content=content, source=source))
        return messages

    return [UserMessage(content=str(prompt), source="user")]


def get_llm(role: str, temperature: float = 0.4, api_key: str | None = None) -> SimpleLLM:
    max_tokens = config.LLM_OUT_TOKENS.get(role, 2048)
    client = _get_default_model_client()
    return SimpleLLM(client, temperature=temperature, max_tokens=max_tokens)


def extract_json_block(s: str) -> Optional[dict]:
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return {"items": parsed}
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except Exception:
                        break
        start = s.find("{", start + 1)

    start = s.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return {"items": json.loads(s[start : i + 1])}
                    except Exception:
                        break
    return None


def safe_get(obj: Any, path: Any, default: Any = None) -> Any:
    if not isinstance(path, (list, tuple)):
        path = [path]
    cur = obj
    for key in path:
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def minimal_by_schema(schema_cls):
    if schema_cls is schemas.QuerySpec:
        return schemas.QuerySpec()
    if schema_cls is schemas.QuerySpecDiff:
        return schemas.QuerySpecDiff()
    if schema_cls is schemas.LLMSelectSpec:
        return schemas.LLMSelectSpec(should_fetch=False)
    if schema_cls is schemas.LLMSelectSpecWithValue:
        return schemas.LLMSelectSpecWithValue(
            should_fetch=False, value_score=0.0, reason="LLM unavailable"
        )
    if schema_cls is schemas.LLMSelectSpecHasAuthorInfo:
        return schemas.LLMSelectSpecHasAuthorInfo(
            has_author_info=False, confidence=0.0, reason="LLM unavailable"
        )
    if schema_cls is schemas.LLMSelectSpecVerifyIdentity:
        return schemas.LLMSelectSpecVerifyIdentity(
            is_target_author=False, confidence=0.0, reason="LLM unavailable"
        )
    if schema_cls is schemas.LLMHomepageIdentitySpecSimple:
        return schemas.LLMHomepageIdentitySpecSimple(
            is_personal_homepage=False, confidence=0.0, reason="LLM unavailable"
        )
    if schema_cls is schemas.LLMHomepageIdentitySpec:
        return schemas.LLMHomepageIdentitySpec(
            is_target_author_homepage=False,
            confidence=0.0,
            author_name_found="",
            research_area_match=False,
            reason="LLM unavailable",
        )
    if schema_cls is schemas.LLMPaperNameSpec:
        return schemas.LLMPaperNameSpec(paper_name="", have_paper_name=False)
    if schema_cls is schemas.LLMAuthorProfileSpec:
        return schemas.LLMAuthorProfileSpec()
    if schema_cls is schemas.HomepageInsightsSpec:
        return schemas.HomepageInsightsSpec(
            current_status="",
            role_affiliation_detailed="",
            research_focus=[],
            research_keywords=[],
            highlights=[],
        )
    if schema_cls is schemas.SearchValidationResult:
        return schemas.SearchValidationResult(
            is_valid_search=False,
            search_terms_found=[],
            missing_elements=["LLM service error"],
            suggestion="Please check your API configuration and try again later.",
        )
    if schema_cls is schemas.LLMStringListSpec:
        return schemas.LLMStringListSpec(items=[])
    if schema_cls is schemas.LLMStringSpec:
        return schemas.LLMStringSpec(value="")
    if schema_cls is schemas.LLMResearchInterestsSpec:
        return schemas.LLMResearchInterestsSpec(research_interests=[])
    if schema_cls is schemas.LLMRoleDeterminationSpec:
        return schemas.LLMRoleDeterminationSpec(
            role_category="Unknown",
            role_text="",
            affiliation="",
            explanation="LLM parsing failed, using default",
        )
    return schema_cls()


def safe_structured(llm_client: SimpleLLM, prompt: str, schema_cls):
    for attempt in range(3):
        try:
            resp = llm_client.invoke(prompt)
            txt = safe_get(resp, "content", "") or safe_get(resp, "text", "") or str(resp)
            if not isinstance(txt, str):
                txt = str(txt)
            data = extract_json_block(txt)
            if data is not None:
                try:
                    return schema_cls.model_validate(data)
                except Exception:
                    if config.VERBOSE:
                        print(f"[safe_structured] Validation error for {schema_cls.__name__}")
                    raise
            if config.VERBOSE:
                print(f"[safe_structured] attempt {attempt+1}: no valid JSON for {schema_cls.__name__}")
        except Exception as e:
            if config.VERBOSE:
                print(f"[safe_structured] attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
            continue
    if config.VERBOSE:
        print("[safe_structured] all attempts failed, using minimal fallback")
    return minimal_by_schema(schema_cls)
