from __future__ import annotations

import asyncio
import csv
import json
import os
import queue
import re
 
from typing import Any, AsyncGenerator, List, Optional, Sequence
import io

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.utils import content_to_str
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.models import ChatCompletionClient
from pydantic import BaseModel
from typing_extensions import Self


class TalentSearchAgentConfig(BaseModel):
    """Configuration for TalentSearchAgent."""

    name: str
    model_client: ComponentModel
    description: str | None = None
    max_rounds_per_run: int = 50
    browser_resource: ComponentModel | dict | None = None


class TalentSearchAgent(BaseChatAgent, Component[TalentSearchAgentConfig]):
    """Agent wrapper around the in-repo TalentSearch pipeline."""

    component_config_schema = TalentSearchAgentConfig
    component_provider_override = "magentic_ui.agents.talent_search.TalentSearchAgent"

    DEFAULT_DESCRIPTION = """
    A talent search specialist for recruitment-style queries (e.g., find PhD researchers/candidates).
    Runs the TalentSearch pipeline to discover and rank candidates, returning JSON, markdown cards, and a CSV summary.
    If a LEADS_JSON block is provided, use it as seed candidates for enrichment.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        description: str = DEFAULT_DESCRIPTION,
        max_rounds_per_run: int = 50,
        browser_resource: ComponentModel | dict | None = None,
    ) -> None:
        super().__init__(name, description)
        self._model_client = model_client
        self._max_rounds_per_run = max_rounds_per_run
        self._browser_resource = browser_resource
        self._pipeline_loaded = False
        self._pipeline_error: str | None = None
        self._talent_agents: Any = None
        self._talent_schemas: Any = None
        self._talent_task_manager: Any = None
        self._talent_llm: Any = None
        self._talent_search: Any = None
        self._chat_history: List[BaseChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    @classmethod
    def from_config(cls, config: TalentSearchAgentConfig) -> Self:  # type: ignore[override]
        client = ChatCompletionClient.load_component(config.model_client)
        return cls(
            name=config.name,
            model_client=client,
            description=config.description or cls.DEFAULT_DESCRIPTION,
            max_rounds_per_run=config.max_rounds_per_run,
            browser_resource=config.browser_resource,
        )

    def _load_pipeline(self) -> None:
        if self._pipeline_loaded:
            return
        try:
            from magentic_ui.talent_search import agents as talent_agents
            from magentic_ui.talent_search import schemas as talent_schemas
            from magentic_ui.talent_search import task_manager as talent_task_manager
            from magentic_ui.talent_search import llm as talent_llm
            from magentic_ui.talent_search import search as talent_search
        except Exception as exc:
            self._pipeline_error = f"Failed to import talent search pipeline: {exc}"
            self._pipeline_loaded = True
            return
        self._talent_agents = talent_agents
        self._talent_schemas = talent_schemas
        self._talent_task_manager = talent_task_manager
        self._talent_llm = talent_llm
        self._talent_search = talent_search
        self._pipeline_loaded = True

    def _extract_query(self, message: BaseChatMessage) -> str:
        raw = content_to_str(message.content)
        raw = raw.strip()
        if not raw:
            return ""
        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    for key in ("search_query", "query", "task"):
                        value = payload.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
            except json.JSONDecodeError:
                pass
        return raw

    def _extract_json_block(self, text: str) -> dict | list | None:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
            start = text.find("{", start + 1)

        start = text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
        return None

    def _extract_seed_leads(self, messages: Sequence[BaseChatMessage]) -> List[dict]:
        payload: dict | list | None = None
        for msg in reversed(list(messages) + list(self._chat_history)):
            content = content_to_str(msg.content)
            if "LEADS_JSON" in content:
                content = content.split("LEADS_JSON", 1)[-1]
            payload = self._extract_json_block(content)
            if payload is not None:
                break
        if payload is None:
            for msg in reversed(list(messages) + list(self._chat_history)):
                content = content_to_str(msg.content)
                leads = self._extract_seed_leads_from_text(content)
                if leads:
                    return leads
            return []

        leads: Any = None
        if isinstance(payload, dict):
            leads = (
                payload.get("leads")
                or payload.get("candidates")
                or payload.get("seed_candidates")
                or payload.get("people")
            )
        elif isinstance(payload, list):
            leads = payload

        if not leads or not isinstance(leads, list):
            return []

        normalized: List[dict] = []
        for item in leads:
            if isinstance(item, str):
                normalized.append({"name": item})
                continue
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("author") or item.get("candidate")
            paper_title = item.get("paper_title") or item.get("paper") or item.get("title")
            paper_url = item.get("paper_url") or item.get("url")
            author_id = (
                item.get("author_id")
                or item.get("authorId")
                or item.get("s2_author_id")
                or item.get("openreview_id")
            )
            normalized.append(
                {
                    "name": name,
                    "paper_title": paper_title,
                    "paper_url": paper_url,
                    "author_id": author_id,
                    "paper_venue": item.get("paper_venue") or item.get("venue"),
                    "paper_year": item.get("paper_year") or item.get("year"),
                    "affiliation": item.get("affiliation"),
                    "homepage": item.get("homepage"),
                }
            )
        return normalized

    def _extract_seed_leads_from_text(self, text: str) -> List[dict]:
        if not text:
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        blocks: List[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        bullet_re = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*)$")
        for line in lines:
            match = bullet_re.match(line)
            if match:
                name_line = match.group(1).strip()
                current = {"name_line": name_line, "lines": []}
                blocks.append(current)
                continue
            if current is not None:
                current["lines"].append(line)

        leads: List[dict] = []
        url_re = re.compile(r"(https?://\S+)")

        def extract_field(field_lines: List[str], keys: List[str]) -> str:
            for line in field_lines:
                lower = line.lower()
                for key in keys:
                    key_lc = key.lower()
                    if lower.startswith(f"{key_lc}:"):
                        return line.split(":", 1)[1].strip()
            return ""

        for block in blocks:
            name_line = block.get("name_line", "")
            name = re.split(r"\s*[-|]\s*", name_line, 1)[0].strip()
            if not name or len(name.split()) > 6:
                continue
            block_lines = block.get("lines", [])
            block_text = " ".join(block_lines)
            affiliation = extract_field(block_lines, ["affiliation", "institution"])
            paper_title = extract_field(block_lines, ["paper", "paper title", "trigger paper"])
            if paper_title:
                paper_title = paper_title.strip().strip('"').strip("'")
            paper_venue = extract_field(block_lines, ["venue", "conference", "journal"])
            paper_year = ""
            for line in block_lines:
                if "year" in line.lower() or "venue" in line.lower() or "paper" in line.lower():
                    year_match = re.search(r"\b(19|20)\d{2}\b", line)
                    if year_match:
                        paper_year = year_match.group(0)
                        break
            paper_url = ""
            url_match = url_re.search(block_text)
            if url_match:
                paper_url = url_match.group(1).rstrip(").,;")

            leads.append(
                {
                    "name": name,
                    "paper_title": paper_title,
                    "paper_url": paper_url,
                    "paper_venue": paper_venue,
                    "paper_year": paper_year,
                    "affiliation": affiliation,
                }
            )

        # Deduplicate by name
        seen = set()
        deduped: List[dict] = []
        for lead in leads:
            key = (lead.get("name") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(lead)
        return deduped

    def _get_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    def _run_search(
        self,
        query: str,
        progress_callback: Any | None = None,
        log_callback: Any | None = None,
        seed_candidates: List[dict] | None = None,
    ) -> tuple[Any | None, str | None]:
        self._load_pipeline()
        if self._pipeline_error:
            return None, self._pipeline_error
        if not self._talent_agents or not self._talent_schemas:
            return None, "Talent search pipeline is not available."
        if self._talent_search and self._browser_resource:
            try:
                self._talent_search.set_browser_resource_config(self._browser_resource)
            except Exception:
                pass
        if self._talent_llm:
            try:
                self._talent_llm.set_default_model_client(self._model_client)
            except Exception:
                pass
        api_key = self._get_api_key()
        try:
            spec = self._talent_agents.agent_parse_search_query(query, api_key=api_key)
            result = self._talent_agents.agent_execute_search(
                spec,
                api_key=api_key,
                max_rounds_per_run=self._max_rounds_per_run,
                progress_callback=progress_callback,
                log_callback=log_callback,
                seed_candidates=seed_candidates,
            )
            while isinstance(result, self._talent_schemas.PartialSearchResults) and result.need_user_decision:
                if not self._talent_task_manager:
                    return None, "Search paused but task manager is unavailable."
                state = self._talent_task_manager.load_task_state(result.task_id)
                if not state:
                    return None, f"Search paused but could not load task state {result.task_id}"
                result = self._talent_agents.agent_execute_search(
                    spec,
                    api_key=api_key,
                    max_rounds_per_run=self._max_rounds_per_run,
                    resume_state=state,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    seed_candidates=seed_candidates,
                )
            if isinstance(result, self._talent_schemas.SearchResults):
                return result, None
            if isinstance(result, self._talent_schemas.PartialSearchResults):
                return result, None
        except Exception as exc:
            return None, f"Talent search failed: {exc}"
        return None, "Talent search returned an unexpected result."

    def _format_markdown_cards(self, results: Any) -> str:
        lines: List[str] = []
        candidates = list(results.recommended_candidates or []) + list(
            results.additional_candidates or []
        )
        if not candidates:
            return "No candidates found."
        for cand in candidates:
            name = getattr(cand, "name", "Unknown")
            intro = getattr(cand, "introduction", None)
            position = getattr(intro, "position", "") if intro else ""
            affiliation = getattr(intro, "affiliation", "") if intro else ""
            profile = results.enhanced_profiles.get(name) if results.enhanced_profiles else None
            if profile and getattr(profile, "current_role", None):
                current_role = profile.current_role
                if current_role.role_text:
                    position = current_role.role_text
                if current_role.affiliation:
                    affiliation = current_role.affiliation
            score_paper = getattr(cand, "score_paper", 0.0)
            final_score = getattr(cand, "final_score", 0.0)
            category = getattr(cand, "candidate_category", "Unknown")
            trigger_title = getattr(cand, "trigger_paper_title", "")
            trigger_venue = getattr(cand, "trigger_paper_venue", "")
            trigger_score = getattr(cand, "trigger_paper_score", 0.0)
            interests = getattr(cand, "research_interests", []) or []
            interest_names = [getattr(i, "name", "") for i in interests if getattr(i, "name", "")]
            papers = []
            selected = getattr(cand, "selected_research", None)
            if selected and getattr(selected, "all_publications", None):
                papers = list(selected.all_publications)
            if papers:
                papers.sort(key=lambda p: getattr(p, "relevance_score", 0.0), reverse=True)
                papers = papers[:3]
            lines.append(f"### {name}")
            lines.append(
                f"- Role: {position or 'Unknown'} | Affiliation: {affiliation or 'Unknown'} | Category: {category}"
            )
            lines.append(
                f"- Scores: final={final_score:.2f}, paper={score_paper:.2f}"
            )
            if trigger_title:
                lines.append(
                    f"- Trigger paper: {trigger_title} ({trigger_venue}) score={trigger_score:.1f}"
                )
            if interest_names:
                lines.append(f"- Interests: {', '.join(interest_names)}")
            if papers:
                lines.append("- Top papers:")
                for pub in papers:
                    title = getattr(pub, "title", "")
                    venue = getattr(pub, "venue", "")
                    year = getattr(pub, "year", None)
                    pos = getattr(pub, "candidate_author_position_label", "") or ""
                    score = getattr(pub, "relevance_score", 0.0)
                    year_str = str(year) if year else ""
                    parts = [title]
                    if venue or year_str:
                        parts.append(f"{venue} {year_str}".strip())
                    if pos:
                        parts.append(pos)
                    parts.append(f"score={score:.1f}")
                    lines.append(f"  - " + " | ".join(p for p in parts if p))
            lines.append("")
        return "\n".join(lines).strip()

    def _format_csv_summary(self, results: Any) -> str:
        rows = []
        candidates = list(results.recommended_candidates or []) + list(
            results.additional_candidates or []
        )
        for cand in candidates:
            name = getattr(cand, "name", "Unknown")
            intro = getattr(cand, "introduction", None)
            position = getattr(intro, "position", "") if intro else ""
            affiliation = getattr(intro, "affiliation", "") if intro else ""
            profile = results.enhanced_profiles.get(name) if results.enhanced_profiles else None
            if profile and getattr(profile, "current_role", None):
                current_role = profile.current_role
                if current_role.role_text:
                    position = current_role.role_text
                if current_role.affiliation:
                    affiliation = current_role.affiliation
            contact = getattr(cand, "contact", None)
            email = getattr(contact, "email", "") if contact else ""
            homepage = getattr(contact, "homepage", "") if contact else ""
            score_paper = getattr(cand, "score_paper", 0.0)
            final_score = getattr(cand, "final_score", 0.0)
            category = getattr(cand, "candidate_category", "Unknown")
            trigger_title = getattr(cand, "trigger_paper_title", "")
            rows.append(
                {
                    "name": name,
                    "role": position,
                    "affiliation": affiliation,
                    "category": category,
                    "final_score": f"{final_score:.2f}",
                    "paper_score": f"{score_paper:.2f}",
                    "trigger_paper": trigger_title,
                    "email": email,
                    "homepage": homepage,
                }
            )
        if not rows:
            return ""
        buffer = io.StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().strip()

    def _format_progress_event(self, event: str, pct: float) -> str:
        name = event
        details: dict[str, Any] | None = None
        if ":::" in event:
            name, payload = event.split(":::", 1)
            try:
                details = json.loads(payload)
            except Exception:
                details = None
        parts = [f"[{name}] {int(pct * 100)}%"]
        if isinstance(details, dict):
            if details.get("round"):
                parts.append(f"round {details['round']}")
            if details.get("current_term"):
                parts.append(f"term={details['current_term']}")
            if details.get("papers_count") is not None:
                parts.append(f"papers={details['papers_count']}")
            if details.get("target_count") is not None:
                parts.append(f"target={details['target_count']}")
            if details.get("processing_author"):
                parts.append(f"author={details['processing_author']}")
        return " | ".join(parts)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        response: Response | None = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        self._chat_history.extend(messages)
        last_message = messages[-1]
        query = self._extract_query(last_message)
        seed_leads = self._extract_seed_leads(messages)
        if not query:
            yield Response(
                chat_message=TextMessage(
                    content="No search query provided.",
                    source=self.name,
                    metadata={"internal": "no", "type": "progress_message"},
                )
            )
            return

        yield TextMessage(
            content="Starting talent search pipeline...",
            source=self.name,
            metadata={"internal": "no", "type": "progress_message"},
        )
        progress_queue: "queue.Queue[tuple[str, float]]" = queue.Queue()

        def progress_callback(event: str, pct: float) -> None:
            progress_queue.put((event, pct))

        search_task = asyncio.create_task(
            asyncio.to_thread(self._run_search, query, progress_callback, None, seed_leads)
        )
        last_progress: str | None = None
        while True:
            try:
                event, pct = progress_queue.get_nowait()
            except queue.Empty:
                if search_task.done():
                    break
                await asyncio.sleep(0.1)
                continue
            message = self._format_progress_event(event, pct)
            if message == last_progress:
                continue
            last_progress = message
            yield TextMessage(
                content=message,
                source=self.name,
                metadata={"internal": "no", "type": "progress_message"},
            )

        result, error = await search_task
        if error:
            yield Response(
                chat_message=TextMessage(
                    content=error,
                    source=self.name,
                    metadata={"internal": "no"},
                )
            )
            return

        if result is None:
            yield Response(
                chat_message=TextMessage(
                    content="Talent search returned no results.",
                    source=self.name,
                    metadata={"internal": "no"},
                )
            )
            return

        if self._talent_schemas and isinstance(result, self._talent_schemas.PartialSearchResults):
            payload = result.model_dump()
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            yield Response(
                chat_message=TextMessage(
                    content=f"Partial results (task_id={result.task_id}):\n\n{text}",
                    source=self.name,
                    metadata={"internal": "no"},
                )
            )
            return

        json_output = json.dumps(result.model_dump(by_alias=True), indent=2, ensure_ascii=False)
        markdown_output = self._format_markdown_cards(result)
        csv_output = self._format_csv_summary(result)
        combined = (
            "RESULT_JSON:\n"
            f"{json_output}\n\n"
            "MARKDOWN_CARDS:\n"
            f"{markdown_output}\n\n"
            "CSV_SUMMARY:\n"
            f"{csv_output}"
        )
        yield Response(
            chat_message=TextMessage(
                content=combined,
                source=self.name,
                metadata={"internal": "no"},
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Clear the chat history."""
        self._chat_history.clear()
