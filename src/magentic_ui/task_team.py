import json
import os
from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_core import ComponentModel
from autogen_core.models import ChatCompletionClient, ModelFamily

from .agents import (
    USER_PROXY_DESCRIPTION,
    CoderAgent,
    FileSurfer,
    FaraWebSurfer,
    TalentSearchAgent,
    WebSurfer,
)
from .agents.mcp import McpAgent
from .agents.users import DummyUserProxy, MetadataUserProxy
from .agents.web_surfer import WebSurferConfig
from .approval_guard import (
    ApprovalConfig,
    ApprovalGuard,
    ApprovalGuardContext,
    BaseApprovalGuard,
)
from .input_func import InputFuncType, make_agentchat_input_func
from .learning.memory_provider import MemoryControllerProvider
from .magentic_ui_config import MagenticUIConfig, ModelClientConfigs
from .teams import GroupChat, RoundRobinGroupChat
from .teams.orchestrator.orchestrator_config import OrchestratorConfig
from .tools.playwright.browser import get_browser_resource_config
from .types import RunPaths
from .utils import get_internal_urls


async def get_task_team(
    magentic_ui_config: Optional[MagenticUIConfig] = None,
    input_func: Optional[InputFuncType] = None,
    *,
    paths: RunPaths,
) -> GroupChat | RoundRobinGroupChat:
    """
    Creates and returns a GroupChat team with specified configuration.

    Args:
        magentic_ui_config (MagenticUIConfig, optional): Magentic UI configuration for team. Default: None.
        paths (RunPaths): Paths for internal and external run directories.

    Returns:
        GroupChat | RoundRobinGroupChat: An instance of GroupChat or RoundRobinGroupChat with the specified agents and configuration.
    """
    if magentic_ui_config is None:
        magentic_ui_config = MagenticUIConfig()

    default_model_info: Dict[str, Any] = {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
        "multiple_system_messages": True,
    }

    def _load_env_json(var_name: str) -> Any:
        raw = os.environ.get(var_name)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _coerce_model_family(model_info: Dict[str, Any]) -> Dict[str, Any]:
        family = model_info.get("family")
        if isinstance(family, ModelFamily):
            return model_info
        if isinstance(family, str):
            key = family.strip().upper()
            # ModelFamily is an Enum; prefer attribute lookup by name.
            if hasattr(ModelFamily, key):
                model_info["family"] = getattr(ModelFamily, key)
        return model_info

    def _allow_autogen_ext_extra_body() -> None:
        # autogen-ext validates `extra_create_args` keys against a hard-coded
        # set derived from OpenAI's typed params. Some OpenAI-compatible
        # gateways require request-level `extra_body`.
        try:
            from autogen_ext.models.openai import _openai_client as _autogen_openai_client  # type: ignore

            create_kwargs = getattr(_autogen_openai_client, "create_kwargs", None)
            if isinstance(create_kwargs, set):
                create_kwargs.add("extra_body")
        except Exception:
            return

    def _inject_extra_body(client: ChatCompletionClient, extra_body: Dict[str, Any]) -> ChatCompletionClient:
        if not extra_body:
            return client

        _allow_autogen_ext_extra_body()

        # Patch instance methods to always include `extra_body`.
        orig_create = getattr(client, "create", None)
        if callable(orig_create):

            async def create(messages: Any, *args: Any, **kwargs: Any) -> Any:
                extra_create_args = dict(kwargs.pop("extra_create_args", {}) or {})
                extra_create_args.setdefault("extra_body", extra_body)
                kwargs["extra_create_args"] = extra_create_args
                return await orig_create(messages, *args, **kwargs)

            setattr(client, "create", create)

        orig_create_stream = getattr(client, "create_stream", None)
        if callable(orig_create_stream):

            async def create_stream(messages: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                extra_create_args = dict(kwargs.pop("extra_create_args", {}) or {})
                extra_create_args.setdefault("extra_body", extra_body)
                kwargs["extra_create_args"] = extra_create_args
                async for chunk in orig_create_stream(messages, *args, **kwargs):
                    yield chunk

            setattr(client, "create_stream", create_stream)

        return client

    def _to_dict(cfg: Any) -> Any:
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            return dict(cfg)
        model_dump = getattr(cfg, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return cfg

    def _apply_openai_env_overrides(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Only apply to OpenAI-like providers.
        provider = cfg_dict.get("provider")
        if not isinstance(provider, str) or "OpenAI" not in provider:
            return cfg_dict

        inner = cfg_dict.get("config")
        if not isinstance(inner, dict):
            inner = {}

        env_model = os.environ.get("OPENAI_MODEL")
        if env_model:
            inner["model"] = env_model
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            inner["api_key"] = env_key
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        if env_base_url:
            inner["base_url"] = env_base_url

        env_model_info = _load_env_json("OPENAI_MODEL_INFO")
        if isinstance(env_model_info, dict):
            inner["model_info"] = _coerce_model_family(dict(env_model_info))

        cfg_dict["config"] = inner
        return cfg_dict

    def _ensure_model_info(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        provider = cfg_dict.get("provider")
        if not isinstance(provider, str) or "OpenAI" not in provider:
            return cfg_dict

        inner = cfg_dict.get("config")
        if not isinstance(inner, dict):
            inner = {}

        if "model_info" not in inner:
            env_model_info = _load_env_json("OPENAI_MODEL_INFO")
            if isinstance(env_model_info, dict):
                inner["model_info"] = _coerce_model_family(dict(env_model_info))
            elif os.environ.get("OPENAI_MODEL"):
                inner["model_info"] = dict(default_model_info)

        cfg_dict["config"] = inner
        return cfg_dict

    def _prepare_model_client_config(
        model_client_config: Union[ComponentModel, Dict[str, Any], None],
        *,
        is_action_guard: bool = False,
    ) -> Any:
        base_config = (
            ModelClientConfigs.get_default_action_guard_config()
            if is_action_guard
            else ModelClientConfigs.get_default_client_config()
        )

        cfg = base_config if model_client_config is None else _to_dict(model_client_config)
        if isinstance(cfg, dict):
            cfg = _apply_openai_env_overrides(cfg)
            cfg = _ensure_model_info(cfg)
        return cfg

    def _load_web_surfer_from_config(
        config: WebSurferConfig,
        *,
        use_fara: bool,
    ) -> WebSurfer:
        builder = FaraWebSurfer if use_fara else WebSurfer
        try:
            return builder.from_config(config)
        except ValueError as e:
            if "model_info is required" in str(e):
                model_client_cfg = _to_dict(config.model_client)
                if isinstance(model_client_cfg, dict):
                    inner = model_client_cfg.get("config")
                    if not isinstance(inner, dict):
                        inner = {}
                    inner.setdefault("model_info", dict(default_model_info))
                    model_client_cfg["config"] = inner
                    updated_config = config.model_copy(update={"model_client": model_client_cfg})
                    return builder.from_config(updated_config)
            raise

    def get_model_client(
        model_client_config: Union[ComponentModel, Dict[str, Any], None],
        is_action_guard: bool = False,
    ) -> ChatCompletionClient:
        cfg = _prepare_model_client_config(
            model_client_config,
            is_action_guard=is_action_guard,
        )

        env_extra_body = _load_env_json("OPENAI_EXTRA_BODY")
        extra_body: Dict[str, Any] = env_extra_body if isinstance(env_extra_body, dict) else {}

        if isinstance(cfg, dict):
            try:
                client = ChatCompletionClient.load_component(cfg)
                return _inject_extra_body(client, extra_body)
            except ValueError as e:
                # autogen-ext requires model_info for unknown model names.
                if "model_info is required" in str(e):
                    inner = cfg.get("config")
                    if not isinstance(inner, dict):
                        inner = {}
                    inner.setdefault("model_info", dict(default_model_info))
                    cfg["config"] = inner
                    client = ChatCompletionClient.load_component(cfg)
                    return _inject_extra_body(client, extra_body)
                raise

        client = ChatCompletionClient.load_component(cfg)
        return _inject_extra_body(client, extra_body)

    if not magentic_ui_config.inside_docker:
        assert (
            paths.external_run_dir == paths.internal_run_dir
        ), "External and internal run dirs must be the same in non-docker mode"

    model_client_orch = get_model_client(
        magentic_ui_config.model_client_configs.orchestrator
    )
    approval_guard: BaseApprovalGuard | None = None

    approval_policy = (
        magentic_ui_config.approval_policy
        if magentic_ui_config.approval_policy
        else "never"
    )

    websurfer_loop_team: bool = (
        magentic_ui_config.websurfer_loop if magentic_ui_config else False
    )

    model_client_coder = get_model_client(magentic_ui_config.model_client_configs.coder)
    model_client_file_surfer = get_model_client(
        magentic_ui_config.model_client_configs.file_surfer
    )
    model_client_talent_search = model_client_orch
    browser_resource_config, _novnc_port, _playwright_port = (
        get_browser_resource_config(
            paths.external_run_dir,
            magentic_ui_config.novnc_port,
            magentic_ui_config.playwright_port,
            magentic_ui_config.inside_docker,
            headless=magentic_ui_config.browser_headless,
            local=magentic_ui_config.browser_local
            or magentic_ui_config.run_without_docker,
            network_name=magentic_ui_config.network_name,
        )
    )

    orchestrator_config = OrchestratorConfig(
        cooperative_planning=magentic_ui_config.cooperative_planning,
        autonomous_execution=magentic_ui_config.autonomous_execution,
        allowed_websites=magentic_ui_config.allowed_websites,
        plan=magentic_ui_config.plan,
        model_context_token_limit=magentic_ui_config.model_context_token_limit,
        do_bing_search=magentic_ui_config.do_bing_search,
        retrieve_relevant_plans=magentic_ui_config.retrieve_relevant_plans,
        memory_controller_key=magentic_ui_config.memory_controller_key,
        allow_follow_up_input=magentic_ui_config.allow_follow_up_input,
        final_answer_prompt=magentic_ui_config.final_answer_prompt,
        sentinel_plan=magentic_ui_config.sentinel_plan,
    )
    websurfer_model_client = _prepare_model_client_config(
        magentic_ui_config.model_client_configs.web_surfer
    )
    websurfer_config = WebSurferConfig(
        name="web_surfer",
        model_client=websurfer_model_client,
        browser=browser_resource_config,
        single_tab_mode=False,
        max_actions_per_step=magentic_ui_config.max_actions_per_step,
        url_statuses={key: "allowed" for key in orchestrator_config.allowed_websites}
        if orchestrator_config.allowed_websites
        else None,
        url_block_list=get_internal_urls(magentic_ui_config.inside_docker, paths),
        multiple_tools_per_call=magentic_ui_config.multiple_tools_per_call,
        downloads_folder=str(paths.internal_run_dir),
        debug_dir=str(paths.internal_run_dir),
        animate_actions=True,
        start_page=None,
        use_action_guard=True,
        to_save_screenshots=False,
    )

    user_proxy: DummyUserProxy | MetadataUserProxy | UserProxyAgent

    if magentic_ui_config.user_proxy_type == "dummy":
        user_proxy = DummyUserProxy(name="user_proxy")
    elif magentic_ui_config.user_proxy_type == "metadata":
        assert (
            magentic_ui_config.task is not None
        ), "Task must be provided for metadata user proxy"
        assert (
            magentic_ui_config.hints is not None
        ), "Hints must be provided for metadata user proxy"
        assert (
            magentic_ui_config.answer is not None
        ), "Answer must be provided for metadata user proxy"
        user_proxy = MetadataUserProxy(
            name="user_proxy",
            description="Metadata User Proxy Agent",
            task=magentic_ui_config.task,
            helpful_task_hints=magentic_ui_config.hints,
            task_answer=magentic_ui_config.answer,
            model_client=model_client_orch,
        )
    else:
        user_proxy_input_func = make_agentchat_input_func(input_func)
        user_proxy = UserProxyAgent(
            description=USER_PROXY_DESCRIPTION,
            name="user_proxy",
            input_func=user_proxy_input_func,
        )

    if magentic_ui_config.user_proxy_type in ["dummy", "metadata"]:
        model_client_action_guard = get_model_client(
            magentic_ui_config.model_client_configs.action_guard,
            is_action_guard=True,
        )

        # Simple approval function that always returns yes
        def always_yes_input(prompt: str, input_type: str = "text_input") -> str:
            return "yes"

        approval_guard = ApprovalGuard(
            input_func=always_yes_input,
            default_approval=False,
            model_client=model_client_action_guard,
            config=ApprovalConfig(
                approval_policy=approval_policy,
            ),
        )
    elif input_func is not None:
        model_client_action_guard = get_model_client(
            magentic_ui_config.model_client_configs.action_guard
        )
        approval_guard = ApprovalGuard(
            input_func=input_func,
            default_approval=False,
            model_client=model_client_action_guard,
            config=ApprovalConfig(
                approval_policy=approval_policy,
            ),
        )
    with ApprovalGuardContext.populate_context(approval_guard):
        web_surfer = _load_web_surfer_from_config(
            websurfer_config,
            use_fara=magentic_ui_config.use_fara_agent,
        )

    env_extra_body = _load_env_json("OPENAI_EXTRA_BODY")
    extra_body_websurfer: Dict[str, Any] = (
        env_extra_body if isinstance(env_extra_body, dict) else {}
    )
    web_surfer._model_client = _inject_extra_body(web_surfer._model_client, extra_body_websurfer)
    if websurfer_loop_team:
        # simplified team of only the web surfer
        team = RoundRobinGroupChat(
            participants=[web_surfer, user_proxy],
            max_turns=10000,
        )
        return team
    coder_agent: CoderAgent | None = None
    file_surfer: FileSurfer | None = None
    talent_search_agent = TalentSearchAgent(
        name="talent_search_agent",
        model_client=model_client_talent_search,
        browser_resource=browser_resource_config,
    )
    if not magentic_ui_config.run_without_docker:
        coder_agent = CoderAgent(
            name="coder_agent",
            model_client=model_client_coder,
            work_dir=paths.internal_run_dir,
            bind_dir=paths.external_run_dir,
            model_context_token_limit=magentic_ui_config.model_context_token_limit,
            approval_guard=approval_guard,
        )

        file_surfer = FileSurfer(
            name="file_surfer",
            model_client=model_client_file_surfer,
            work_dir=paths.internal_run_dir,
            bind_dir=paths.external_run_dir,
            model_context_token_limit=magentic_ui_config.model_context_token_limit,
            approval_guard=approval_guard,
        )

    # Setup any mcp_agents
    mcp_agents: List[McpAgent] = [
        # TODO: Init from constructor?
        McpAgent._from_config(config)  # type: ignore
        for config in magentic_ui_config.mcp_agent_configs
    ]

    if (
        orchestrator_config.memory_controller_key is not None
        and orchestrator_config.retrieve_relevant_plans in ["reuse", "hint"]
    ):
        memory_provider = MemoryControllerProvider(
            internal_workspace_root=paths.internal_root_dir,
            external_workspace_root=paths.external_root_dir,
            inside_docker=magentic_ui_config.inside_docker,
        )
    else:
        memory_provider = None

    team_participants: List[ChatAgent] = [
        web_surfer,
        user_proxy,
    ]
    if not magentic_ui_config.run_without_docker:
        assert coder_agent is not None
        assert file_surfer is not None
        team_participants.extend([coder_agent, file_surfer])
    team_participants.append(talent_search_agent)
    team_participants.extend(mcp_agents)

    team = GroupChat(
        participants=team_participants,
        orchestrator_config=orchestrator_config,
        model_client=model_client_orch,
        memory_provider=memory_provider,
    )

    return team
