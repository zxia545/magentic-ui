# Magentic-UI `WebSurfer` Agent 实现说明（中文）

本文档面向想要“读懂/改造/扩展” `src/magentic_ui/agents/web_surfer` 的开发者，重点解释 `WebSurfer` 是如何把 **页面状态（截图 + 可交互元素）→ prompt → LLM 工具调用 → Playwright 执行 → 观察（截图/文本）** 串起来的，以及它与 **审批（ApprovalGuard）/URL 白名单（UrlStatusManager）/teams orchestrator** 的集成方式。

> 不覆盖：`src/magentic_ui/agents/web_surfer/fara/`（你已说明可以不用管）。

---

## 目录
1. [目录结构与关键文件](#1-目录结构与关键文件)
2. [核心对象：`WebSurferConfig` / `WebSurferState` / `WebSurfer`](#2-核心对象websurferconfig--websurferstate--websurfer)
3. [运行生命周期：lazy init / reset / close](#3-运行生命周期lazy-init--reset--close)
4. [主循环：`on_messages_stream`（多动作一回合）](#4-主循环on_messages_stream多动作一回合)
5. [LLM 决策：`_get_llm_response`（构造页面上下文 + 调用模型）](#5-llm-决策_get_llm_response构造页面上下文--调用模型)
6. [Set-of-Mark：给截图打编号与 ID 映射](#6-set-of-mark给截图打编号与-id-映射)
7. [Tool 系统：Schema 定义与执行映射](#7-tool-系统schema-定义与执行映射)
8. [审批与安全：URL 白名单 + ActionGuard](#8-审批与安全url-白名单--actionguard)
9. [`answer_question`：整页阅读/问答（Markdown + 截图）](#9-answer_question整页阅读问答markdown--截图)
10. [状态持久化：`save_state` / `load_state`](#10-状态持久化save_state--load_state)
11. [与 teams/orchestrator 的集成方式](#11-与-teamsorchestrator-的集成方式)
12. [扩展与踩坑：如何启用/新增工具、常见问题](#12-扩展与踩坑如何启用新增工具常见问题)

---

## 1. 目录结构与关键文件

`src/magentic_ui/agents/web_surfer`（排除 `fara/`）的核心文件如下：

```
src/magentic_ui/agents/web_surfer/
├── __init__.py
├── _web_surfer.py          # 主实现：WebSurferConfig/WebSurferState/WebSurfer
├── _tool_definitions.py    # ToolSchema（工具声明）
├── _prompts.py             # system/tool/no-tools/QA/OCR prompts
├── _set_of_mark.py         # 给截图标注可交互区域的编号（Set-of-Mark）
├── _events.py              # WebSurferEvent（日志/调试事件结构）
└── _cua_web_surfer.py      # WebSurferCUA 变体（当前默认 team 未使用）
```

`src/magentic_ui/agents/web_surfer/__init__.py` 导出：
- `WebSurfer`, `WebSurferConfig`（主线）
- `WebSurferCUA`（变体）
- `FaraWebSurfer`（在 `fara/`，本文不讨论）

---

## 2. 核心对象：`WebSurferConfig` / `WebSurferState` / `WebSurfer`

### 2.1 `WebSurferConfig`（可序列化的 Agent 配置）
位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py`

核心字段（按功能分组）：
- **模型与上下文**
  - `model_client`: LLM client component（支持 tool calling 或 JSON 输出）
  - `model_context_token_limit`: 用 `TokenLimitedChatCompletionContext` 限制每次请求的 token
  - `json_model_output`: 当模型不支持 function calling 时自动走 JSON 输出模式
  - `multiple_tools_per_call`: 对部分模型 family，允许 `parallel_tool_calls`
- **浏览器与页面**
  - `browser`: `PlaywrightBrowser` / `VncDockerPlaywrightBrowser`
  - `start_page`: 默认 `about:blank`
  - `single_tab_mode`: 是否强制单标签（新 tab 会被关闭并重定向到主 tab）
  - `viewport_width/height`, `to_resize_viewport`
  - `downloads_folder`: 下载目录
- **可观测与调试**
  - `animate_actions`: 是否显示鼠标/高亮动画（由 `PlaywrightController` 实现）
  - `to_save_screenshots` + `debug_dir`: 是否在本地保存 raw/SoM 截图
- **安全/策略**
  - `url_statuses`: 允许/拒绝的网站（配合 `UrlStatusManager`）
  - `url_block_list`: 显式 block 列表
  - `use_action_guard`: 是否启用 ApprovalGuard（用于 URL 授权与工具审批）
  - `search_engine`: `"duckduckgo" | "google" | "bing" | "yahoo" | 自定义域名/URL`

### 2.2 `WebSurferState`（可保存/恢复的运行态）
同文件内：
- `chat_history: List[LLMMessage]`：给模型的历史（WebSurfer 自己维护）
- `browser_state: BrowserState | None`：浏览器 tab/active page 等（见 `src/magentic_ui/tools/playwright/playwright_state.py`）

### 2.3 `WebSurfer`（核心 Agent）
`WebSurfer(BaseChatAgent, Component[WebSurferConfig])`：
- 实际浏览器控制由 `PlaywrightController` 完成（`src/magentic_ui/tools/playwright/playwright_controller.py`）
- 通过 LLM 产出工具调用（function calls 或 JSON），并逐个执行
- 每一步执行后产出“观察”：截图 + 页面文字摘要（用于下一轮决策和给 UI 显示）

---

## 3. 运行生命周期：lazy init / reset / close

### 3.1 `lazy_init()`：首次使用时才启动浏览器
位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:402`

要点：
- `did_lazy_init` 防重复
- 进入 `self._browser.__aenter__()`，拿到 `browser_context`
- 创建 `Page` 并交给 `PlaywrightController.on_new_page(...)` 做初始化
- `single_tab_mode=True` 时注册 `context.on("page", ...)`：任何新 tab 都会被关闭，并把 URL 重定向回主 tab
- 尝试访问 `start_page`
- 若开启 `to_save_screenshots` 则强制要求 `debug_dir` 存在
- 对 `VncDockerPlaywrightBrowser` 会记录 `novnc_port`/`playwright_port`，并在第一次 `on_messages_stream` 时给 UI 发 `type=browser_address`

### 3.2 `on_reset()`：回到起始页并清空内部历史
位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:500`
- 清空 `_chat_history`
- `PlaywrightController.visit_page(..., start_page)` 回到起始页
- 重置 `_last_download`、`_prior_metadata_hash`

### 3.3 `close()`：释放浏览器与模型资源
位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:475`
- `self._browser.__aexit__(...)`
- 若 model_client 支持 `close()` 也会关闭

---

## 4. 主循环：`on_messages_stream`（多动作一回合）

位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:534`

`WebSurfer` 的一个关键设计是：**一次收到任务（一个 step 的 instruction）时，会在单次 `on_messages_stream` 调用里连续执行多个 browser action**，直到：
- 达到 `max_actions_per_step`
- 或模型决定 `stop_action`
- 或执行 `answer_question`（整页阅读）后停止
- 或 paused/cancelled

### 4.1 输入消息如何进入 WebSurfer 的 LLM 历史
处理规则：
- 只保留：
  - 所有 `MultiModalMessage`（通常来自用户上传图）
  - 最后一个 `TextMessage`（作为本轮任务指令）
- 这些消息会被转成 `UserMessage(...)` 追加到 `self._chat_history`
- `self._last_outside_message` 记录最后一条外部请求文本（用于 prompt）

### 4.2 Pause/Cancel 机制（贯穿整个执行）
- `pause()`/`resume()` 通过 `_pause_event` 控制
- `monitor_pause()` 作为后台 task：一旦 paused，会 cancel 用于 LLM/工具的 cancellation_token
- 在 `_execute_tool()` 中也会监控 `_pause_event`，必要时 cancel 当前工具执行 task

### 4.3 单次 step 的“行动-观察”循环（简化伪码）

```
lazy_init()
for step_action in range(max_actions_per_step):
  response, rects, tools, id_map, need_tool = _get_llm_response()

  if not need_tool:
    yield TextMessage(模型给出的自然语言回应)
    break

  for tool_call in response:   # 可能多个 tool calls
    maybe ask approval (action_guard)
    action_result = _execute_tool(tool_call)
    screenshot = get_screenshot()
    yield MultiModalMessage([action_result, screenshot], type=browser_screenshot)
    append "Observation: ..." to chat_history
    if tool in {stop_action, answer_question}: break
```

最后还会构造一次“面向其他 agents 的最终输出”：
- 汇总本轮所有 actions + observations
- 附上当前页面 `describe_page()` 的文字摘要 + 截图
- 以 `metadata={"internal": "yes"}` 的 `MultiModalMessage` 作为最终 `Response` 产出

---

## 5. LLM 决策：`_get_llm_response`（构造页面上下文 + 调用模型）

位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:995`

### 5.1 上下文构造（History + Page State）
核心步骤：
1. **SystemMessage**：`WEB_SURFER_SYSTEM_MESSAGE`（含 tool 说明、策略、日期）
2. **历史过滤**：
   - 只保留 user/user_proxy 的 `UserMessage` 原样（含图片）
   - 其他消息会 `remove_images(...)`（避免历史图片占用 token；也避免把“上一步截图”喂进上下文）
3. **页面状态采样**：
   - `get_interactive_rects(page)` → `rects: Dict[str, InteractiveRegion]`
   - `get_screenshot(page)` → 原始 screenshot bytes
   - `add_set_of_mark(screenshot, rects, use_sequential_ids=True)` → SoM 截图 + 可见/上下方元素 + `element_id_mapping`
4. **tools 列表（动态）**：
   - 基础：`self.default_tools`
   - `single_tab_mode=False`：保证 `create_tab` 可用
   - 多 tab 时：追加 `switch_tab`（`close_tab` 默认未启用）
   - 页面存在 role=option 时：追加 `select_option`
5. **focused hint**：
   - `get_focused_rect_id(page)` → 当前 focus 的元素 id
   - 转成 SoM 的“序号 id”，生成提示：`The <role> with ID <id> ... currently has the input focus.`
6. **可交互目标列表**：
   - `visible_targets`：把可见元素格式化成 JSON 行（包含 `id/name/role/tools`）
   - `other_targets_str`：把 viewport 外元素的 *名字* 拼成一段提示（强调需要 scroll 才能交互）
7. `webpage_text`：`get_visible_text(page)`（只取可见文本，给模型更多语义线索）

### 5.2 Prompt 选择：tools vs no-tools(JSON) 模式
- 默认：`WEB_SURFER_TOOL_PROMPT`（配合 function calling + `tools=ToolSchema[]`）
- 当 `json_model_output=True`（或模型不支持 function_calling）：用 `WEB_SURFER_NO_TOOLS_PROMPT` 要求模型输出严格 JSON：`{"tool_name": ..., "tool_args": ..., "explanation": ...}`

> 注意：`WebSurfer.__init__` 会在模型不支持 function_calling 时强制 `json_model_output=True`。

### 5.3 多模态输入：同时喂 SoM 截图 + 原始截图
当 `model_client.model_info["vision"] == True`：
- 会把 SoM 截图 resize 到 `MLM_WIDTH/MLM_HEIGHT`
- 同时附上一张“未标号”的原始截图（同样 resize）
- 组成 `UserMessage(content=[text_prompt, som_image, raw_image])`

### 5.4 调用模型：工具调用 or JSON
当 `json_model_output=False`：
- `model_client.create(messages, tools=tools, ...)`
- 对指定 family（`gpt-4o/gpt-41/gpt-45/o3/o4`）额外设置：
  - `tool_choice="required"`
  - 若 `multiple_tools_per_call=True`：`parallel_tool_calls=True`

返回值解析：
- `response.content` 是 `list[FunctionCall]` → `need_execute_tool=True`
- 否则是 `str` → `need_execute_tool=False`（WebSurfer 将直接输出文本而不执行浏览器动作）

当 `json_model_output=True`：
- 尝试 parse `response.content` 为 JSON（支持 ```json fenced code block）
- 再包装成 `FunctionCall(id="json_response", name=tool_name, arguments=...)`

---

## 6. Set-of-Mark：给截图打编号与 ID 映射

位置：`src/magentic_ui/agents/web_surfer/_set_of_mark.py`

### 6.1 为什么要 SoM？
Playwright 抓到的 `interactive_rects` 的 id 通常是内部 DOM 生成的字符串/序号，直接暴露给模型不利于稳定引用。

SoM 做两件事：
1. 在截图上把可交互区域画红框，并在框角落标上 **连续数字 ID**
2. 返回 `id_mapping: Dict[new_id, original_id]`，让执行阶段能把模型说的 “3号按钮” 映射回真实元素 id

### 6.2 可见/上方/下方元素分类
`_add_set_of_mark(...)` 会遍历每个元素的 rect，按 rect 中心点的 y 位置归类：
- `visible_rects`：在当前 viewport 内
- `rects_above`：在 viewport 上方（需要 scroll up）
- `rects_below`：在 viewport 下方（需要 scroll down）

### 6.3 映射如何在 WebSurfer 中被使用
在 `_get_llm_response`：
- `element_id_mapping` 是 `new_id -> original_id`
- WebSurfer 会把 `rects` 的 key 改写成 `new_id`（以便 prompt 中的 targets 列表与截图编号一致）
- 在执行时（如 click/input_text），再通过 `element_id_mapping[new_id]` 找回真实 id 执行 Playwright 操作

---

## 7. Tool 系统：Schema 定义与执行映射

### 7.1 ToolSchema 定义在哪里？
位置：`src/magentic_ui/agents/web_surfer/_tool_definitions.py`
- 用 `load_tool({...})` 声明工具 schema（name/description/parameters）
- 同时在 `src/magentic_ui/tools/tool_metadata.py` 内部缓存 metadata（例如 `requires_approval`）

### 7.2 WebSurfer 实际启用的工具集合（默认行为）
在 `WebSurfer.__init__` 中的 `self.default_tools`，以及 `_get_llm_response` 的动态追加逻辑，最终“实际传给模型的 tools”大体是：
- `stop_action(answer)`
- `visit_url(url)`
- `web_search(query)`
- `click(target_id)`
- `input_text(input_field_id, text_value, press_enter, delete_existing_text)`
- `answer_question(question)`
- `sleep(duration)`
- `hover(target_id)`
- `history_back()`
- `keypress(keys)`
- `refresh_page()`
- `scroll_down()`
- `scroll_up()`
- `create_tab(url)`（非 `single_tab_mode` 时）
- `switch_tab(tab_index)`（多 tab 时）
- `select_option(target_id)`（页面存在 role=option 时）

> 提醒：`_prompts.py` 里的 tool 列表描述是一个“能力宣称”，不一定等于本轮实际 `tools=` 里启用的集合；真实可用工具以 `_get_llm_response` 传给 `model_client.create(..., tools=tools)` 的 `tools` 为准。

### 7.3 工具执行：`_execute_tool` 的命名约定
位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:1704`

执行机制：
- 约定：tool 名称 `name` → 方法名 `_execute_tool_{name}`
- `_execute_tool(...)` 会根据工具类型补充参数：
  - 对需要 element 的工具（`click/input_text/hover/select_option/upload_file/click_full`）会额外传入 `rects` 和 `element_id_mapping`
  - 对阅读型工具（`answer_question/summarize_page`）会传入 `cancellation_token`
- 执行时用 `asyncio.create_task(...)` 运行，若 paused 会 cancel task
- 执行后统一调用 `cleanup_animations(page)`
- 若发生下载（`self._last_download` 被 download handler 捕获），会把保存路径拼到 action_description 里返回

---

## 8. 审批与安全：URL 白名单 + ActionGuard

WebSurfer 有两层“需要用户确认”的机制：

### 8.1 URL 访问控制：`UrlStatusManager` + 域名临时授权
相关文件：
- `src/magentic_ui/tools/url_status_manager.py`
- `src/magentic_ui/agents/web_surfer/_web_surfer.py:_check_url_and_generate_msg`

规则：
- `url_statuses is None`：默认 **所有 URL 允许**
- 否则：
  - `is_url_blocked(url)`：命中 blocklist，直接拒绝
  - `is_url_allowed(url) == False`：如果该域名尚未被显式拒绝，会（在 action_guard 存在时）询问用户是否临时允许该 domain
  - 用户拒绝后会把该 domain 记为 `rejected`，并在后续访问中直接拒绝

注意点：
- 这里的授权提示文案有 UI 侧依赖（代码注释提示 “UI 检测特定 wording”），所以修改提示文案要小心影响前端交互。

### 8.2 工具动作审批：`ApprovalGuard.requires_approval(...)`
相关文件：
- `src/magentic_ui/approval_guard.py`
- `src/magentic_ui/tools/tool_metadata.py`
- `src/magentic_ui/agents/web_surfer/_web_surfer.py`（`require_approval` 计算与 `get_approval` 调用）

机制：
1. 每个工具 schema 在 `metadata` 可标记 `requires_approval: "always" | "maybe" | "never"`（由 `load_tool` 缓存）
2. WebSurfer 在执行每个 tool call 前会：
   - 计算 baseline（来自工具 metadata）
   - 若 tool_args 里显式带 `require_approval`（仅用于 baseline=maybe 的语义），会把 LLM guess 强制为 always/never
   - 调用 `action_guard.requires_approval(baseline, llm_guess, chat_history)` 决定是否询问用户
3. 若需要询问：
   - 先用 `PlaywrightController.preview_action(...)` 预览高亮目标元素（如果该工具有 target_id/input_field_id）
   - 再 `action_guard.get_approval(TextMessage(...))`
   - 用户拒绝则停止本次工具执行，并 `cleanup_animations(...)`

> 当前仓库版本的 `src/magentic_ui/agents/web_surfer/_tool_definitions.py` 中，工具 metadata 的 `requires_approval` 全部是 `"never"`，因此“动作审批”通常不会触发；真正会触发的更多是 **URL 域名授权**（当你启用了 `allowed_websites` / `url_statuses` 且 action_guard 存在时）。

---

## 9. `answer_question`：整页阅读/问答（Markdown + 截图）

实现位置：
- 工具 schema：`src/magentic_ui/agents/web_surfer/_tool_definitions.py:TOOL_READ_PAGE_AND_ANSWER`（tool name 是 `answer_question`）
- 执行：`src/magentic_ui/agents/web_surfer/_web_surfer.py:_execute_tool_answer_question` → `_summarize_page`
- prompts：`src/magentic_ui/agents/web_surfer/_prompts.py:WEB_SURFER_QA_SYSTEM_MESSAGE` + `WEB_SURFER_QA_PROMPT`

行为：
1. 用 `PlaywrightController.get_page_markdown(page)` 抓取整页内容（不仅是 viewport）
2. 再截一张当前 viewport 截图，作为多模态输入的一部分
3. 组装 prompt（带 question 或 summary 需求）
4. 通过 tiktoken 做 token budget（当前实现固定用 `encoding_for_model("gpt-4o")`，并假设上下文上限约 `128000`）
5. 调用 `model_client.create(...)` 生成回答

该工具被 WebSurfer 视为“non_action tool”（`non_action_tools = ["stop_action", "answer_question"]`）：
- 一旦执行完成，就会结束本次 `on_messages_stream` 的 action loop（避免继续点击导致页面变化与回答不一致）。

---

## 10. 状态持久化：`save_state` / `load_state`

位置：`src/magentic_ui/agents/web_surfer/_web_surfer.py:2093`

### 10.1 `save_state(save_browser=True)`
- 未 lazy init 时：只保存 `chat_history`，`browser_state=None`
- 已 init 且 `save_browser=True`：
  - `save_browser_state(context, page)` → `BrowserState`
  - 存到 `WebSurferState(browser_state=...)`

### 10.2 `load_state(state, load_browser=True)`
- 反序列化为 `WebSurferState`
- 恢复 `_chat_history`
- 若 `load_browser=True` 且 `browser_state` 存在：
  - `lazy_init()`
  - `load_browser_state(context, browser_state)`
  - 把 `self._page` 指向 active tab（`activeTabIndex`）

该能力常用于：
- UI/后端的断点续跑（team state）
- Sentinel 轮询中“每次 check 复用同一初始 agent 状态”式的执行（由 orchestrator 控制）

---

## 11. 与 teams/orchestrator 的集成方式

典型团队组装见 `src/magentic_ui/task_team.py`：
- `WebSurferConfig(...)` 创建 web_surfer agent
- `GroupChat(participants=[web_surfer, user_proxy, ...], orchestrator_config=..., ...)` 组装 team

### 11.1 单独跑 WebSurfer 的最小示例（参考样例）
仓库自带了一个最小可运行样例：`samples/sample_web_surfer.py`，它用 `RoundRobinGroupChat([web_surfer, user_proxy])` 把 WebSurfer 跑起来，适合用来：
- 验证浏览器是否能启动（local / docker / novnc）
- 观察 WebSurfer 每一步输出的 `TextMessage`（动作解释）与 `MultiModalMessage`（`type=browser_screenshot`）

运行方式（示例）：
- 本地浏览器：`python samples/sample_web_surfer.py`
- docker + noVNC：`python samples/sample_web_surfer.py --port 3000 --novnc-port 6080`

运行时的消息驱动（简述）：
- orchestrator 决定下一步由 `web_surfer` 执行后，会对 web_surfer 的 topic 发布 `GroupChatRequestPublish`
- AutoGen 的 `ChatAgentContainer` 收到请求后调用 `WebSurfer.on_messages_stream(buffered_messages, ...)`
- WebSurfer 产生的 `TextMessage/MultiModalMessage` 会被 container 逐条发布到 output topic（前端 UI 可见）
- 最后一个 `Response` 会作为 `GroupChatAgentResponse` 发回 orchestrator（让它继续调度下一位 agent）

更完整的 teams 消息流见：`docs/TEAMS_DESIGN_CN.md`。

---

## 12. 扩展与踩坑：如何启用/新增工具、常见问题

### 12.1 启用当前“已实现但默认没开”的能力
在 `_web_surfer.py` 里可以看到部分 `_execute_tool_*` 已实现，但默认 tools 列表里没启用（例如 `click_full`、`close_tab`、`upload_file`、`summarize_page`）。

启用步骤（通用思路）：
1. 在 `src/magentic_ui/agents/web_surfer/_tool_definitions.py` 确认工具 schema 已定义并导出
2. 在 `src/magentic_ui/agents/web_surfer/_web_surfer.py`：
   - import 对应 `TOOL_*`
   - 加入 `self.default_tools` 或 `_get_llm_response` 的动态追加逻辑
3. 确认执行函数存在：`_execute_tool_{tool_name}`
4. 如涉及审批：
   - 给 tool schema 的 metadata 设置合适的 `requires_approval`（并确保 UI/ApprovalGuard 策略符合预期）

### 12.2 “prompt 里写了某个工具，但模型调用时报 Unknown tool”
原因通常是：`_prompts.py` 的“工具宣称列表”是 superset，但 `_get_llm_response` 传给 `model_client.create(..., tools=...)` 的 `tools` 没包含它。

排查建议：
- 以 `_get_llm_response` 里构造的 `tools` 为准（这才是模型真正可调用的集合）
- 若要稳定减少误调用，可同步精简 `WEB_SURFER_SYSTEM_MESSAGE` / `WEB_SURFER_NO_TOOLS_PROMPT` 的工具列表

### 12.3 “为什么每一步都给两张图（SoM + 原图）？”
目的不同：
- SoM 图：让模型能用数字稳定引用元素（ID 与红框对应）
- 原图：保留视觉细节（字体、布局、颜色、非交互信息），提升模型理解

### 12.4 “为什么历史里不保留 web_surfer 自己的截图？”
`_get_llm_response` 过滤历史时只保留 user/user_proxy 的图片，其他消息会 `remove_images`：
- 减少 token/带宽
- 避免把旧截图当成当前页面
- 当前页面会在每轮决策时重新采样并附图

---

## 附：关键文件索引
- 主实现：`src/magentic_ui/agents/web_surfer/_web_surfer.py`
- prompts：`src/magentic_ui/agents/web_surfer/_prompts.py`
- tool schemas：`src/magentic_ui/agents/web_surfer/_tool_definitions.py`
- SoM：`src/magentic_ui/agents/web_surfer/_set_of_mark.py`
- Playwright 控制层：`src/magentic_ui/tools/playwright/playwright_controller.py`
- URL 管控：`src/magentic_ui/tools/url_status_manager.py`
- 审批守卫：`src/magentic_ui/approval_guard.py`
