# Magentic-UI `teams` 设计与运行机制（中文）

本文档在 `docs/HR_AGENT_GUIDE_CN.md` 的基础上，专门聚焦于 `src/magentic_ui/teams` 这一层：**团队（Team）如何被组装、如何通过 AutoGen 的事件/消息总线跑起来、Orchestrator（协调器）用哪些 prompts 做计划与执行、以及 Sentinel（长时监控）步骤如何触发与循环**。

> 适用范围：当前仓库版本的 `magentic_ui.teams`（自定义 `GroupChat` + `Orchestrator` + `RoundRobinGroupChat`）。

---

## 目录
1. [目录结构与职责划分](#1-目录结构与职责划分)
2. [核心抽象：Team / Manager / Container / Topic](#2-核心抽象team--manager--container--topic)
3. [团队如何被组装（入口：`get_task_team`）](#3-团队如何被组装入口get_task_team)
4. [AutoGen 如何触发 `@rpc` / `@event`（消息流与事件流）](#4-autogen-如何触发-rpc--event消息流与事件流)
5. [Orchestrator 工作流：Planning → Execution](#5-orchestrator-工作流planning--execution)
6. [Orchestrator 使用的 prompts（计划、进度账本、最终回答、Sentinel 条件检查）](#6-orchestrator-使用的-prompts计划进度账本最终回答sentinel-条件检查)
7. [SentinelPlanStep：长时监控/周期任务如何跑起来](#7-sentinelplanstep长时监控周期任务如何跑起来)
8. [RoundRobinGroupChat：简化版 teams（websurfer_loop）](#8-roundrobingroupchat简化版-teamswebsurfer_loop)
9. [如何运行与端到端串起来（CLI / Backend）](#9-如何运行与端到端串起来cli--backend)
10. [扩展指南：增加新 Agent / 调整编排逻辑](#10-扩展指南增加新-agent--调整编排逻辑)

---

## 1. 目录结构与职责划分

`src/magentic_ui/teams` 目录主要有两套 team：

```
src/magentic_ui/teams/
├── __init__.py
├── orchestrator/
│   ├── __init__.py
│   ├── _group_chat.py
│   ├── _orchestrator.py
│   ├── _prompts.py
│   ├── _sentinel_prompts.py
│   ├── _utils.py
│   └── orchestrator_config.py
└── roundrobin_orchestrator.py
```

### 1.1 `orchestrator/`（主线：规划 + 执行 + Sentinel）
- `src/magentic_ui/teams/orchestrator/_group_chat.py`
  - 定义 `GroupChat` team（继承 AutoGen 的 `BaseGroupChat`）。
  - 关键点：创建 `Orchestrator`（group chat manager），并在 `run_stream()` 中额外产出 `CheckpointEvent`（用于 UI/运行时持久化）。
- `src/magentic_ui/teams/orchestrator/_orchestrator.py`
  - 核心协调器 `Orchestrator`（继承 `BaseGroupChatManager`）。
  - 实现两阶段编排：Planning（生成/修订计划）与 Execution（逐步执行 + 可能 Replan + 最终回答）。
  - 实现 SentinelPlanStep 的循环执行、条件检查、动态 sleep、pause/resume。
- `src/magentic_ui/teams/orchestrator/_prompts.py`
  - Orchestrator 的核心 prompts：计划 prompt、progress ledger prompt、final answer prompt 等。
- `src/magentic_ui/teams/orchestrator/_sentinel_prompts.py`
  - SentinelPlanStep 的条件检查 prompt + JSON 校验。
- `src/magentic_ui/teams/orchestrator/orchestrator_config.py`
  - `OrchestratorConfig`：控制 cooperative planning / autonomous execution / replans / token_limit / sentinel 等。
- `src/magentic_ui/teams/orchestrator/_utils.py`
  - 辅助函数：从用户输入中判断 “accept/run/do it”等，JSON 提取等。

### 1.2 `roundrobin_orchestrator.py`（简化版轮询调度）
- `src/magentic_ui/teams/roundrobin_orchestrator.py`
  - 定义 `RoundRobinGroupChatManager`（轮询选 speaker）与 `RoundRobinGroupChat` team。
  - 主要用在 `websurfer_loop`（例如只跑 web surfer 的循环场景），也产出 `CheckpointEvent`。

---

## 2. 核心抽象：Team / Manager / Container / Topic

Magentic-UI 的 `teams` 建在 AutoGen 0.5.x 的 group chat 基础设施上（`autogen_agentchat.teams._group_chat.*`）。理解它的关键是 4 个角色：

1) **Team（团队）**
- 对应 `GroupChat` / `RoundRobinGroupChat`（都继承 AutoGen 的 `BaseGroupChat`）。
- 对外暴露 `run_stream(task=...)`：产出“消息流 + 最终 TaskResult”。

2) **Group Chat Manager（协调器/调度器）**
- 对应 `Orchestrator` / `RoundRobinGroupChatManager`（都继承 AutoGen 的 `BaseGroupChatManager`）。
- 负责：
  - 接收启动请求（`GroupChatStart`，RPC）。
  - 决定下一位说话者（谁来产出下一个响应）。
  - 通过发布 `GroupChatRequestPublish` 触发某个 agent 真正执行。

3) **ChatAgentContainer（每个参与者外面包一层容器）**
- AutoGen 的设计是：team 内部注册的不是“裸 ChatAgent”，而是一个 container（`autogen_agentchat.teams._group_chat._chat_agent_container.ChatAgentContainer`）。
- container 负责：
  - 缓冲收到的 group 消息（start + 其他 agent 的 response）。
  - 收到 manager 的 `GroupChatRequestPublish` 后，调用底层 `ChatAgent.on_messages_stream(...)`，并把输出转换成 group chat 的标准事件。

4) **Topic（主题/通道）**
AutoGen 在 `BaseGroupChat` 初始化时，会为每个 team 生成一组 topic type（与 team_id 绑定），典型有：
- `group_topic_{team_id}`：广播通道（manager + 所有 participant 都订阅）。
- `{participant_name}_{team_id}`：每个 participant 的专用通道（manager 用它来点名触发某个 agent）。
- `{manager_name}_{team_id}`：manager 的专用通道（team.run_stream 通过 RPC 发 start/reset/pause/resume）。
- `output_topic_{team_id}`：输出通道（container / manager 把消息发布到这里；manager订阅并转存到 output_queue，最终被 `run_stream()` yield 给调用方）。

---

## 3. 团队如何被组装（入口：`get_task_team`）

### 3.1 Team 组装入口
团队并不是直接在 `teams/` 内创建，而是统一在：
- `src/magentic_ui/task_team.py:get_task_team(...)`

它负责：
1. 构造不同 agent 的 model_client（orchestrator/web_surfer/coder/file_surfer/action_guard）。
2. 配置浏览器资源（Playwright/NoVNC）。
3. 创建 participants：`WebSurfer` / `CoderAgent` / `FileSurfer` / `UserProxyAgent`（或 Dummy/Metadata user proxy）/ `McpAgent` 等。
4. 构造 `OrchestratorConfig`（co-planning / autonomous / replans / token_limit / sentinel_plan 等）。
5. 创建 team：
   - 默认：`GroupChat(participants=[...], model_client=model_client_orch, orchestrator_config=..., ...)`
   - 若 `websurfer_loop`：`RoundRobinGroupChat(participants=[web_surfer, user_proxy], max_turns=10000)`

### 3.2 OrchestratorConfig 如何影响 team 行为
`src/magentic_ui/teams/orchestrator/orchestrator_config.py:OrchestratorConfig` 里最关键的开关：
- `cooperative_planning`：是否进入“用户确认计划”的共创模式（plan 生成后先让 user_proxy 反馈/accept）。
- `autonomous_execution`：是否执行阶段完全不向用户提问（会把 user_proxy 从执行候选列表中剔除）。
- `allow_for_replans` + `max_replans`：执行中是否允许重规划，以及最多几次。
- `do_bing_search`：planning 时是否做 bing search（本项目里通常由 UI/配置控制）。
- `retrieve_relevant_plans` + `memory_controller_key`：是否从 memory 中提示/复用历史 plan。
- `sentinel_plan.enable_sentinel_steps`：是否允许生成并执行 SentinelPlanStep。
- `sentinel_plan.dynamic_sentinel_sleep`：Sentinel 条件未满足时，是否允许 LLM 动态给出 sleep_duration。

---

## 4. AutoGen 如何触发 `@rpc` / `@event`（消息流与事件流）

### 4.1 两种触发方式：Direct RPC vs Pub/Sub Event
AutoGen Runtime（`autogen_core.AgentRuntime`）有两类通信：

1) `send_message(...)`（RPC / 点对点）
- 调用方显式指定 `recipient=AgentId(...)`。
- 收件 agent 的 `@rpc` handler 被触发。
- 有返回值（但 group chat start 通常不依赖返回值）。

2) `publish_message(...)`（Event / 主题广播）
- 发布到 `TopicId`，由订阅该 topic 的 agent 接收。
- 收件 agent 的 `@event` handler 被触发。
- 无返回值。
- **重要细节**：AutoGen 的 runtime 默认 *不会把 publish 的消息回投给 sender*（避免自触发循环）。

### 4.2 group chat 的标准事件类型
AutoGen group chat 使用一组结构化消息（见 `autogen_agentchat.teams._group_chat._events`）：
- `GroupChatStart(messages=[...])`：启动时广播给所有参与者（让 container 缓冲初始消息）。
- `GroupChatRequestPublish()`：manager 点名让某个 participant “出声”（触发它执行 `on_messages_stream`）。
- `GroupChatAgentResponse(agent_response=Response(...))`：participant 执行完毕后发布给 group topic（manager 接收并继续调度）。
- `GroupChatMessage(message=...)`：把过程消息（中间步骤、工具输出、文本/多模态消息等）发到 output topic（给 UI/调用方看）。
- `GroupChatTermination(...)`：team 终止信号（最后一次放入 output queue，`run_stream()`据此结束）。

### 4.3 端到端消息流（最重要的一张图）

下面的流程图同时解释了“AutoGen 怎么 trigger event”和“teams 怎么跑起来”：

```
调用方 (CLI/UI)
  |
  | team.run_stream(task=...)
  v
BaseGroupChat.run_stream()
  |
  | runtime.send_message(GroupChatStart, recipient=manager_topic)   # 触发 @rpc
  v
Orchestrator.handle_start()  [@rpc]  (src/magentic_ui/teams/orchestrator/_orchestrator.py)
  |
  | publish GroupChatStart to group_topic                          # 触发 participants 的 @event handle_start
  | _orchestrate_step() -> 决定下一步（可能先 planning）
  | publish GroupChatRequestPublish to {speaker}_topic             # 触发某个 participant container 的 @event handle_request
  v
ChatAgentContainer.handle_request() [@event]  (AutoGen 内部)
  |
  | agent.on_messages_stream(buffered_messages)                    # 真正运行 agent
  | publish GroupChatMessage to output_topic                       # UI可见的流式消息
  | publish GroupChatAgentResponse to group_topic                  # 触发 manager 的 @event handle_agent_response
  v
Orchestrator.handle_agent_response() [@event]
  |
  | 更新 state/message_history
  | _orchestrate_step() -> 继续调度下一位 speaker 或收敛到 final answer
  v
(循环直到)
  |
  | publish GroupChatTermination to output_topic / output_queue
  v
BaseGroupChat.run_stream() yield TaskResult(stop_reason=...)
```

你可以把 `@rpc` 理解为“点对点命令入口”（start/reset/pause/resume），把 `@event` 理解为“订阅主题的消息驱动回调”（agent response / output message / request publish）。

---

## 5. Orchestrator 工作流：Planning → Execution

`src/magentic_ui/teams/orchestrator/_orchestrator.py:Orchestrator` 维护一个 `OrchestratorState`（任务、计划、当前 step、消息历史、是否暂停等），并且把整个运行拆成两个阶段：

### 5.1 Planning（计划阶段）
触发点：
- `handle_start()` 把初始 user message 放入 `message_history` 后，进入 `_orchestrate_step()`。
- `_state.in_planning_mode == True` 时走 `_orchestrate_step_planning()`。

Planning 的核心行为：
1. 解析 user_proxy 输入：
   - user_proxy 可以返回纯文本，也可以返回 JSON（`HumanInputFormat`），例如：
     - `{"content": "...", "accepted": true, "plan": {...}}`
   - Orchestrator 用 `HumanInputFormat.from_str(...)` 解析，并用 `_utils.is_accepted_str(...)` 兼容“accept/run/do it”等自然语言。
2. 计划来源优先级（概念上）：
   - 若 `OrchestratorConfig.plan` 预先给了 plan：直接使用该 plan。
   - 否则若 `retrieve_relevant_plans == "reuse"`：从 memory 找到最相关 plan 并作为初始 plan。
   - 否则调用 LLM 生成 plan（见第 6 节 prompts）。
3. cooperative planning（共创）：
   - 如果 `cooperative_planning=True`：
     - Orchestrator 会把 plan 作为 `type=plan_message` 的消息发给用户（通过 output_topic）。
     - 下一 speaker 通常是 `user_proxy`，等待用户“accept / 修改 / 提问”。
   - 如果 `cooperative_planning=False`：
     - plan 生成后立即切换到 execution 并开始跑第一个 step。

输出形态：
- plan 消息通常以 JSON 字符串形式发给 UI，metadata 会带 `{"type": "plan_message"}`（便于前端做特殊渲染）。

### 5.2 Execution（执行阶段）
触发点：
- 用户 accept plan，或 `cooperative_planning=False` 自动进入。
关键循环逻辑在 `_orchestrate_step_execution(...)`：

1. 首次进入执行时，会构造一个“task ledger”内部消息（包含 task/team/plan），作为执行阶段的长期上下文基座（主要给 LLM 做进度判断与调度）。
2. 每一轮执行都让 LLM 产出一个 **progress ledger JSON**（见第 6.3 节），用于回答：
   - 当前 step 是否完成？
   - 是否需要 replan？
   - 下一步应该由哪个 agent 执行，以及给它什么指令？
   - 当前整体进展总结（用于最终回答的 `information_collected`）
3. 若 `need_to_replan == True`：
   - 进入 `_replan()`：把已完成 steps + 失败原因写进 replan prompt，再让 LLM 产出新 steps，并拼接回当前 plan。
4. 若当前 step 完成：
   - `current_step_idx += 1`
   - plan 全部完成则进入 `_prepare_final_answer(...)`。
5. 若当前 step 未完成：
   - Orchestrator 把 `instruction_or_question.answer` 包装成统一格式 `INSTRUCTION_AGENT_FORMAT`，并通过 `GroupChatRequestPublish` 点名下一位 agent 去执行。

执行阶段对外输出两个非常关键的 message 类型（metadata）：
- `type=step_execution`：每次选定下一步执行时，发一个 JSON（step title/index/details/agent_name/instruction/progress_summary...），便于 UI 显示“正在执行第几步”。
- `type=final_answer`：最终回答。

---

## 6. Orchestrator 使用的 prompts（计划、进度账本、最终回答、Sentinel 条件检查）

Orchestrator 的 prompts 全部集中在：
- `src/magentic_ui/teams/orchestrator/_prompts.py`
- `src/magentic_ui/teams/orchestrator/_sentinel_prompts.py`

下面按功能拆解（只展示结构与关键字段，完整文案以源码为准）。

### 6.1 Planning：System message（两种模式）
- cooperative planning：`get_orchestrator_system_message_planning(...)`
- autonomous planning：`get_orchestrator_system_message_planning_autonomous(...)`

共同点：
- 都会注入 `{team}`（agent 列表 + description）与 `{date_today}`。
- 在 sentinel 开启时，会把 `SENTINEL_STEP_TYPES` 与示例拼到 system message 里，指导 LLM 生成 SentinelPlanStep。

### 6.2 Planning：计划生成 JSON schema
`get_orchestrator_plan_prompt_json(...)` 要求模型输出严格 JSON（`validate_plan_json(...)` 会校验）：

```json
{
  "response": "Case 1 时直接回答用户",
  "task": "用户任务的完整描述",
  "plan_summary": "需要 plan 时的摘要，否则空字符串",
  "needs_plan": true,
  "steps": [
    {
      "title": "step 标题",
      "details": "给 agent 的具体指令",
      "agent_name": "web_surfer / coder_agent / file_surfer / user_proxy / ...",
      "step_type": "SentinelPlanStep",
      "condition": "次数(int) 或可验证条件(str)",
      "sleep_duration": "秒"
    }
  ]
}
```

备注：
- Sentinel 未开启时，step 不允许出现 `step_type/condition/sleep_duration`。
- `Plan.from_list_of_dicts_or_str(...)` 会把包含 `condition + sleep_duration` 的 step 解析为 `SentinelPlanStep`（见 `src/magentic_ui/types.py`）。

### 6.3 Execution：Progress Ledger JSON schema（“调度决策器”）
`get_orchestrator_progress_ledger_prompt(...)` 让 LLM 充当“执行调度器”，输出严格 JSON（`validate_ledger_json(...)` 校验）：

```json
{
  "is_current_step_complete": { "reason": "...", "answer": false },
  "need_to_replan": { "reason": "...", "answer": false },
  "instruction_or_question": {
    "answer": "给下一位 agent 的指令/问题",
    "agent_name": "下一位执行者（必须在 names 列表中）"
  },
  "progress_summary": "一两句话的进度摘要"
}
```

Orchestrator 会据此：
- 决定是否 `current_step_idx += 1`
- 是否进入 `_replan()`
- 下一位 speaker 是谁（`GroupChatRequestPublish` 点名）
- 把 `progress_summary` 累积进 `information_collected`（用于 final answer）。

### 6.4 Final Answer prompt
`ORCHESTRATOR_FINAL_ANSWER_PROMPT` 让模型根据全过程消息与 `information_collected` 输出最终回答，并明确“信息来自在线搜索还是内部知识”。

### 6.5 Sentinel 条件检查 prompt（只在 string condition 时使用）
当 `SentinelPlanStep.condition` 是字符串时，Orchestrator 用 `_sentinel_prompts.py` 的 `ORCHESTRATOR_SENTINEL_CONDITION_CHECK_PROMPT` 再调用一次 LLM，让其输出：

```json
{
  "reason": "一句话解释",
  "condition_met": false,
  "sleep_duration_reason": "为何建议这个 sleep",
  "sleep_duration": 60,
  "error_encountered": false
}
```

它的核心作用是：
- 把“是否满足条件”变成一个可机器判断的布尔值；
- 在 `dynamic_sentinel_sleep=True` 时提供下一轮 sleep 的建议。

---

## 7. SentinelPlanStep：长时监控/周期任务如何跑起来

Sentinel 是 `teams` 层最“特别”的能力：它允许一个 step **重复执行同一个 agent 行为**，并在每轮结束后判断“条件是否满足”，直到满足或被暂停/取消。

### 7.1 什么时候会走 Sentinel 分支？
在执行阶段，当前 step 满足以下条件就进入 Sentinel 执行：
- `isinstance(current_step, SentinelPlanStep)`
- 且 `OrchestratorConfig.sentinel_plan.enable_sentinel_steps == True`

### 7.2 Sentinel 的主循环（`_execute_sentinel_step`）
位置：`src/magentic_ui/teams/orchestrator/_orchestrator.py:_execute_sentinel_step(...)`

核心逻辑（概念化）：
1. 生成 `sentinel_step_id`，并发送 `type=sentinel_start` 事件给 UI（包含 step_title/details/condition/sleep_duration）。
2. 每次迭代：
   - 如果 paused / cancelled：提前返回（UI 可显示 `type=sentinel_paused`）。
   - 取到目标 agent 的底层实例（`runtime.try_get_underlying_agent_instance(AgentId(...))`），并调用该 agent 的 `on_messages_stream(...)` 执行一次“检查/动作”。
   - 若 condition 是整数：`iteration >= condition` 则完成。
   - 若 condition 是字符串：调用 Sentinel condition-check prompt，让 LLM 判断 `condition_met`。
   - 若完成：发送 `type=sentinel_check`（condition_met=true）+ `type=sentinel_complete`，并推进到下一个 plan step。
   - 若未完成：根据配置选择 sleep_duration：
     - `dynamic_sentinel_sleep=True`：使用 LLM 建议的 `sleep_duration`
     - 否则：使用 step 固定的 `sleep_duration`
   - 发送 `type=sentinel_sleeping`，然后 `asyncio.sleep(sleep_duration)`（期间可被 pause_event 打断）。

### 7.3 Sentinel 的 UI/日志事件类型（metadata.type）
Sentinel 相关事件（都通过 output_topic 流式输出）：
- `sentinel_start`：开始一个 sentinel step（含 sentinel_id）。
- `sentinel_status`：每次 check 开始时的状态（“Performing check #n...”）。
- `sentinel_check`：每次 check 的结果（condition_met true/false，reason，next_check_in 等）。
- `sentinel_sleeping`：进入 sleep（含 sleep_duration 与 timestamp）。
- `sentinel_sleep`：当 dynamic sleep 开启时，补充解释建议 sleep 的原因。
- `sentinel_complete`：条件满足，step 完成。
- `sentinel_paused` / `sentinel_error`：暂停或错误中止。

---

## 8. RoundRobinGroupChat：简化版 teams（websurfer_loop）

当你只想让 agent “轮流说话/轮流执行”而不需要复杂的 plan + ledger（例如只跑 web agent），可以用：
- `src/magentic_ui/teams/roundrobin_orchestrator.py:RoundRobinGroupChat`

特点：
- `RoundRobinGroupChatManager.select_speaker(...)` 按 index 轮询。
- 支持 paused 时优先让 `user_proxy` 说话（如果存在）。
- 同样在 `run_stream()` 中产出 `CheckpointEvent`，便于 UI 持久化中间状态。

在 `src/magentic_ui/task_team.py` 中，当 `magentic_ui_config.websurfer_loop == True` 时会走这个分支。

---

## 9. 如何运行与端到端串起来（CLI / Backend）

### 9.1 CLI 跑法（本地终端）
入口：
- `src/magentic_ui/_cli.py`

关键流程：
1. `team = await get_task_team(...)`
2. `stream = team.run_stream(task=task)`
3. CLI/Console 迭代 `async for message in stream: ...`，把消息渲染到终端。
4. 最后 `await team.save_state()` 持久化，`await team.close()` 释放资源。

### 9.2 Backend 跑法（前端 UI）
入口：
- `src/magentic_ui/backend/teammanager/teammanager.py`

关键流程：
1. `_create_team(...)` 内部调用 `get_task_team(...)` 创建 team。
2. `async for message in self.team.run_stream(task=task, ...)` 把消息逐条 yield 给 websocket/前端。
3. 额外逻辑：
   - 追踪运行目录中新生成文件，并发 `type=file` 的系统消息给前端。
   - 监听 `type=final_answer` 时把文件列表合并发给前端。
   - 支持 `pause_run()` / `resume_run()`，内部调用 `team.pause()` / `team.resume()`。

### 9.3 最小运行示例（命令 / 代码）

命令行脚本入口在 `pyproject.toml` 的 `[project.scripts]`：
- 终端交互（偏“开发调试”）：`magentic-cli` → `magentic_ui._cli:main`
- Web UI/后端（偏“产品形态”）：`magentic-ui`/`magentic` → `magentic_ui.backend.cli:run`

代码级最小思路（核心就两句：`get_task_team(...)` + `team.run_stream(...)`）：

```python
from pathlib import Path
import asyncio

from magentic_ui.magentic_ui_config import MagenticUIConfig
from magentic_ui.task_team import get_task_team
from magentic_ui.types import RunPaths


async def main() -> None:
    # paths 需要是可写目录；实际项目里 CLI/Backend 会自动准备这些路径
    run_dir = Path("./.runs/dev")
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        internal_root_dir=run_dir,
        external_root_dir=run_dir,
        run_suffix="dev",
        internal_run_dir=run_dir,
        external_run_dir=run_dir,
    )

    team = await get_task_team(magentic_ui_config=MagenticUIConfig(), paths=paths)
    async for msg in team.run_stream(task="解释一下 teams 是怎么工作的"):
        print(msg)


asyncio.run(main())
```

---

## 10. 扩展指南：增加新 Agent / 调整编排逻辑

### 10.1 增加一个新的参与者 Agent（最常见的扩展）
建议流程：
1. 实现一个新的 `ChatAgent`（遵循 AutoGen `on_messages_stream`/`on_reset` 等协议）。
2. 在 `src/magentic_ui/task_team.py:get_task_team` 中实例化它，并加入 `team_participants`。
3. 保证 `agent.name` 唯一（group chat 要求 participant names 唯一）。
4. 确保它产出的消息类型能被 `MessageFactory` 识别（通常 `TextMessage`/`MultiModalMessage` 已覆盖；自定义 `StructuredMessage` 需要注册）。
5. Orchestrator 侧无需硬编码支持：它会把 team 描述注入 prompts，并在 ledger 校验时限制 `agent_name` 必须属于 participants。

### 10.2 调整 Orchestrator 的规划/执行策略
你通常只需要改两类东西：
- prompts：`src/magentic_ui/teams/orchestrator/_prompts.py` / `_sentinel_prompts.py`
  - 如果你改了 JSON schema，务必同步更新 `validate_plan_json` / `validate_ledger_json`。
- 调度逻辑：`src/magentic_ui/teams/orchestrator/_orchestrator.py`
  - 例如：改变什么时候 replan、怎样选下一位 agent、是否在 planning 阶段允许某些 agent 自动执行等。

---

## 附：关键文件索引（从“理解运行”角度）
- Team 组装：`src/magentic_ui/task_team.py`
- Team 实现：`src/magentic_ui/teams/orchestrator/_group_chat.py`、`src/magentic_ui/teams/roundrobin_orchestrator.py`
- 协调器/编排核心：`src/magentic_ui/teams/orchestrator/_orchestrator.py`
- Prompts：`src/magentic_ui/teams/orchestrator/_prompts.py`、`src/magentic_ui/teams/orchestrator/_sentinel_prompts.py`
- Plan/Step 数据结构：`src/magentic_ui/types.py`
- CLI 运行：`src/magentic_ui/_cli.py`
- Backend 运行：`src/magentic_ui/backend/teammanager/teammanager.py`
