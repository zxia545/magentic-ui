# Magentic-UI HR Talent Search Agent 开发指南

> 详细的中文教程：学习项目架构，构建你自己的HR人才搜索Agent

## 目录
1. [项目结构概览](#1-项目结构概览)
2. [AutoGen集成与Agent架构](#2-autogen集成与agent架构)
3. [Agent拆解与实现](#3-agent拆解与实现)
4. [Tool系统详解](#4-tool系统详解)
5. [Playwright Docker浏览器](#5-playwright-docker浏览器)
6. [实现HR Talent Search Agent](#6-实现hr-talent-search-agent)
7. [完整开发流程](#7-完整开发流程)

---

## 1. 项目结构概览

### 1.1 核心目录结构

```
magentic-ui/
├── src/magentic_ui/           # 核心代码库
│   ├── agents/                 # 所有Agent实现
│   │   ├── web_surfer/        # Web浏览Agent
│   │   ├── file_surfer/       # 文件操作Agent
│   │   ├── mcp/               # MCP协议Agent
│   │   ├── talent_search/     # HR人才搜索(如需自定义)
│   │   ├── _coder.py          # 代码执行Agent
│   │   └── __init__.py        # Agent导出
│   ├── tools/                  # 工具库
│   │   ├── playwright/        # 浏览器控制工具
│   │   └── tool_metadata.py   # 工具元数据
│   ├── backend/                # 后端服务
│   ├── task_team.py           # Agent团队组装
│   ├── magentic_ui_config.py  # 配置系统
│   └── approval_guard.py      # 审批守卫
├── docker/                     # Docker容器
│   └── magentic-ui-browser-docker/  # 浏览器容器
├── prev_hr_agent/             # 之前的HR Agent实现
│   └── kdd_hr_system/         # 学术人才搜索系统
└── samples/                    # 示例代码
    ├── sample_web_surfer.py   # WebSurfer示例
    └── sample_coder.py        # Coder示例
```

### 1.2 关键文件说明

| 文件路径 | 作用 | 重要性 |
|---------|------|--------|
| `task_team.py` | Agent团队组装，系统入口 | ⭐⭐⭐⭐⭐ |
| `agents/web_surfer/_web_surfer.py` | Web浏览核心Agent | ⭐⭐⭐⭐⭐ |
| `agents/web_surfer/_tool_definitions.py` | Web工具定义 | ⭐⭐⭐⭐⭐ |
| `agents/_coder.py` | 代码执行Agent | ⭐⭐⭐⭐ |
| `tools/playwright/playwright_controller.py` | 浏览器控制器 | ⭐⭐⭐⭐⭐ |
| `approval_guard.py` | 操作审批系统 | ⭐⭐⭐ |

---

## 2. AutoGen集成与Agent架构

### 2.1 AutoGen框架介绍

Magentic-UI使用 **Microsoft AutoGen 0.5.x**，这是一个多Agent对话框架：

```python
# 依赖关系 (pyproject.toml)
dependencies = [
    "autogen-agentchat==0.5.7",       # Agent对话核心
    "autogen-core==0.5.7",            # 基础框架
    "autogen-ext[openai,docker,...]", # 扩展功能
]
```

### 2.2 Agent基类架构

所有Agent都继承自 `BaseChatAgent`:

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_core import Component, ComponentModel

class WebSurfer(BaseChatAgent, Component[WebSurferConfig]):
    """
    BaseChatAgent提供:
    - on_messages(): 处理输入消息
    - on_messages_stream(): 流式响应
    - on_reset(): 重置状态
    """
    
    def __init__(self, name: str, model_client: ChatCompletionClient, ...):
        super().__init__(name, description)
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = []
    
    async def on_messages(
        self, 
        messages: Sequence[BaseChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """接收消息并返回响应"""
        pass
```

### 2.3 Agent团队组装 (task_team.py)

这是整个系统的核心组装文件：

```python
async def get_task_team(
    magentic_ui_config: Optional[MagenticUIConfig] = None,
    input_func: Optional[InputFuncType] = None,
    paths: RunPaths,
) -> GroupChat | RoundRobinGroupChat:
    """
    创建Agent团队的完整流程:
    
    1. 配置模型客户端 (OpenAI/Azure/Ollama)
    2. 创建各个Agent实例
    3. 配置审批守卫
    4. 组装GroupChat团队
    """
    
    # 步骤1: 获取模型客户端
    model_client_orch = get_model_client(
        magentic_ui_config.model_client_configs.orchestrator
    )
    
    # 步骤2: 配置浏览器资源
    browser_resource_config, novnc_port, playwright_port = (
        get_browser_resource_config(
            paths.external_run_dir,
            magentic_ui_config.novnc_port,
            magentic_ui_config.playwright_port,
            inside_docker=magentic_ui_config.inside_docker,
        )
    )
    
    # 步骤3: 创建WebSurfer Agent
    websurfer_config = WebSurferConfig(
        name="web_surfer",
        model_client=websurfer_model_client,
        browser=browser_resource_config,
        max_actions_per_step=5,
        # ... 其他配置
    )
    web_surfer = WebSurfer.from_config(websurfer_config)
    
    # 步骤4: 创建Coder Agent (如果不是无Docker模式)
    if not magentic_ui_config.run_without_docker:
        coder_agent = CoderAgent(
            name="coder_agent",
            model_client=model_client_coder,
            work_dir=paths.internal_run_dir,
        )
    
    # 步骤5: 创建FileSurfer Agent
    file_surfer = FileSurfer(
        name="file_surfer",
        model_client=model_client_file_surfer,
        work_dir=paths.internal_run_dir,
    )
    
    # 步骤6: 创建UserProxy
    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func=user_proxy_input_func,
    )
    
    # 步骤7: 组装团队
    team_participants = [web_surfer, user_proxy, coder_agent, file_surfer]
    
    team = GroupChat(
        participants=team_participants,
        orchestrator_config=orchestrator_config,
        model_client=model_client_orch,
    )
    
    return team
```

### 2.4 GroupChat工作流程

```
用户输入
    ↓
Orchestrator (协调器)
    ↓
决策: 选择哪个Agent处理
    ↓
┌─────────────┬─────────────┬─────────────┐
│ WebSurfer   │ CoderAgent  │ FileSurfer  │
│ (浏览网页)  │ (执行代码)  │ (处理文件)  │
└─────────────┴─────────────┴─────────────┘
    ↓
返回结果给Orchestrator
    ↓
Orchestrator决定: 继续 or 完成
    ↓
返回给用户
```

---

## 3. Agent拆解与实现

### 3.1 WebSurfer Agent深度分析

#### 位置
- 主文件: `src/magentic_ui/agents/web_surfer/_web_surfer.py`
- 工具定义: `src/magentic_ui/agents/web_surfer/_tool_definitions.py`
- 提示词: `src/magentic_ui/agents/web_surfer/_prompts.py`

#### 核心功能

```python
class WebSurfer(BaseChatAgent):
    """
    Web浏览Agent的核心能力:
    1. 访问URL和执行搜索
    2. 与网页交互(点击、输入、滚动)
    3. 截图和内容提取
    4. 多标签页管理
    5. 文件下载
    """
    
    def __init__(self, name, model_client, browser, ...):
        # 核心组件
        self._model_client = model_client          # LLM客户端
        self._browser = browser                     # Playwright浏览器
        self._playwright_controller = PlaywrightController(...)  # 控制器
        self._url_status_manager = UrlStatusManager(...)        # URL管理
        
        # 默认工具集
        self.default_tools = [
            TOOL_VISIT_URL,      # 访问URL
            TOOL_WEB_SEARCH,     # 网页搜索
            TOOL_CLICK,          # 点击元素
            TOOL_TYPE,           # 输入文本
            TOOL_SCROLL_DOWN,    # 向下滚动
            TOOL_SCROLL_UP,      # 向上滚动
            # ... 更多工具
        ]
    
    async def on_messages_stream(
        self, 
        messages: Sequence[BaseChatMessage], 
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseChatMessage | Response, None]:
        """
        主循环: 观察 -> 思考 -> 行动
        """
        for _ in range(self.max_actions_per_step):
            # 1. 获取当前页面状态
            screenshot = await self._playwright_controller.get_screenshot(self._page)
            rects = await self._playwright_controller.get_interactive_rects(self._page)
            
            # 2. 使用LLM决定下一步行动
            response = await self._get_llm_response(cancellation_token)
            
            # 3. 执行工具调用
            action_result = await self._execute_tool(response, rects, tools)
            
            # 4. 返回观察结果
            yield Response(chat_message=MultiModalMessage(...))
```

#### WebSurfer执行流程图

```
开始
  ↓
lazy_init() - 初始化浏览器
  ↓
进入主循环 (max_actions_per_step次)
  ↓
获取页面状态:
  - 截图
  - 可交互元素 (rects)
  - Set-of-Mark标注
  ↓
调用LLM决策:
  - 输入: 历史对话 + 当前截图 + 可用工具
  - 输出: FunctionCall列表
  ↓
执行工具:
  - visit_url()
  - web_search()
  - click(target_id)
  - input_text(field_id, text)
  - scroll_down()
  ↓
获取执行结果
  ↓
更新历史记录
  ↓
判断: 继续 or 停止?
  ↓
返回最终响应
```

### 3.2 CoderAgent分析

#### 位置
- `src/magentic_ui/agents/_coder.py`

#### 核心功能

```python
class CoderAgent(BaseChatAgent):
    """
    代码执行Agent:
    - 生成Python代码
    - 在Docker/本地环境执行
    - 调试和错误处理
    """
    
    def __init__(self, name, model_client, work_dir, ...):
        # 代码执行器
        if inside_docker:
            self._code_executor = DockerCommandLineCodeExecutor(
                image="python:3.11-slim",
                work_dir=work_dir,
            )
        else:
            self._code_executor = LocalCommandLineCodeExecutor(
                work_dir=work_dir,
            )
    
    async def _coding_and_debug(
        self,
        system_prompt: str,
        thread: Sequence[BaseChatMessage],
        ...
    ):
        """
        代码生成和调试循环:
        1. LLM生成代码
        2. 提取代码块
        3. 执行代码
        4. 如果出错，将错误信息反馈给LLM
        5. 重复直到成功或达到最大轮次
        """
        for i in range(max_debug_rounds):
            # 生成代码
            create_result = await model_client.create(messages=context)
            
            # 提取代码块
            code_blocks = _extract_markdown_code_blocks(create_result.content)
            
            # 执行代码
            for cb in code_blocks:
                result = await code_executor.execute_code_blocks([cb])
                
                if result.exit_code != 0:
                    # 将错误反馈给LLM继续调试
                    error_message = f"Code failed: {result.output}"
                    continue
```

### 3.3 FileSurfer Agent

#### 位置
- `src/magentic_ui/agents/file_surfer/_file_surfer.py`

#### 核心功能

```python
class FileSurfer(BaseChatAgent):
    """
    文件操作Agent:
    - 列出目录
    - 读取文件内容
    - 搜索文件
    - 理解和摘要文件内容
    """
    
    # 可用工具
    default_tools = [
        list_directory,     # 列出目录
        read_file,          # 读取文件
        find_files,         # 搜索文件
    ]
```

---

## 4. Tool系统详解

### 4.1 Tool定义结构

所有工具都遵循 `ToolSchema` 格式:

```python
from autogen_core.tools import ToolSchema

TOOL_VISIT_URL: ToolSchema = load_tool({
    "type": "function",
    "function": {
        "name": "visit_url",
        "description": "Navigate to a URL using browser",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Explain the action to user"
                },
                "url": {
                    "type": "string",
                    "description": "The URL to visit"
                }
            },
            "required": ["explanation", "url"]
        }
    },
    "metadata": {
        "requires_approval": "maybe"  # 审批策略
    }
})
```

### 4.2 WebSurfer现有工具清单

| 工具名称 | 功能描述 | 参数 | 审批要求 |
|---------|---------|------|---------|
| `visit_url` | 访问指定URL | url, explanation | maybe |
| `web_search` | 执行搜索 | query, explanation | never |
| `click` | 点击元素 | target_id, explanation | maybe |
| `input_text` | 输入文本 | input_field_id, text_value, press_enter, explanation | maybe |
| `scroll_down` | 向下滚动 | explanation | never |
| `scroll_up` | 向上滚动 | explanation | never |
| `history_back` | 后退 | explanation | maybe |
| `refresh_page` | 刷新 | explanation | never |
| `read_page_and_answer` | 阅读页面并回答 | question, explanation | never |
| `sleep` | 等待 | duration, explanation | never |
| `hover` | 悬停 | target_id, explanation | maybe |
| `keypress` | 按键 | key, explanation | maybe |
| `select_option` | 选择下拉项 | target_id, option, explanation | maybe |
| `create_tab` | 创建标签页 | explanation | never |
| `switch_tab` | 切换标签页 | tab_index, explanation | never |
| `stop_action` | 停止并返回 | answer, explanation | never |

### 4.3 Tool执行流程

```python
# 在WebSurfer中
async def _execute_tool(
    self,
    message: List[FunctionCall],
    rects: Dict[str, InteractiveRegion],
    tools: List[ToolSchema],
    ...
) -> str:
    """
    工具执行的通用流程:
    """
    for action in message:
        tool_name = action.name
        tool_args = json.loads(action.arguments)
        
        # 根据工具名称分发
        if tool_name == "visit_url":
            result = await self._execute_tool_visit_url(tool_args)
        elif tool_name == "click":
            result = await self._execute_tool_click(tool_args, rects)
        elif tool_name == "input_text":
            result = await self._execute_tool_type(tool_args, rects)
        # ... 更多工具
        
        return result

# 具体工具实现示例
async def _execute_tool_visit_url(self, args: Dict[str, Any]) -> str:
    """访问URL"""
    url = args["url"]
    
    # 1. 检查URL是否允许
    if not self._url_status_manager.is_url_allowed(url):
        return f"Access to {url} is blocked"
    
    # 2. 使用PlaywrightController访问
    await self._playwright_controller.visit_page(self._page, url)
    
    # 3. 等待加载
    await self._page.wait_for_load_state("domcontentloaded")
    
    return f"Successfully visited {url}"
```

### 4.4 添加自定义Tool

为HR Agent添加LinkedIn搜索工具示例:

```python
# 步骤1: 定义Tool Schema
TOOL_LINKEDIN_SEARCH: ToolSchema = load_tool({
    "type": "function",
    "function": {
        "name": "linkedin_search",
        "description": "Search for candidates on LinkedIn",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": EXPLANATION_TOOL_PROMPT,
                },
                "keywords": {
                    "type": "string",
                    "description": "Search keywords (skills, title, etc.)"
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location"
                },
                "years_experience": {
                    "type": "integer",
                    "description": "Minimum years of experience"
                }
            },
            "required": ["explanation", "keywords"]
        }
    },
    "metadata": {
        "requires_approval": "maybe"
    }
})

# 步骤2: 在WebSurfer中添加工具
class HRWebSurfer(WebSurfer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加自定义工具
        self.default_tools.append(TOOL_LINKEDIN_SEARCH)
    
    # 步骤3: 实现工具执行函数
    async def _execute_tool_linkedin_search(
        self, 
        args: Dict[str, Any]
    ) -> str:
        keywords = args["keywords"]
        location = args.get("location", "")
        years_exp = args.get("years_experience", 0)
        
        # 构建LinkedIn搜索URL
        search_url = f"https://www.linkedin.com/search/results/people/"
        query_params = f"?keywords={keywords}"
        if location:
            query_params += f"&location={location}"
        
        # 访问搜索页面
        await self._playwright_controller.visit_page(
            self._page, 
            search_url + query_params
        )
        
        # 等待结果加载
        await self._page.wait_for_selector(".search-results-container")
        
        # 提取候选人信息
        candidates = await self._page.evaluate("""
            () => {
                const results = [];
                document.querySelectorAll('.reusable-search__result-container').forEach(item => {
                    results.push({
                        name: item.querySelector('.entity-result__title-text')?.innerText,
                        title: item.querySelector('.entity-result__primary-subtitle')?.innerText,
                        location: item.querySelector('.entity-result__secondary-subtitle')?.innerText
                    });
                });
                return results;
            }
        """)
        
        return f"Found {len(candidates)} candidates matching criteria"
```

---

## 5. Playwright Docker浏览器

### 5.1 Docker容器架构

#### Dockerfile位置
- `docker/magentic-ui-browser-docker/Dockerfile`

#### 容器组成

```dockerfile
FROM node:lts-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    xvfb \           # X虚拟帧缓冲
    x11vnc \         # VNC服务器
    openbox \        # 窗口管理器
    supervisor \     # 进程管理
    novnc \          # Web VNC客户端
    websockify       # WebSocket代理

# 安装Playwright
COPY package.json ./
RUN npm install playwright@1.51

# 安装Chromium浏览器
RUN npx playwright install --with-deps chromium

# 暴露端口
EXPOSE 6080 37367
# 6080: noVNC Web界面
# 37367: Playwright WebSocket服务
```

### 5.2 容器启动流程

#### supervisord配置
所有服务通过`supervisord`管理:

```ini
[supervisord]
nodaemon=true

[program:xvfb]
command=/app/x11-setup.sh
priority=1

[program:x11vnc]
command=x11vnc -display :99 -forever -shared
priority=2

[program:novnc]
command=websockify --web=/usr/local/novnc 6080 localhost:5900
priority=3

[program:playwright]
command=node /app/playwright-server.js
priority=4
```

#### Playwright服务器 (playwright-server.js)

```javascript
const { chromium } = require("playwright");

(async () => {
  // 启动Chromium服务器
  const browserServer = await chromium.launchServer({
    headless: false,        // 非无头模式
    port: 37367,           // WebSocket端口
    wsPath: "default",     // WebSocket路径
    args: [
      "--start-fullscreen",
      "--kiosk",           // 全屏模式
      "--disable-infobars",
    ],
  });

  console.log(`Playwright server running: ${browserServer.wsEndpoint()}`);
  // ws://localhost:37367/default
})();
```

### 5.3 Python端连接浏览器

#### VncDockerPlaywrightBrowser类

```python
# src/magentic_ui/tools/playwright/browser/vnc_docker_playwright_browser.py

class VncDockerPlaywrightBrowser(BasePlaywrightBrowser):
    """
    带VNC的Docker浏览器:
    - 启动Docker容器
    - 连接到Playwright WebSocket
    - 提供noVNC访问
    """
    
    def __init__(
        self,
        bind_dir: Path,
        playwright_port: int = 37367,
        novnc_port: int = 6080,
        inside_docker: bool = False,
    ):
        self.playwright_port = playwright_port
        self.novnc_port = novnc_port
        self.bind_dir = bind_dir
        
        # Docker镜像
        self.image = "ghcr.io/microsoft/magentic-ui-browser:latest"
        
        # noVNC访问地址
        self.vnc_address = f"http://localhost:{novnc_port}/vnc.html"
    
    async def __aenter__(self):
        """启动容器并连接"""
        # 1. 启动Docker容器
        await self._start_container()
        
        # 2. 等待Playwright服务就绪
        await self._wait_for_playwright_ready()
        
        # 3. 连接到Playwright
        ws_endpoint = f"ws://localhost:{self.playwright_port}/default"
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect(ws_endpoint)
        
        # 4. 创建浏览器上下文
        self.browser_context = await self.browser.new_context(
            viewport={"width": 1440, "height": 900}
        )
        
        return self
    
    async def _start_container(self):
        """启动Docker容器"""
        client = docker.from_env()
        
        self.container = client.containers.run(
            self.image,
            detach=True,
            ports={
                "6080/tcp": self.novnc_port,    # noVNC
                "37367/tcp": self.playwright_port,  # Playwright
            },
            volumes={
                str(self.bind_dir): {
                    "bind": "/workspace",
                    "mode": "rw"
                }
            },
            environment={
                "PLAYWRIGHT_PORT": str(self.playwright_port),
                "WS_PATH": "default",
            },
            remove=True,  # 停止后自动删除
        )
```

### 5.4 使用示例

```python
from magentic_ui.tools.playwright import VncDockerPlaywrightBrowser
from pathlib import Path

# 创建浏览器实例
browser = VncDockerPlaywrightBrowser(
    bind_dir=Path("/tmp/workspace"),
    playwright_port=37367,
    novnc_port=6080,
)

async with browser:
    # 浏览器已启动，访问noVNC界面
    print(f"Browser UI: {browser.vnc_address}?autoconnect=1")
    # 打开浏览器访问: http://localhost:6080/vnc.html?autoconnect=1
    
    # 创建页面
    context = browser.browser_context
    page = await context.new_page()
    
    # 访问网页
    await page.goto("https://www.linkedin.com")
    
    # 你可以在noVNC界面看到实时浏览器操作
    
    # 截图
    screenshot = await page.screenshot()
```

### 5.5 如何使用Playwright获取网页内容

#### 基本页面操作

```python
from playwright.async_api import Page

async def extract_page_content(page: Page) -> dict:
    """
    从页面提取内容的各种方法
    """
    
    # 1. 获取页面标题
    title = await page.title()
    
    # 2. 获取URL
    url = page.url
    
    # 3. 获取纯文本内容
    text_content = await page.inner_text("body")
    
    # 4. 获取HTML
    html = await page.content()
    
    # 5. 执行JavaScript提取数据
    data = await page.evaluate("""
        () => {
            // 提取所有链接
            const links = Array.from(document.querySelectorAll('a')).map(a => ({
                text: a.innerText,
                href: a.href
            }));
            
            // 提取所有图片
            const images = Array.from(document.querySelectorAll('img')).map(img => ({
                src: img.src,
                alt: img.alt
            }));
            
            return { links, images };
        }
    """)
    
    # 6. 等待特定元素
    await page.wait_for_selector(".profile-card", timeout=5000)
    
    # 7. 提取特定元素内容
    profiles = await page.evaluate("""
        () => {
            return Array.from(document.querySelectorAll('.profile-card')).map(card => ({
                name: card.querySelector('.name')?.innerText,
                title: card.querySelector('.title')?.innerText,
                company: card.querySelector('.company')?.innerText
            }));
        }
    """)
    
    return {
        "title": title,
        "url": url,
        "text": text_content,
        "profiles": profiles,
        "data": data
    }
```

#### LinkedIn搜索示例

```python
async def search_linkedin_talent(
    page: Page,
    keywords: str,
    location: str = "",
) -> list[dict]:
    """
    在LinkedIn搜索人才
    """
    # 1. 访问搜索页面
    search_url = "https://www.linkedin.com/search/results/people/"
    await page.goto(f"{search_url}?keywords={keywords}")
    
    # 2. 等待结果加载
    await page.wait_for_selector(".search-results-container")
    
    # 3. 滚动加载更多结果
    for _ in range(3):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
    
    # 4. 提取候选人信息
    candidates = await page.evaluate("""
        () => {
            const results = [];
            document.querySelectorAll('.reusable-search__result-container').forEach(container => {
                const nameElement = container.querySelector('.entity-result__title-text a');
                const titleElement = container.querySelector('.entity-result__primary-subtitle');
                const locationElement = container.querySelector('.entity-result__secondary-subtitle');
                const profileLink = nameElement?.href;
                
                results.push({
                    name: nameElement?.innerText.trim(),
                    title: titleElement?.innerText.trim(),
                    location: locationElement?.innerText.trim(),
                    profile_url: profileLink,
                });
            });
            return results;
        }
    """)
    
    return candidates

async def get_profile_details(page: Page, profile_url: str) -> dict:
    """
    获取详细的个人资料
    """
    await page.goto(profile_url)
    await page.wait_for_selector(".pv-top-card")
    
    profile = await page.evaluate("""
        () => {
            return {
                name: document.querySelector('.text-heading-xlarge')?.innerText,
                headline: document.querySelector('.text-body-medium')?.innerText,
                location: document.querySelector('.text-body-small.inline')?.innerText,
                about: document.querySelector('.pv-about__summary-text')?.innerText,
                experience: Array.from(document.querySelectorAll('.pvs-list__item--line-separated')).map(item => ({
                    title: item.querySelector('.mr1')?.innerText,
                    company: item.querySelector('.t-14')?.innerText,
                    duration: item.querySelector('.t-black--light')?.innerText,
                })),
                education: Array.from(document.querySelectorAll('.pv-education-entity')).map(edu => ({
                    school: edu.querySelector('.pv-entity__school-name')?.innerText,
                    degree: edu.querySelector('.pv-entity__degree-name')?.innerText,
                    field: edu.querySelector('.pv-entity__fos')?.innerText,
                })),
                skills: Array.from(document.querySelectorAll('.pv-skill-category-entity')).map(skill => 
                    skill.querySelector('.pv-skill-category-entity__name')?.innerText
                ),
            };
        }
    """)
    
    return profile
```

---

## 6. 实现HR Talent Search Agent

### 6.1 设计HR Agent架构

```python
# src/magentic_ui/agents/talent_search/hr_talent_search_agent.py

from typing import Dict, List, Any
from autogen_agentchat.agents import BaseChatAgent
from autogen_core.models import ChatCompletionClient
from playwright.async_api import Page

class HRTalentSearchAgent(BaseChatAgent):
    """
    HR人才搜索专用Agent:
    - LinkedIn搜索
    - GitHub profile分析
    - Resume解析
    - 候选人匹配和评分
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        web_surfer: WebSurfer,  # 复用WebSurfer
        database_path: str = "./talent_db.sqlite",
    ):
        super().__init__(name, "HR Talent Search Specialist")
        self._model_client = model_client
        self._web_surfer = web_surfer
        self._db = TalentDatabase(database_path)
        
        # 定义HR特定工具
        self.tools = [
            TOOL_SEARCH_LINKEDIN,
            TOOL_ANALYZE_GITHUB,
            TOOL_PARSE_RESUME,
            TOOL_MATCH_JOB_REQUIREMENTS,
        ]
    
    async def search_talent(
        self,
        job_requirements: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        核心搜索流程:
        1. 解析职位需求
        2. 在多个平台搜索
        3. 提取候选人信息
        4. 评分和排序
        """
        # 步骤1: 使用LLM解析职位需求
        structured_req = await self._parse_job_requirements(job_requirements)
        
        # 步骤2: LinkedIn搜索
        linkedin_candidates = await self._search_linkedin(structured_req)
        
        # 步骤3: GitHub搜索 (针对技术岗位)
        if structured_req.get("is_technical"):
            github_candidates = await self._search_github(structured_req)
        
        # 步骤4: 合并和去重
        all_candidates = self._merge_candidates(
            linkedin_candidates, 
            github_candidates
        )
        
        # 步骤5: 使用LLM评分
        scored_candidates = await self._score_candidates(
            all_candidates, 
            structured_req
        )
        
        # 步骤6: 保存到数据库
        await self._db.save_candidates(scored_candidates)
        
        return scored_candidates
    
    async def _parse_job_requirements(
        self, 
        job_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM解析非结构化的职位描述
        """
        prompt = f"""
        Extract structured information from this job description:
        
        {job_requirements['description']}
        
        Extract:
        - Required skills (list)
        - Years of experience (int)
        - Education level (string)
        - Location (string)
        - Industry (string)
        - Is technical role (bool)
        
        Return as JSON.
        """
        
        result = await self._model_client.create([
            UserMessage(content=prompt)
        ])
        
        return json.loads(result.content)
    
    async def _search_linkedin(
        self, 
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用WebSurfer在LinkedIn搜索
        """
        # 构建搜索查询
        keywords = " ".join(requirements["required_skills"])
        location = requirements.get("location", "")
        
        # 让WebSurfer执行搜索
        search_message = TextMessage(
            content=f"Search LinkedIn for candidates with skills: {keywords}, location: {location}",
            source="hr_agent"
        )
        
        response = await self._web_surfer.on_messages([search_message])
        
        # 从响应中提取候选人列表
        candidates = self._extract_candidates_from_response(response)
        
        return candidates
    
    async def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        requirements: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        使用LLM评分候选人匹配度
        """
        scored = []
        
        for candidate in candidates:
            prompt = f"""
            Score this candidate against job requirements (0-100):
            
            Requirements:
            {json.dumps(requirements, indent=2)}
            
            Candidate:
            Name: {candidate['name']}
            Title: {candidate['title']}
            Experience: {candidate.get('experience', 'Unknown')}
            Skills: {candidate.get('skills', [])}
            Education: {candidate.get('education', 'Unknown')}
            
            Provide:
            1. Overall score (0-100)
            2. Skill match score (0-100)
            3. Experience match score (0-100)
            4. Brief explanation
            
            Return as JSON: {{"overall_score": 85, "skill_match": 90, "experience_match": 80, "explanation": "..."}}
            """
            
            result = await self._model_client.create([
                UserMessage(content=prompt)
            ])
            
            score_data = json.loads(result.content)
            candidate.update(score_data)
            scored.append(candidate)
        
        # 按分数排序
        scored.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return scored
```

### 6.2 Tool定义示例

```python
# src/magentic_ui/agents/talent_search/_tool_definitions.py

TOOL_SEARCH_LINKEDIN: ToolSchema = load_tool({
    "type": "function",
    "function": {
        "name": "search_linkedin",
        "description": "Search for candidates on LinkedIn",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Search keywords (skills, title)"
                },
                "location": {
                    "type": "string",
                    "description": "Location filter"
                },
                "years_experience": {
                    "type": "integer",
                    "description": "Minimum years of experience"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 20
                }
            },
            "required": ["keywords"]
        }
    },
    "metadata": {
        "requires_approval": "never"
    }
})

TOOL_ANALYZE_GITHUB: ToolSchema = load_tool({
    "type": "function",
    "function": {
        "name": "analyze_github_profile",
        "description": "Analyze a GitHub profile for technical skills",
        "parameters": {
            "type": "object",
            "properties": {
                "github_username": {
                    "type": "string",
                    "description": "GitHub username"
                },
                "analyze_repos": {
                    "type": "boolean",
                    "description": "Whether to analyze repositories",
                    "default": True
                }
            },
            "required": ["github_username"]
        }
    },
    "metadata": {
        "requires_approval": "never"
    }
})

TOOL_PARSE_RESUME: ToolSchema = load_tool({
    "type": "function",
    "function": {
        "name": "parse_resume",
        "description": "Parse resume PDF/DOCX and extract structured data",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to resume file"
                },
                "extract_skills": {
                    "type": "boolean",
                    "description": "Extract skills",
                    "default": True
                },
                "extract_experience": {
                    "type": "boolean",
                    "description": "Extract work experience",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
    },
    "metadata": {
        "requires_approval": "never"
    }
})
```

### 6.3 数据库设计

```python
# src/magentic_ui/agents/talent_search/database.py

from sqlalchemy import create_engine, Column, Integer, String, JSON, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class Candidate(Base):
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True)
    phone = Column(String)
    linkedin_url = Column(String, unique=True)
    github_url = Column(String)
    
    # 基本信息
    title = Column(String)
    location = Column(String)
    years_experience = Column(Integer)
    
    # JSON字段
    skills = Column(JSON)  # ["Python", "Machine Learning", ...]
    experience = Column(JSON)  # [{"company": "...", "title": "...", ...}, ...]
    education = Column(JSON)  # [{"school": "...", "degree": "...", ...}, ...]
    
    # 评分
    overall_score = Column(Float)
    skill_match_score = Column(Float)
    experience_match_score = Column(Float)
    
    # 元数据
    source = Column(String)  # "linkedin", "github", "resume"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)

class JobRequirement(Base):
    __tablename__ = "job_requirements"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String)
    required_skills = Column(JSON)
    preferred_skills = Column(JSON)
    min_experience = Column(Integer)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TalentDatabase:
    def __init__(self, db_path: str = "./talent_search.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    async def save_candidate(self, candidate_data: Dict[str, Any]) -> int:
        """保存候选人"""
        candidate = Candidate(**candidate_data)
        self.session.add(candidate)
        self.session.commit()
        return candidate.id
    
    async def search_candidates(
        self,
        skills: List[str] = None,
        min_score: float = 0.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """搜索候选人"""
        query = self.session.query(Candidate)
        
        if min_score:
            query = query.filter(Candidate.overall_score >= min_score)
        
        if skills:
            # SQLite JSON查询
            for skill in skills:
                query = query.filter(
                    Candidate.skills.contains(skill)
                )
        
        candidates = query.order_by(
            Candidate.overall_score.desc()
        ).limit(limit).all()
        
        return [self._candidate_to_dict(c) for c in candidates]
```

### 6.4 集成到task_team.py

```python
# 在task_team.py中添加HR Agent

async def get_task_team(...) -> GroupChat:
    # ... 现有代码 ...
    
    # 创建HR Talent Search Agent
    hr_agent = None
    if magentic_ui_config.enable_hr_agent:
        hr_agent = HRTalentSearchAgent(
            name="hr_talent_search",
            model_client=model_client_orch,
            web_surfer=web_surfer,
            database_path=str(paths.internal_run_dir / "talent_db.sqlite"),
        )
    
    # 添加到团队
    team_participants = [web_surfer, user_proxy, coder_agent, file_surfer]
    if hr_agent:
        team_participants.append(hr_agent)
    
    team = GroupChat(
        participants=team_participants,
        orchestrator_config=orchestrator_config,
        model_client=model_client_orch,
    )
    
    return team
```

---

## 7. 完整开发流程

### 7.1 环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/microsoft/magentic-ui.git
cd magentic-ui

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -e .

# 4. 配置API密钥
export OPENAI_API_KEY="your-key-here"

# 5. 测试运行
magentic-ui --port 8081
```

### 7.2 开发HR Agent步骤

#### 步骤1: 创建Agent目录结构

```bash
mkdir -p src/magentic_ui/agents/talent_search
touch src/magentic_ui/agents/talent_search/__init__.py
touch src/magentic_ui/agents/talent_search/_talent_search_agent.py
touch src/magentic_ui/agents/talent_search/_tool_definitions.py
touch src/magentic_ui/agents/talent_search/database.py
```

#### 步骤2: 实现基础Agent

```python
# _talent_search_agent.py

from autogen_agentchat.agents import BaseChatAgent
from autogen_core.models import ChatCompletionClient
from typing import AsyncGenerator, Sequence
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.base import Response

class TalentSearchAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
    ):
        super().__init__(
            name=name,
            description="HR Talent Search Specialist"
        )
        self._model_client = model_client
    
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token,
    ) -> Response:
        # 实现核心逻辑
        pass
```

#### 步骤3: 定义工具

```python
# _tool_definitions.py

from autogen_core.tools import ToolSchema

TOOL_SEARCH_TALENT: ToolSchema = {
    "type": "function",
    "function": {
        "name": "search_talent",
        "description": "Search for talent based on criteria",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["keywords"]
        }
    },
    "metadata": {
        "requires_approval": "never"
    }
}
```

#### 步骤4: 集成到主系统

```python
# 在 agents/__init__.py 中导出
from .talent_search import TalentSearchAgent

__all__ = [
    "WebSurfer",
    "CoderAgent",
    "TalentSearchAgent",  # 新增
]
```

#### 步骤5: 修改task_team.py

```python
# 在 get_task_team() 中添加
from .agents import TalentSearchAgent

# 创建实例
talent_agent = TalentSearchAgent(
    name="talent_search",
    model_client=model_client_orch,
)

# 添加到团队
team_participants.append(talent_agent)
```

#### 步骤6: 测试

```bash
# 运行测试
python -m pytest tests/test_talent_search_agent.py

# 启动系统
magentic-ui --port 8081

# 在浏览器访问并测试
# http://localhost:8081
```

### 7.3 调试技巧

```python
# 1. 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. 使用loguru
from loguru import logger
logger.add("talent_search_{time}.log", rotation="1 day")

# 3. 在Agent中添加调试输出
async def on_messages(self, messages, cancellation_token):
    logger.debug(f"Received messages: {messages}")
    
    # 处理逻辑
    result = await self._process()
    
    logger.debug(f"Returning result: {result}")
    return result

# 4. 使用VS Code调试
# launch.json:
{
    "type": "python",
    "request": "launch",
    "module": "magentic_ui.backend.cli",
    "args": ["--port", "8081", "--debug"]
}
```

### 7.4 常见问题解决

#### 问题1: Docker连接失败

```bash
# 检查Docker是否运行
docker ps

# 检查端口占用
lsof -i :37367
lsof -i :6080

# 手动启动容器测试
docker run -it --rm \
  -p 6080:6080 \
  -p 37367:37367 \
  ghcr.io/microsoft/magentic-ui-browser:latest
```

#### 问题2: LLM调用失败

```python
# 添加重试逻辑
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_llm(self, prompt):
    return await self._model_client.create([
        UserMessage(content=prompt)
    ])
```

#### 问题3: Playwright超时

```python
# 增加超时时间
page.set_default_timeout(60000)  # 60秒

# 使用显式等待
await page.wait_for_selector(".element", timeout=30000)
await page.wait_for_load_state("networkidle")
```

---

## 8. 最佳实践

### 8.1 Agent设计原则

1. **单一职责**: 每个Agent只做一件事
2. **可组合性**: Agent之间通过消息通信
3. **错误处理**: 优雅处理失败情况
4. **可观测性**: 添加详细日志

### 8.2 Tool设计原则

1. **明确的描述**: LLM需要理解工具用途
2. **合理的参数**: 不要太多或太少
3. **审批策略**: 敏感操作需要审批
4. **幂等性**: 多次调用结果一致

### 8.3 性能优化

```python
# 1. 使用异步并发
import asyncio

async def process_candidates(candidates):
    tasks = [
        analyze_candidate(c) 
        for c in candidates
    ]
    results = await asyncio.gather(*tasks)
    return results

# 2. 缓存LLM结果
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_skill_embedding(skill: str):
    # 缓存技能向量
    pass

# 3. 批量操作
async def score_candidates_batch(candidates, batch_size=10):
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        await process_batch(batch)
```

---

## 9. 参考资源

### 官方文档
- [Magentic-UI GitHub](https://github.com/microsoft/magentic-ui)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Playwright Python](https://playwright.dev/python/)

### 代码示例
- `samples/sample_web_surfer.py` - WebSurfer使用示例
- `samples/sample_coder.py` - Coder使用示例
- `prev_hr_agent/kdd_hr_system/` - 之前的HR系统实现

### 相关论文
- [Magentic-UI arXiv](https://arxiv.org/abs/2507.22358)

---

## 附录: 完整示例代码

### A. 简单的LinkedIn搜索Agent

```python
# examples/simple_linkedin_agent.py

import asyncio
from autogen_agentchat.agents import BaseChatAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from magentic_ui.agents import WebSurfer
from magentic_ui.tools.playwright import LocalPlaywrightBrowser

class SimpleLinkedInAgent(BaseChatAgent):
    def __init__(self, name: str, web_surfer: WebSurfer):
        super().__init__(name, "LinkedIn Search Agent")
        self.web_surfer = web_surfer
    
    async def search(self, keywords: str, location: str = ""):
        message = TextMessage(
            content=f"""
            Go to LinkedIn and search for:
            Keywords: {keywords}
            Location: {location}
            
            Extract the names, titles, and locations of top 10 results.
            """,
            source="user"
        )
        
        response = await self.web_surfer.on_messages([message])
        return response

async def main():
    # 创建浏览器
    browser = LocalPlaywrightBrowser(headless=False)
    
    # 创建WebSurfer
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    web_surfer = WebSurfer(
        name="web_surfer",
        model_client=model_client,
        browser=browser,
    )
    await web_surfer.lazy_init()
    
    # 创建LinkedIn Agent
    linkedin_agent = SimpleLinkedInAgent(
        name="linkedin_agent",
        web_surfer=web_surfer,
    )
    
    # 搜索
    results = await linkedin_agent.search(
        keywords="Python Machine Learning",
        location="San Francisco",
    )
    
    print(results)
    
    await web_surfer.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

**结语**: 本文档提供了Magentic-UI项目的全面学习指南，涵盖了从架构理解到Agent开发的完整流程。通过学习这些内容，你应该能够构建自己的HR Talent Search Agent。如有问题，请参考源代码和官方文档。祝开发顺利！
