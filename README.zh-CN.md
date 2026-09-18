# RAGX

> 一个**插件化、生产级、全链路可观测**的开源 RAG 引擎。
> 多模态「描述即向量」接入 + 混合检索（向量 × BM25 × 图谱）+ 自适应 Agentic 查询路径，并为解析器 / 模型 / 存储提供统一的 SPI 插件层。

[English](README.md) | **简体中文**

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 为什么选 RAGX？

| 常见痛点 | RAGX 的解法 |
|---|---|
| 厂商锁定（绑死某个向量库 / 某个模型） | 7 个 SPI 接口——换任何后端都无需改代码 |
| LLM 成本失控 | 弹性路由器（重试 + 降级 + 熔断）+ 语义缓存 + 按角色拆分便宜/重型模型 |
| 多模态解析脆弱 | 视觉感知分块器 + VLM 处理器可插拔（DeepDoc / MinerU） |
| 缺乏可观测性 | OpenTelemetry 链路 + Prometheus 指标 + 成本台账 + RAG Trace 回放 |
| Prompt 迭代慢 | 版本化 Prompt（`*.v1.yaml`）+ 按知识库覆盖 + 评测回归门禁（`make eval`） |

## 五分钟快速上手（lite 精简版，零外部依赖）

```bash
pip install -e ".[lite]"
python -c "from ragx.api.app import create_app; print(create_app().title)"
# -> RAGX

# 或者直接起服务
ragx-serve
# -> 打开 http://127.0.0.1:8000/v1/health
```

lite 精简版使用：
- **解析器**：text/markdown
- **Embedder**：确定性哈希（零依赖）
- **向量库**：SQLite（`sqlite-vec`）
- **图存储**：NetworkX（内存）
- **LLM**：接入任意 OpenAI 兼容端点

## Full 完整版（ES + Neo4j + MinIO + PG + Redis + Grafana）

```bash
pip install -e ".[full]"
docker compose -f deploy/compose/full.yml up -d
python scripts/smoke_full.py
```

## 架构

```text
Client ──▶ API 网关（认证 · 限流 · 租户）
              │
              ├─▶ Query API   ──▶ Fast / Standard / Agentic
              │                   │
              │                   ├─▶ 语义缓存
              │                   ├─▶ 混合检索器（dense + BM25 + 图谱）
              │                   └─▶ 弹性 LLM 路由（重试 · 降级 · 熔断）
              │
              └─▶ Ingest API  ──▶ 任务队列（Redis Streams）
                                      │
                                      ├─▶ 解析器 ──▶ 处理器（VLM）──▶ 分块器
                                      │                                ├─▶ Embedder ─▶ 向量库
                                      │                                └─▶ KG 构建  ─▶ 图存储
                                      └─▶ 对象存储（本地 FS / MinIO）
```

五层架构、只允许向下依赖、七个 SPI 接口：

```text
api/                  L4 接入层
agentic/ retrieval/ ingestion/ chunking/ kg/   L3 领域服务
llm/ observability/                              L2 横切能力
spi/ plugins/                                     L1 契约 + 插件
core/                                             L0 地基（不依赖任何上层）
```

详见**[架构文档](docs/architecture.zh-CN.md)**（分层规则、SPI 插件层、错误模型）与**[业务流程文档](docs/workflow.zh-CN.md)**（摄入与查询端到端流程）。

## API

| 端点 | 用途 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 兼容对话（支持 SSE 流式） |
| `POST /v1/search` | 纯检索（适合作为 Agent 工具） |
| `POST /v1/documents` | 上传文档（返回 `task_id`） |
| `GET  /v1/tasks/{task_id}` | 轮询接入任务状态 |
| `GET  /v1/documents/{id}/chunks` | 列出分块（治理用） |
| `PUT  /v1/chunks/{id}` | 编辑分块（乐观锁） |
| `GET  /v1/health?deep=true` | 深度依赖健康检查 |
| `GET  /v1/metrics` | Prometheus 指标暴露 |
| `GET  /v1/mcp/sse` · `POST /v1/mcp/messages` | MCP over SSE（可选开启，见下文） |

所有响应共用统一错误信封 `{error: {code, message, trace_id}}`，错误码按段位划分——见[架构文档的错误模型](docs/architecture.zh-CN.md#错误模型)。

## MCP（Model Context Protocol）

把 RAGX 的检索 / 生成能力以标准 MCP 工具暴露——`ragx_search`、
`ragx_generate`、`ragx_list_kbs`（流程概览：[业务流程 §3](docs/workflow.zh-CN.md#3-mcp-工具流程)）。

**stdio** —— 用于本地 Agent / CLI，无需改动服务端：

```bash
ragx-mcp            # 或：python -m ragx.mcp
```

**SSE** —— 用于远程 / 服务集成；默认关闭（会挂载额外路由）：

```bash
RAGX_MCP.ENABLED=true RAGX_MCP.TRANSPORT=sse ragx-serve
# GET  /v1/mcp/sse        -> 连接时下发 `event: sessionId`
# POST /v1/mcp/messages?sessionId=<sid>
```

| 配置 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `mcp.enabled` | `RAGX_MCP.ENABLED` | `false` | `false` = 不挂载 `/v1/mcp/*` 路由 |
| `mcp.transport` | `RAGX_MCP.TRANSPORT` | `stdio` | 首选传输方式 |
| `mcp.sse_idle_timeout` | `RAGX_MCP.SSE_IDLE_TIMEOUT` | `300` | 秒；`<= 0` 表示关闭空闲自动断开 |

每个 SSE 会话的作用域限定在调用方 API key 的 `kb_acl` 内：越权的 `kb_id`
会返回工具级错误，`ragx_list_kbs` 也只列出有权访问的知识库——与 REST API
相同的租户隔离。

## 开发

```bash
# 安装全部依赖
pip install -e ".[lite,dev]"

# 跑测试（359 passed / 31 skipped；skip 的是 ES/Qdrant/Neo4j/Milvus 契约套件）
pytest -q tests/unit tests/contract tests/integration tests/e2e

# 覆盖率门禁（阈值与 omit 配置见 pyproject [tool.coverage.*]）
pytest -q --cov=ragx --cov-fail-under=80 tests/unit tests/contract tests/integration tests/e2e

# lint + 类型检查
ruff check ragx tests scripts
mypy ragx/core ragx/spi ragx/llm ragx/retrieval ragx/api

# 评测门禁——离线冒烟，或安装 ragx[eval] 后用 ragas
make eval-local
make eval            # 需要 ragx[eval] + 评审模型 + 已灌入的评测语料

# lite 技术栈冒烟测试（上传 → 查询 → 引用，端到端）
python scripts/smoke_lite.py
```

## 文档

- [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md) —— 分层架构、SPI 插件层、并发模型、错误模型（**[English](docs/architecture.md)**）
- [`docs/workflow.zh-CN.md`](docs/workflow.zh-CN.md) —— 摄入与查询应答端到端流程（**[English](docs/workflow.md)**）
- [`CHANGELOG.md`](CHANGELOG.md) —— 发布说明

## 版本

RAGX 遵循 [SemVer](https://semver.org/)。SPI 自 v1.0.0 起冻结；此前的 `0.x` 小版本之间可能破坏 SPI。

| 版本 | 状态 | 亮点 |
|---|---|---|
| `0.1.0` | 关键路径 | SPI + lite 技术栈 + Standard 查询 + 弹性路由 + OTel + `/v1/metrics` |
| `0.2.0` | 生产化 | Redis Streams + 语义缓存 + full compose（ES/Neo4j/MinIO/PG）+ 认证/审计 |
| `0.3.0` | 多模态 + 图谱 | DeepDoc/MinerU 解析器 + VLM 成本闸门 + KG 双层检索 + 14 个 Prompt |
| `0.4.0` | Agentic + 评测 | 三级查询路由 + LangGraph 编排 + RAGAS/DeepEval 评测框架 |
| `1.0.0` | 已发布 2026-09-03 | MCP（SSE+stdio，后端打通）· Helm · 多租户/审计（过安全评审）· 进程内 E2E · L4 评测门禁（`make eval`；ragas 夜间跑） |

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
