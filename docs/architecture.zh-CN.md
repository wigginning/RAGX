# 架构

> RAGX 的组织方式：五层架构 + 严格的单向依赖规则 + 统一 SPI 插件层 + 横切的 LLM / 可观测能力。

[English](architecture.md) | **简体中文**

## 分层架构

```text
┌─────────────────────────────────────────────┐
│ api/          FastAPI 路由、中间件、MCP       │  第 4 层：接入层
├─────────────────────────────────────────────┤
│ agentic/ retrieval/ ingestion/ chunking/ kg/ │  第 3 层：领域服务层
├─────────────────────────────────────────────┤
│ llm/ observability/                          │  第 2 层：横切能力层
├─────────────────────────────────────────────┤
│ spi/  plugins/                               │  第 1 层：接口与插件层
├─────────────────────────────────────────────┤
│ core/         领域模型、异常、配置             │  第 0 层：地基（无上游依赖）
└─────────────────────────────────────────────┘
```

**依赖规则**（通过 import-linter 强制执行）：

1. 只允许向下依赖：`api → 领域服务 → 横切能力 → spi → core`。禁止反向依赖，禁止领域服务之间直接互调。
2. 领域服务之间协作只允许两种方式：
   - 通过 `core/` 中的领域模型传递数据（无行为依赖）；
   - 通过 `spi/` 接口调用（如 retrieval 调用 `GraphStore`，而不是 import `kg/` 模块内部实现）。
3. `plugins/` 实现 `spi/` 接口，只允许依赖 `spi + core + 第三方 SDK`，禁止依赖领域服务层。
4. `observability/` 与 `llm/` 为横切层：第 3、4 层可以依赖它们；它们自身只依赖 `spi + core`。
5. 第三方框架隔离：LangGraph 只允许出现在 `agentic/`；FastAPI 只允许出现在 `api/`；Redis/Celery 客户端只允许出现在 `ingestion/` 与 `plugins/`。

## 端到端请求拓扑

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

## SPI 插件层

七个接口构成扩展面。实现对应接口并注册插件即可替换任何后端，无需修改应用代码：

| 接口 | 职责 | 内置实现 |
|---|---|---|
| Parser | 文件 → 结构化内容 | text/markdown（lite）、DeepDoc / MinerU（重型） |
| Processor | 用 VLM 增强内容 | 可选，受 `vlm_enabled` 控制 |
| Chunker | 内容 → 分块 | 视觉感知分块器 |
| Embedder | 分块 → 向量 | 确定性哈希（lite，零依赖）；sentence-transformers（`ragx[embed-st]`） |
| VectorStore | 向量存储与 ANN 检索 | SQLite `sqlite-vec`（lite）；Elasticsearch / Qdrant / Milvus（full） |
| GraphStore | 知识图谱存储 | NetworkX（内存，lite）；Neo4j（full） |
| LLM Provider | 对话 / 生成 | 任意 OpenAI 兼容端点 |

## 并发模型

| 边界 | 约定 |
|---|---|
| 摄入管线 | 全异步（任务队列驱动），对外只暴露「提交任务 + 查询状态」两个同步 API |
| 查询管线 | 异步处理函数 + 同步请求语义；SSE 流式输出；多路召回并行用 `asyncio.gather` |
| SPI 接口 | 所有方法一律 `async def`；同步 SDK 在插件内部用 `asyncio.to_thread` 包装 |
| LLM 调用 | 一律经过 `llm.router`，业务代码禁止直接实例化 provider 客户端 |

## 错误模型

所有错误共用统一信封 `{error: {code, message, trace_id}}`。错误码按段位划分；HTTP 状态码只表达大类（400/401/403/404/409/429/500/503）。

| 段位 | 归属 | 示例 |
|---|---|---|
| 1xxx | 接入层 | 1001 参数校验失败、1002 未认证、1003 越权、1004 限流 |
| 2xxx | 摄入 | 2001 解析失败、2002 不支持格式、2003 任务不存在、2004 重复文档 |
| 3xxx | 分块治理 | 3001 Chunk 不存在、3002 编辑冲突 |
| 4xxx | 检索 | 4001 知识库不存在、4002 召回为空、4003 过滤表达式非法 |
| 5xxx | 图 | 5001 建图失败、5002 图存储不可用、5003 提取 Schema 校验失败 |
| 6xxx | LLM | 6001 全部 provider 失败、6002 熔断中、6003 Token Budget 超限、6004 缓存后端不可用 |
| 7xxx | Agentic | 7001 规划失败、7002 任务执行全败、7003 验证未通过（降级 Standard） |
| 9xxx | 基础设施 | 9001 存储不可用、9002 队列不可用、9003 配置错误 |

## 全局约定

- **ID**：带类型前缀（`doc_` / `chk_` / `ent_` / `rel_` / `task_` / `kb_`）+ ULID（可排序、URL 安全）。
- **时间**：一律 UTC，ISO-8601 序列化。
- **trace_id**：API 入口生成，贯穿摄入与查询全链；日志、span、成本台账、RAG Trace 均携带。
- **配置**：三级覆盖——全局设置 ← 知识库级配置 ← 请求级 override。
- **Feature Flag**：`kg_enabled` / `vlm_enabled` / `agentic_enabled` / `cache_enabled` / `av_enabled`，粒度到知识库。
- **技术栈**：Python ≥ 3.11、Pydantic v2、类型标注全覆盖、核心模块 `mypy --strict`。
