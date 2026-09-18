# 业务流程

> RAGX 的两条端到端流程：文档摄入（异步、任务队列驱动）与查询应答（混合检索 → 带引用的生成）。

[English](workflow.md) | **简体中文**

## 1. 文档摄入流程

```
上传 ──▶ POST /v1/documents ──▶ 任务队列（Redis Streams）──▶ Worker 流水线
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   │
   ├─ 1. 解析        文件 → 结构化内容
   │                  （lite：text/markdown；full：DeepDoc / MinerU 处理 PDF、
   │                   扫描件、跨页表格等）
   ├─ 2. 处理        可选 VLM 增强（图片理解、图表/表格描述——「描述即向量」）
   ├─ 3. 分块        视觉感知分块；写入分块级元数据
   ├─ 4. 向量化      分块 → 向量（lite 用哈希 Embedder）
   ├─ 5. 入库        向量 → 向量库（SQLite-vec / ES / Qdrant / Milvus）
   ├─ 6. 建图        实体与关系抽取 → 图存储
   │                 （NetworkX / Neo4j；kg_enabled=false 时跳过）
   └─ 7. 收尾        原件入对象存储（本地 FS / MinIO），任务状态置为 done
```

关键特性：

- **全异步**：对外只暴露两个同步操作——`POST /v1/documents`（返回 `task_id`）
  与 `GET /v1/tasks/{task_id}`（轮询状态）。
- **成本闸门**：VLM 与建图步骤按知识库级 Feature Flag 控制（`vlm_enabled` /
  `kg_enabled`），昂贵步骤可以按知识库关闭。
- **增量重索引**：重新上传文档只更新其分块，不重建整个知识库；重复上传会被
  拒绝（错误码 `2004`）。
- **分块治理**：摄入完成后分块可查看、可编辑（`GET /v1/documents/{id}/chunks`、
  `PUT /v1/chunks/{id}`，乐观锁）——检索永远基于治理后的分块状态。

## 2. 查询应答流程

```
提问 ──▶ POST /v1/search（或 /v1/chat/completions）
                 │
                 ├─ 0. 网关             认证 → 限流 → 租户隔离
                 ├─ 1. 语义缓存         命中？→ 直接返回
                 ├─ 2. 查询路由         Fast / Standard / Agentic
                 │
                 ├─ 3. 混合召回         （并行）
                 │      ├─ 向量：       查询 → embed → 向量库 ANN
                 │      ├─ 稀疏：       BM25 / FTS（trigram 分词器，对 CJK 友好）
                 │      └─ 图谱：       实体链接 → 子图扩展（kg_enabled 时）
                 │
                 ├─ 4. 融合             合并 + 去重 + 打分 → top_k 分块
                 ├─ 5. 生成             LLM 路由按角色选模型
                 │                      （简单步骤用便宜模型，综合用重型模型）
                 │                      重试 → 降级 → 熔断
                 └─ 6. 响应             答案 + 引用（文档/分块溯源）
```

### 三种查询模式

| 模式 | 路径 | 典型场景 |
|---|---|---|
| **Fast** | 缓存 → 单次检索 → 短答案 | 自动补全、低延迟查询 |
| **Standard** | 混合召回 → 融合 → 单轮带引用生成 | 默认 API 模式 |
| **Agentic** | LangGraph 编排：规划 → 多步检索/工具调用 → 验证 | 复杂多跳问题；验证失败自动降级 Standard |

### 可靠性设计

- **语义缓存** —— 重复/近似重复的问题直接跳过检索与生成（`cache_enabled`）。
- **弹性 LLM 路由** —— 所有 LLM 调用经过同一个路由器，统一负责重试、provider
  降级与熔断；Token 预算按请求强制执行（错误码 `6003`）。
- **优雅降级** —— BM25 不可用时运行时跳过该路；缺图存储时跳过图谱召回；
  Agentic 流水线验证失败时降级 Standard，而不是直接报错。
- **全链路可观测** —— `trace_id` 在 API 入口生成，贯穿每一个 span、日志行与
  成本台账；OpenTelemetry span 与 Prometheus 指标（`/v1/metrics`）覆盖两条流程。

## 3. MCP 工具流程

RAGX 的能力同时以标准 MCP 工具暴露——`ragx_search`、`ragx_generate`、
`ragx_list_kbs`——支持 stdio（本地 Agent / CLI）与 SSE（远程 / 服务集成）。
每个 SSE 会话限定在调用方 API key 的知识库 ACL 内：越权的 `kb_id` 返回工具级
错误，与 REST API 相同的租户隔离。端点与配置见
[README MCP 章节](../README.zh-CN.md#mcpmodel-context-protocol)。
