# Eval Flow: 多阶段任务评估框架

## 概念及原理

### benchmark: 一个不断填充的字典，用来承载：

- 评测集初始配置与数据
- 每个评测 case 的预测（中间产物）
- 每个 case 的结果与统计（最终产物）

-启动后根据 `data` 里已经“ready”的 trace 自动派发各个 flow，把资源的输出写回对应 trace。

### trace: 一个地址，用来定位 benchmark(data) 字典中的某个数据

trace 仍然是 dot-separated 的路径。

保留字（仍保持一致）：

- `~0, ~1, ..., ~9`：通配符段，用于 flow 声明支持“一条 flow 覆盖多条具体 trace”

通配符匹配规则：不含保留字的 trace和包含保留字的 trace（通过通配符完成展开）

### resource: 资源（LLM、数据库、数据加载器等）

### flow: 处理数据的流（use/by/obtain/reuse）

flow 定义是：

- `use`: 使用何种 resource
- `by`: 输入 trace（可能多个，因此是列表）
- `obtain`: 输出写入 trace（可能多个，因此是列表）
- `reuse`: 使用 by trace 后是否继续使用该 trace（默认否）

---

## 工作原理：多次启动 + ready traces 扫描 + 异步派发

调度逻辑可以概括为：

1. 每次启动/循环时，检查 benchmark 字典中存在的所有可能 trace
2. 若存在一种 flow，使得 trace 满足 flow 的 `by_trace` 匹配规则
3. 且该 flow 未处理过该 trace（或 flow.reuse == True）
4. 则异步启动该 flow，获取结果并根据 flow.obtain 写入 benchmark 字典
---

## 使用

```bash
.venv/bin/python -m memsense_eval configs/full_pipeline.yaml
```

常用配置变体：

- `configs/qa_judge.yaml`：跳过 ingest，仅跑 QA -> judge -> summary
- `configs/judge_only.yaml`：只跑 judge + summary（从 JSONL 读取已有 QA 结果）

---



