---
name: remote-spec-to-markdown
description: Use when remote competition rule pages must be converted into local Markdown with cleaned content, traceable sources, and stable REQ references.
---

# Overview

将远端网页题面抓取为仓库内 Markdown,保留来源追溯与 REQ 编号,供 Agent 本地稳定检索。

# Inputs

- `urls`: 远端规则链接列表
- `output_dir`: 默认 `docs/problem_statement`

# Outputs

- `docs/problem_statement/spec.md`
- `docs/problem_statement/qa.md`
- `docs/problem_statement/sources.md`
- `docs/problem_statement/README.md`

# Core Workflow

1. 抓取网页正文并清理页面噪声
2. 合并规则并生成 REQ 编号
3. 记录来源链接与清洗边界
4. 写入 `docs/problem_statement/` 并回归检查

# Validation Checklist

- 文档产物齐全
- `spec.md` 含 `REQ-GEN/REQ-ANT/REQ-QA`
- `sources.md` 可追溯来源 URL
