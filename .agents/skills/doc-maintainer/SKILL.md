---
name: doc-maintainer
description: Use when creating or updating repository documentation to keep Chinese docs, rule traceability, and maintenance checklists consistent.
---

# Overview

统一仓库文档维护规则,确保文档可追溯、可检索、可持续更新。

# When to Use

- 修改 `README.md`、`AGENTS.md` 或 `docs/` 下文档
- 新增规则说明、流程说明、计划文档
- 同步远端规则更新到本地文档

# Core Rules

- 文档默认使用中文
- 内容优先准确与可复现,避免主观扩写
- 关键约束需给出来源路径或来源链接

# Problem Statement Maintenance

- 规则文档目录: `docs/problem_statement/`
- 最小产物:
  - `spec.md`
  - `qa.md`
  - `sources.md`
  - `README.md`
- 官方规则更新后,同步更新以上文件并记录差异

# Checklist

- 是否补齐来源追溯信息
- 是否保留 REQ 编号可引用能力
- 是否移除页面噪声（广告/推荐/交互文案）
- 是否更新使用说明与维护建议

# Deliverables

- 完整且可追溯的文档变更
- 与规则一致的引用编号与更新说明
