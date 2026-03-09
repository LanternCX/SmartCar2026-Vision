---
name: tdd-integration
description: Use when adding features or fixes to enforce RED-GREEN-REFACTOR in this repository's unit, contract, and HIL layers.
---

# Overview

将 superpowers TDD 原则映射到当前仓库测试分层,确保每次改动都遵循先测后改。

# Required Background

- **REQUIRED SUB-SKILL:** superpowers:test-driven-development

# Layer Mapping

- `tests/unit/`: 仓库规则、主机侧确定性逻辑测试
- `tests/contract/`: 文档与流程契约测试
- `tests/hil/`: 板级联调记录与验收证据

# Execution Rules

1. RED: 先写失败测试,并确认失败原因正确
2. GREEN: 写最小实现使测试通过
3. REFACTOR: 在全绿前提下优化
4. 完成后至少回归 `tests/unit tests/contract`

# Quick Commands

```bash
python3 -m pytest tests/unit -q
python3 -m pytest tests/contract -q
python3 -m pytest tests/unit tests/contract -q
```

# Guardrails

- 禁止先写代码后补测试
- 禁止只做口头验证而无测试证据

# Deliverables

- 失败测试与通过测试证据
- 可复现测试命令
