---
name: git-workflow
description: Use when creating branches, committing, or merging changes under this repository's Git Workflow conventions.
---

# Overview

定义本仓库 Git Workflow: `main` + `dev` + 临时分支模型,并统一提交格式。

# When to Use

- 创建功能分支/修复分支/发布分支
- 组织提交历史
- 合并前检查分支与提交规范

# Branch Strategy

- 长期分支: `main`、`dev`
- 临时分支:
  - `feature/<desc>`
  - `fix/<desc>`
  - `hotfix/<desc>`
  - `release/<version>`

# Commit Convention

- 格式: `<type>(<scope>): <subject>`
- 常用 `type`: `feat` `fix` `refactor` `docs` `test` `chore`
- `scope` 建议使用 `agent` `main` `docs` `tests` 等目录/模块域

# Checklist

- 提交前运行必要测试
- 单个提交只包含单一主题
- 不提交敏感信息与无关文件

# Deliverables

- 符合分支策略的开发轨迹
- 符合提交规范的 commit history
