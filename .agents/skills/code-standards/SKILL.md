---
name: code-standards
description: Use when writing or reviewing SmartCar2026-Vision Python code to keep single-file architecture, coding quality, and MicroPython constraints consistent.
---

# Overview

统一 SmartCar2026-Vision 代码规范与架构边界,保证在 OpenART + MicroPython 约束下可维护、可验证。

# When to Use

- 修改 `main.py` 运行时代码
- 新增或修改测试、文档、项目技能文件
- 评审 PR 时检查风格、边界与异常处理

# Core Rules

- 目标平台: MicroPython（OpenART）
- 注释与文档字符串使用中文
- 单行注释行尾不使用句号
- 所有注释统一使用半角标点,中文注释也不例外
- 导入顺序: 标准库 -> 第三方/平台库 -> 本地
- 优先显式类型标注,避免 `Any`
- 捕获具体异常,禁止静默失败
- 避免无关重构与无关格式化

# Single-File Architecture Rules

- `main.py` 是唯一运行时主入口
- 默认不新增 `src/` 多模块运行时代码
- 新功能优先在 `main.py` 内按职责分区组织
- 允许新增 `tests/`、`docs/`、`.agents/skills/` 辅助产物
- 仅当用户明确要求架构重构时,才允许拆分运行时代码

# Review Checklist

- 是否保持 `main.py` 单文件主入口
- 是否满足中文注释,半角标点与异常处理要求
- 是否避免在单行注释行尾使用句号
- 是否存在无关改动
- 是否补充对应测试与验证命令

# Deliverables

- 符合规范的代码/文档改动
- 可复现的验证命令与结果
