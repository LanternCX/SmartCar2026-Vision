# TDD Workflow（项目集成）

本文档定义 superpowers TDD 在本仓库的执行方式。

## 核心规则

无失败测试,不允许修改生产代码。

严格执行:

1. RED: 先写失败测试
2. GREEN: 写最小实现使测试通过
3. REFACTOR: 在全绿前提下优化

## 本仓库测试分层

- `tests/unit/`: 仓库治理与主机侧确定性测试
- `tests/contract/`: 文档与流程契约测试
- `tests/hil/`: 板级联调记录与验收证据

## 常用命令

```bash
python3 -m pytest tests/unit -q
python3 -m pytest tests/contract -q
python3 -m pytest tests/unit tests/contract -q
```

## 合并前检查

- 目标测试层全部通过
- 关键行为变更有失败->通过证据
- 若涉及硬件联调,补充 `tests/hil/` 记录
