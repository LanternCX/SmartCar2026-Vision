# 测试分层说明

当前仓库使用轻量分层测试来验证 Agent 化流程和文档契约.

- `tests/unit/`: 仓库治理基线与规则约束
- `tests/contract/`: 文档结构与引用契约
- `tests/hil/`: 板级联调记录与验收证据

常用命令:

```bash
python3 -m pytest tests/unit -q
python3 -m pytest tests/contract -q
python3 -m pytest tests/unit tests/contract -q
```
