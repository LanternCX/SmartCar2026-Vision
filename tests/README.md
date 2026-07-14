# 测试分层说明

本仓库测试只覆盖 OpenART 视觉运行时行为、协议行为和回归场景。

- `tests/unit/`: 纯函数、阶段判断和速度短包生成等确定性行为。
- `tests/contract/`: 角色 `run.py` 对外协议行为和可导入边界。

不为文档结构、代理规则入口、治理文件存在性或流程文字编写硬约束测试。文档、注释和规则入口通过 review 检查。

常用命令：

```bash
PYTHONPATH=. uv run pytest tests/unit -q
PYTHONPATH=. uv run pytest tests/contract -q
PYTHONPATH=. uv run pytest tests/unit tests/contract -q
```

其中 `tests/contract/` 包含 LSP 契约测试，会调用 `npx pyright` 检查仓库类型状态。
