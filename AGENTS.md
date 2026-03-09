# AGENTS.md

本文件是本仓库给代码代理（agent）使用的统一执行规范。
若与其他说明冲突,优先遵循用户在当前会话中的明确指令。

## 1. 项目快照

- 平台: OpenART + MicroPython.
- 主要语言: Python.
- 运行时代码: `main.py`.
- 文档目录: `docs/`.
- 测试目录: `tests/unit`、`tests/contract`、`tests/hil`.

## 2. 单文件架构约束（强制）

- 本项目当前采用 **单文件架构**: 运行时主逻辑集中在 `main.py`.
- 默认不拆分 `src/` 多模块,除非用户明确要求进行结构重构.
- 允许新增测试、文档、技能文件,但不得改变 `main.py` 作为唯一主入口的事实.

## 3. 环境准备

本地建议 Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install pytest
```

## 4. 检查与测试命令

运行 unit:

```bash
python3 -m pytest tests/unit -q
```

运行 contract:

```bash
python3 -m pytest tests/contract -q
```

运行全量主机侧测试:

```bash
python3 -m pytest tests/unit tests/contract -q
```

## 5. TDD 工作流（强制）

仓库开发必须遵循 `RED -> GREEN -> REFACTOR`:

1. RED: 先写失败测试.
2. 校验失败原因与预期一致.
3. GREEN: 写最小实现使测试通过.
4. REFACTOR: 在全绿前提下重构.

禁止先写生产代码后补测试。

## 6. 代码规范

- 注释与文档字符串默认使用中文.
- 导入顺序: 标准库 -> 第三方/平台库 -> 本地.
- 优先显式类型标注,避免滥用 `Any`.
- 捕获具体异常,禁止静默吞错.
- 避免无关格式化和无关重构.

## 7. Git Workflow（项目规范）

- 本仓库使用项目内 **Git Workflow**: `main` + `dev` + 临时分支模型.
- 临时分支命名:
  - `feature/<desc>`
  - `fix/<desc>`
  - `hotfix/<desc>`
  - `release/<version>`
- 提交信息格式:

```text
<type>(<scope>): <subject>
```

常用 `type`: `feat` `fix` `refactor` `docs` `test` `chore`.

## 8. 文档维护与规则更新

- 文档默认使用中文.
- 远端规则更新机制位于 `docs/problem_statement/`.
- 每次官方规则更新后,同步更新 `spec.md`、`qa.md`、`sources.md`.

## 9. 明确禁止项

- **禁止 mpy-cli**: 不引入 `.mpy-cli.toml`、`.mpy-cli/`、`.mpyignore`.
- 不得用 mpy-cli 作为本仓库部署流程依赖.

## 10. 项目技能索引

- `code-standards`: 代码规范与单文件架构边界.
- `git-workflow`: 分支与提交规范.
- `tdd-integration`: 项目 TDD 执行映射.
- `doc-maintainer`: 文档维护与规则更新守卫.
- `remote-spec-to-markdown`: 远端规则抓取与文档落库流程.
