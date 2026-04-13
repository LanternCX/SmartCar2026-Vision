# Vision Agentization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 SmartCar2026-Vision 仓库升级为与 TransportCar 对齐的 Agent 化工程流程,同时保持单文件运行架构并排除 mpy-cli.

**Architecture:** 在不拆分 `main.py` 的前提下,新增项目级治理层（`AGENTS.md` + `.agents/skills/*`）,补齐 TDD 与 CI 最小闭环,并引入同款远端规则更新文档机制（`docs/problem_statement/*`）. 运行时代码仍以 `main.py` 为唯一主入口,测试优先覆盖治理产物与流程约束.

**Tech Stack:** Python 3 + pytest + GitHub Actions + Markdown 文档与项目本地 skills.

---

### Task 1: 建立 TDD 基线（RED）

**Files:**
- Create: `pytest.ini`
- Create: `tests/README.md`
- Create: `tests/unit/test_agentization_baseline.py`

**Step 1: Write the failing test**

在 `tests/unit/test_agentization_baseline.py` 中先断言以下文件存在（预期失败）：
- `AGENTS.md`
- `.agents/skills/code-standards/SKILL.md`
- `.agents/skills/git-workflow/SKILL.md`
- `.agents/skills/tdd-integration/SKILL.md`
- `.agents/skills/doc-maintainer/SKILL.md`
- `.agents/skills/remote-spec-to-markdown/SKILL.md`

并断言以下文件不存在（预期通过）：
- `.mpy-cli.toml`
- `.mpy-cli/`
- `.mpyignore`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: FAIL（缺少 Agent 化产物）.

**Step 3: Write minimal implementation**

仅补齐测试框架文件（`pytest.ini` 与 `tests/README.md`）,不实现技能文件.

**Step 4: Run test to verify status**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: 仍 FAIL（因为核心产物尚未创建）.

### Task 2: 落地 Agent 规则与技能（GREEN）

**Files:**
- Create: `AGENTS.md`
- Create: `.agents/skills/README.md`
- Create: `.agents/skills/code-standards/SKILL.md`
- Create: `.agents/skills/git-workflow/SKILL.md`
- Create: `.agents/skills/tdd-integration/SKILL.md`
- Create: `.agents/skills/doc-maintainer/SKILL.md`

**Step 1: Write the failing test**

在 `tests/unit/test_agentization_baseline.py` 增加内容校验（先失败）：
- `AGENTS.md` 包含 `TDD`、`Git Workflow`、`单文件架构`、`禁止 mpy-cli`
- `doc-maintainer` skill frontmatter 包含 `name` 与 `description`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: FAIL（内容校验不满足）.

**Step 3: Write minimal implementation**

创建以上文件并补齐最小可用规则：
- Superpowers 作流入口
- 项目 Git Flow + 提交规范
- TDD red-green-refactor 约束
- 单文件架构与中文文档约束
- 文档维护职责与更新流程

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: PASS.

### Task 3: 引入远端规则更新机制（GREEN）

**Files:**
- Create: `.agents/skills/remote-spec-to-markdown/SKILL.md`
- Create: `docs/problem_statement/spec.md`
- Create: `docs/problem_statement/qa.md`
- Create: `docs/problem_statement/sources.md`
- Create: `docs/problem_statement/README.md`
- Create: `tests/contract/test_problem_statement_contract.py`

**Step 1: Write the failing test**

在 `tests/contract/test_problem_statement_contract.py` 先断言（预期失败）：
- `spec.md` 含 `REQ-GEN-`、`REQ-ANT-`、`REQ-QA-`
- `sources.md` 含 3 个来源 URL
- `README.md` 提示规则更新后同步维护

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/contract/test_problem_statement_contract.py -q`
Expected: FAIL（文档尚不存在）.

**Step 3: Write minimal implementation**

将 TransportCar 的同款机制迁入,并沿用同一来源 URL 与 REQ 编号体系.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/contract/test_problem_statement_contract.py -q`
Expected: PASS.

### Task 4: 落地 TDD 文档与 CI 门禁（GREEN）

**Files:**
- Create: `docs/developer/tdd-workflow.md`
- Create: `.github/workflows/tdd.yml`
- Modify: `tests/unit/test_agentization_baseline.py`

**Step 1: Write the failing test**

在 `tests/unit/test_agentization_baseline.py` 增加断言（预期失败）：
- `.github/workflows/tdd.yml` 存在且包含 `tests/unit tests/contract`
- `docs/developer/tdd-workflow.md` 包含 `RED`、`GREEN`、`REFACTOR`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

创建 TDD 指南和 CI 工作流（unit + contract）.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: PASS.

### Task 5: 更新 README 并做全量回归（REFACTOR）

**Files:**
- Modify: `README.md`
- Modify: `tests/README.md`

**Step 1: Write the failing test**

在 `tests/unit/test_agentization_baseline.py` 增加 README 约束（预期失败）：
- README 包含 `AGENTS.md` 与 `.agents/skills` 入口
- README 声明保留单文件架构

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agentization_baseline.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

更新 README 开发工作流章节,补齐 Agent/TDD/规则维护入口.

**Step 4: Run tests to verify everything passes**

Run: `python3 -m pytest tests/unit tests/contract -q`
Expected: PASS.

### Task 6: 提交与核验（可选执行）

**Files:**
- Verify only

**Step 1: Verify git status**

Run: `git status --short --branch`
Expected: 仅包含本次 Agent 化相关变更.

**Step 2: Commit with project Git Workflow**

Run: `git add AGENTS.md .agents docs tests pytest.ini .github/workflows/tdd.yml README.md`
Run: `git commit -m "feat(agent): bootstrap agentized workflow and tdd guardrails"`

**Step 3: Post-commit verify**

Run: `git status --short --branch`
Expected: 工作区干净,分支领先 1 个提交.
