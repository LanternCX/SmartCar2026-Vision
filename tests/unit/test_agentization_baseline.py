"""Agent 化基础约束测试."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_required_agentization_files_exist() -> None:
    """必须存在核心 Agent 化治理产物."""
    required = [
        "AGENTS.md",
        ".agents/skills/git-workflow/SKILL.md",
        ".agents/skills/tdd-integration/SKILL.md",
        ".agents/skills/doc-maintainer/SKILL.md",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing, f"missing required files: {missing}"


def test_mpy_cli_artifacts_are_not_introduced() -> None:
    """仓库禁止引入 mpy-cli 相关文件."""
    forbidden = [".mpy-cli.toml", ".mpy-cli", ".mpyignore"]
    present = [path for path in forbidden if (ROOT / path).exists()]
    assert not present, f"forbidden artifacts found: {present}"


def test_agents_policy_keywords_present() -> None:
    """AGENTS 需包含关键流程约束词."""
    agents_path = ROOT / "AGENTS.md"
    assert agents_path.exists(), "AGENTS.md must exist"
    text = agents_path.read_text(encoding="utf-8")
    for token in ("TDD", "main.py", "单文件架构", "mpy-cli"):
        assert token in text, f"missing keyword in AGENTS.md: {token}"


def test_doc_maintainer_skill_frontmatter() -> None:
    """doc-maintainer 技能需包含最小 frontmatter."""
    skill_path = ROOT / ".agents/skills/doc-maintainer/SKILL.md"
    assert skill_path.exists(), "doc-maintainer skill must exist"
    text = skill_path.read_text(encoding="utf-8")
    assert "name: doc-maintainer" in text
    assert "维护" in text


def test_tdd_workflow_doc_exists_and_mentions_cycle() -> None:
    """TDD 指南必须存在并包含 RED/GREEN/REFACTOR."""
    doc_path = ROOT / "docs/developer/tdd-workflow.md"
    assert doc_path.exists(), "docs/developer/tdd-workflow.md must exist"
    text = doc_path.read_text(encoding="utf-8")
    for token in ("RED", "REFACTOR", "失败测试"):
        assert token in text, f"missing token in tdd-workflow.md: {token}"


def test_ci_workflow_exists_and_runs_unit_contract() -> None:
    """CI 必须包含 unit+contract 的测试门禁."""
    workflow_path = ROOT / ".github/workflows/tdd.yml"
    assert workflow_path.exists(), ".github/workflows/tdd.yml must exist"
    text = workflow_path.read_text(encoding="utf-8")
    assert "tests/unit" in text
    assert "tests/contract" in text


def test_readme_mentions_agent_entry_and_single_file_architecture() -> None:
    """README 需声明 Agent 入口与单文件架构约束."""
    readme_path = ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    text = readme_path.read_text(encoding="utf-8")
    assert "AGENTS.md" in text
    assert "单文件架构" in text
