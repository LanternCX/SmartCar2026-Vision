"""远端规则文档机制契约测试."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "docs/problem_statement"


def test_problem_statement_documents_exist() -> None:
    """规则文档四件套必须齐全."""
    required = ["spec.md", "qa.md", "sources.md", "README.md"]
    missing = [name for name in required if not (DOC_ROOT / name).exists()]
    assert not missing, f"missing problem statement docs: {missing}"


def test_spec_contains_req_prefixes() -> None:
    """主规则文档必须包含 REQ 编号体系."""
    spec_path = DOC_ROOT / "spec.md"
    assert spec_path.exists(), "spec.md must exist"
    text = spec_path.read_text(encoding="utf-8")
    for token in ("REQ-GEN-", "REQ-ANT-", "REQ-QA-"):
        assert token in text, f"missing token in spec.md: {token}"


def test_sources_contains_three_reference_urls() -> None:
    """来源文档必须包含三条远端 URL."""
    sources_path = DOC_ROOT / "sources.md"
    assert sources_path.exists(), "sources.md must exist"
    text = sources_path.read_text(encoding="utf-8")
    expected_urls = [
        "https://zhuoqing.blog.csdn.net/article/details/154598625",
        "https://zhuoqing.blog.csdn.net/article/details/154691441",
        "https://zhuoqing.blog.csdn.net/article/details/157686623",
    ]
    for url in expected_urls:
        assert url in text, f"missing source url: {url}"


def test_problem_statement_readme_mentions_update_sync() -> None:
    """使用说明需强调规则更新后的同步维护."""
    readme_path = DOC_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    text = readme_path.read_text(encoding="utf-8")
    assert "官方规则更新" in text
    assert "同步更新" in text
