"""main.py 纯视觉协议契约测试."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "main.py"


def assert_text_has_no_legacy_patterns(
    text: str, patterns: list[str], message: str
) -> None:
    """用正则拦截单双引号或轻微写法变化后的旧边界残留."""
    for pattern in patterns:
        assert re.search(pattern, text) is None, f"{message}: {pattern}"


def test_main_does_not_use_legacy_state_machine_and_control_fields() -> None:
    """新实现不应残留旧状态机骨架与旧控制字段."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden = [
        '"d_angle=',
        '"rear=1',
        "class SMState",
        "ORBITING",
        "PUSHING",
        "RETURNING",
    ]
    for token in forbidden:
        assert token not in text, f"legacy token found: {token}"


def test_main_rejects_legacy_query_and_control_tokens() -> None:
    """main.py 不应继续保留旧轮询与旧控制语义关键字."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"\bsend_cmd_sync\s*\(",
        r"\bwait_until_idle\s*\(",
        r"['\"]\?lock['\"]",
        r"['\"]\?pos['\"]",
        r"['\"]STATE\?['\"]",
        r"['\"]PING['\"]",
    ]
    assert_text_has_no_legacy_patterns(
        text, forbidden_patterns, "legacy query/control token found"
    )


def test_main_rejects_legacy_alignment_and_offset_tokens() -> None:
    """main.py 不应继续保留旧对齐状态与旧偏移字段."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"\bALIGN_ANGLE\b",
        r"\bALIGN_DIST\b",
        r"\bALIGN_DX\b",
        r"['\"]dx\s*=",
        r"['\"]dy\s*=",
        r"['\"]d_angle\s*=",
        r"['\"]rear\s*=\s*1",
    ]
    assert_text_has_no_legacy_patterns(
        text, forbidden_patterns, "legacy state/control token found"
    )


def test_main_does_not_define_query_or_sync_helpers() -> None:
    """main.py 不应继续定义同步等待辅助函数."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"def\s+wait_until_idle\s*\(",
        r"def\s+query_\w+\s*\(",
        r"def\s+send_cmd_sync\s*\(",
    ]
    assert_text_has_no_legacy_patterns(text, forbidden_patterns, "legacy helper found")


def test_main_formats_bbox_observation_frames() -> None:
    """main.py 必须提供完整框视觉帧格式化函数."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    assert "def format_vision_frame" in text
    assert 'return "x=%s,y=%s"' not in text
    assert 'return "x=%d,y=%d"' not in text
    assert (
        'return "left=%s,top=%s,right=%s,bottom=%s"' in text
        or 'return "left=%d,top=%d,right=%d,bottom=%d"' in text
    )
