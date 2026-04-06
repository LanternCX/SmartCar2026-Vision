"""main.py 纯视觉协议契约测试."""

import ast
import inspect
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "main.py"
PROTOCOL_PATH = ROOT / "docs" / "Protocol.md"
README_PATH = ROOT / "README.md"


def assert_text_has_no_legacy_patterns(
    text: str, patterns: list[str], message: str
) -> None:
    """用正则拦截单双引号或轻微写法变化后的旧边界残留."""
    for pattern in patterns:
        assert re.search(pattern, text) is None, f"{message}: {pattern}"


def load_function(name: str):
    """从 main.py 中提取单个函数,避免执行硬件初始化."""
    source = MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MAIN_PATH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {}
    exec(compile(module, filename=str(MAIN_PATH), mode="exec"), namespace)
    return namespace[name]


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


def test_main_formats_minimal_observation_frames() -> None:
    """main.py 必须按正式行为输出最小协议格式."""
    format_vision_frame = load_function("format_vision_frame")

    assert (
        format_vision_frame(seq=3, valid=1, err_x=-12, err_y=5) == "v=1,s=3,x=-12,y=5"
    )
    assert format_vision_frame(seq=4, valid=0, err_x=99, err_y=88) == "v=0,s=4"


def test_main_format_vision_frame_signature_is_trimmed_to_minimal_protocol() -> None:
    """格式化函数签名不应继续暴露已退出正式输出的空壳参数."""
    format_vision_frame = load_function("format_vision_frame")
    parameter_names = list(inspect.signature(format_vision_frame).parameters)
    assert parameter_names == ["seq", "valid", "err_x", "err_y"]


def test_protocol_docs_use_minimal_text_frames_only() -> None:
    """正式协议文档与 README 只保留最小文本协议作为正式口径."""
    required_tokens = [
        "v=1,s=<seq>,x=<x>,y=<y>",
        "v=0,s=<seq>",
        "`v=1` 表示当前帧存在有效目标",
        "`v=0` 表示当前帧无有效目标",
        "`x` 表示目标中心相对画面中心的横向像素差值",
        "`y` 表示目标底边相对当前期望抓取位置的纵向像素差值",
        "`v=1` 时必须同时携带 `x` 和 `y`",
        "`v=0` 时不发送 `x` 和 `y`",
        "`x > 0` 表示目标在画面中心右侧，`x < 0` 表示目标在画面中心左侧",
        "`y > 0` 表示目标底边超过期望抓取位置，`y < 0` 表示目标底边尚未到达期望抓取位置",
        "`y=0` 表示目标已到达当前设定的抓取距离",
        "无目标时，视觉端发送明确的无目标报文",
    ]
    forbidden_tokens = [
        "vision=1",
        "camera_id",
        "target=",
        "err_x",
        "err_y",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "left=<",
        "top=<",
        "right=<",
        "bottom=<",
        "left/top/right/bottom",
    ]

    for path in (PROTOCOL_PATH, README_PATH):
        text = path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in text, f"missing token in {path.name}: {token}"
        for token in forbidden_tokens:
            assert token not in text, f"legacy token found in {path.name}: {token}"
