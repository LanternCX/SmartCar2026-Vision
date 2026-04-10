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
    """main.py 必须按辅车当前协议输出速度模式请求格式."""
    format_vision_frame = load_function("format_vision_frame")

    assert (
        format_vision_frame(seq=3, valid=1, err_x=-1.2, err_y=0.0)
        == "f=1,m=1,x=-1.2,y=0"
    )
    assert format_vision_frame(seq=4, valid=0, err_x=99, err_y=88) == "f=1,m=1,x=0,y=0"


def test_main_format_vision_frame_signature_is_trimmed_to_minimal_protocol() -> None:
    """格式化函数签名保持最小 follow 请求所需参数即可."""
    format_vision_frame = load_function("format_vision_frame")
    parameter_names = list(inspect.signature(format_vision_frame).parameters)
    assert parameter_names == ["seq", "valid", "err_x", "err_y"]


def test_main_exposes_follow_stage_and_control_builder() -> None:
    """main.py 必须显式包含 ART 端阶段判断与控制量生成入口."""
    builder = load_function("build_follow_command")
    parameter_names = list(inspect.signature(builder).parameters)
    assert parameter_names == ["valid", "err_x", "blob_area"]


def test_protocol_docs_use_minimal_text_frames_only() -> None:
    """正式协议文档与 README 必须直接对齐辅车速度模式请求口径."""
    shared_required_tokens = [
        "f=1,m=1,x=<x>,y=<y>",
        "速度模式入口",
        "`f=1,m=1`",
        "`x` 表示发给辅车的横向速度",
        "`y` 表示发给辅车的纵向速度",
        "OpenArt 直接承担跟随阶段判断和控制量生成",
        "`ALIGN_X`",
        "`ALIGN_Y`",
        "`CENTER_HOLD`",
        "`MARKER_MISSING`",
        "辅车已经支持的速度模式入口",
    ]
    shared_forbidden_tokens = [
        "vision=1",
        "camera_id",
        "target=",
        "err_x",
        "err_y",
        "f=1,s=<seq>,v=<0/1>,x=<x>,y=<y>",
        "OpenArt 不负责后续控制组织",
        "RT1021 负责决定“接下来怎么做”",
    ]

    protocol_only_required_tokens = [
        "不改变“持续发送速度模式请求文本”作为主线机制",
    ]
    protocol_only_forbidden_tokens = [
        "持续视觉观测上报",
        "持续发送最小视觉观测文本",
        "非观测动作",
    ]

    for path in (PROTOCOL_PATH, README_PATH):
        text = path.read_text(encoding="utf-8")
        for token in shared_required_tokens:
            assert token in text, f"missing token in {path.name}: {token}"
        for token in shared_forbidden_tokens:
            assert token not in text, f"legacy token found in {path.name}: {token}"

    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for token in protocol_only_required_tokens:
        assert token in protocol_text, f"missing token in {PROTOCOL_PATH.name}: {token}"
    for token in protocol_only_forbidden_tokens:
        assert token not in protocol_text, (
            f"legacy token found in {PROTOCOL_PATH.name}: {token}"
        )
