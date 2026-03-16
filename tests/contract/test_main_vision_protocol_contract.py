"""main.py 纯视觉协议契约测试."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "main.py"


def test_main_does_not_use_legacy_control_path() -> None:
    """新实现不应残留旧的同步控制与状态机链路."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden = [
        "send_cmd_sync(",
        "?lock",
        "?pos",
        '"dx=',
        '"dy=',
        '"d_angle=',
        '"rear=1',
        "class SMState",
        "ALIGN_ANGLE",
        "ALIGN_DIST",
        "ALIGN_DX",
        "ORBITING",
        "PUSHING",
        "RETURNING",
    ]
    for token in forbidden:
        assert token not in text, f"legacy token found: {token}"


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


def test_main_supports_query_response_with_physical_camera_id_contract() -> None:
    """main.py 必须具备按物理相机查询并返回多检测帧的协议支撑函数."""
    text = MAIN_PATH.read_text(encoding="utf-8")
    required_tokens = [
        "def normalize_camera_id",
        "def is_query_for_camera",
        "def format_detection_line",
        "def format_frame_end_line",
        "def build_frame_response_lines",
        "def build_query_response",
        "camera_id=%s,frame_id=%s,category=%s,%s",
        "camera_id=%s,frame_id=%s,frame_end=1",
    ]
    for token in required_tokens:
        assert token in text, f"missing query-response token: {token}"
