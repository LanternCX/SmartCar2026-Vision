"""纯视觉上报重构的主机侧单元测试."""

import ast
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "main.py"


def load_functions(*names: str):
    """从 main.py 中提取指定纯函数,避免执行硬件初始化."""
    source = MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MAIN_PATH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"time": time}
    exec(compile(module, filename=str(MAIN_PATH), mode="exec"), namespace)
    return [namespace[name] for name in names]


def test_format_vision_frame_outputs_bbox() -> None:
    """视觉帧必须按完整框输出四个边界键."""
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(100, 20, 140, 90)
    assert frame == "left=100,top=20,right=140,bottom=90"


def test_query_frame_returns_multiple_detections_with_physical_camera_id() -> None:
    """被点名相机必须返回同一帧中的多条检测与显式 frame_end."""
    (
        format_vision_frame,
        format_detection_line,
        format_frame_end_line,
        build_frame_response_lines,
    ) = load_functions(
        "format_vision_frame",
        "format_detection_line",
        "format_frame_end_line",
        "build_frame_response_lines",
    )

    format_detection_line.__globals__["format_vision_frame"] = format_vision_frame
    build_frame_response_lines.__globals__["format_detection_line"] = (
        format_detection_line
    )
    build_frame_response_lines.__globals__["format_frame_end_line"] = (
        format_frame_end_line
    )

    lines = build_frame_response_lines(
        "cam_a",
        12,
        [
            {
                "category": "cargo",
                "left": 100,
                "top": 20,
                "right": 140,
                "bottom": 90,
            },
            {
                "category": "follower",
                "left": 150,
                "top": 25,
                "right": 190,
                "bottom": 95,
            },
        ],
    )

    assert lines == [
        "camera_id=cam_a,frame_id=12,category=cargo,left=100,top=20,right=140,bottom=90",
        "camera_id=cam_a,frame_id=12,category=follower,left=150,top=25,right=190,bottom=95",
        "camera_id=cam_a,frame_id=12,frame_end=1",
    ]


def test_unaddressed_camera_stays_silent_even_when_category_overlaps() -> None:
    """类别重叠时, 未被点名物理相机也必须保持静默."""
    (
        normalize_camera_id,
        is_query_for_camera,
        format_vision_frame,
        format_detection_line,
        format_frame_end_line,
        build_frame_response_lines,
        build_query_response,
    ) = load_functions(
        "normalize_camera_id",
        "is_query_for_camera",
        "format_vision_frame",
        "format_detection_line",
        "format_frame_end_line",
        "build_frame_response_lines",
        "build_query_response",
    )

    is_query_for_camera.__globals__["normalize_camera_id"] = normalize_camera_id
    format_detection_line.__globals__["format_vision_frame"] = format_vision_frame
    build_frame_response_lines.__globals__["format_detection_line"] = (
        format_detection_line
    )
    build_frame_response_lines.__globals__["format_frame_end_line"] = (
        format_frame_end_line
    )
    build_query_response.__globals__["is_query_for_camera"] = is_query_for_camera
    build_query_response.__globals__["build_frame_response_lines"] = (
        build_frame_response_lines
    )

    detections = [
        {
            "category": "cargo",
            "left": 120,
            "top": 18,
            "right": 170,
            "bottom": 110,
        }
    ]

    assert build_query_response("?frame=cam_a", "cam_b", 33, detections) == []
    assert build_query_response("?frame=cam_a", "cam_a", 33, detections) == [
        "camera_id=cam_a,frame_id=33,category=cargo,left=120,top=18,right=170,bottom=110",
        "camera_id=cam_a,frame_id=33,frame_end=1",
    ]


def test_blob_rect_to_bbox_converts_width_height_to_edges() -> None:
    """blob.rect() 返回的 x,y,w,h 必须转换成 left,top,right,bottom."""
    (blob_rect_to_bbox,) = load_functions("blob_rect_to_bbox")
    bbox = blob_rect_to_bbox((100, 20, 40, 70))
    assert bbox == (100, 20, 140, 90)


def test_normalize_bbox_for_protocol_flips_vertical_axis_to_ground_down() -> None:
    """协议坐标必须统一成地板在下方的正向视图."""
    (normalize_bbox_for_protocol,) = load_functions("normalize_bbox_for_protocol")
    bbox = normalize_bbox_for_protocol(100, 20, 140, 90, img_height=240)
    assert bbox == (100, 150, 140, 220)


def test_build_blob_candidates_uses_normalized_bottom() -> None:
    """候选排序使用的 bottom 必须来自归一化后的协议坐标."""
    blob_rect_to_bbox, normalize_bbox_for_protocol, build_blob_candidates = (
        load_functions(
            "blob_rect_to_bbox", "normalize_bbox_for_protocol", "build_blob_candidates"
        )
    )

    class FakeBlob:
        def cx(self):
            return 160

        def cy(self):
            return 40

        def rect(self):
            return (100, 20, 40, 70)

    class FakeImg:
        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def height(self):
            return 240

    build_blob_candidates.__globals__["TASKS"] = (("Red", (0, 0, 0, 0, 0, 0)),)
    build_blob_candidates.__globals__["blob_rect_to_bbox"] = blob_rect_to_bbox
    build_blob_candidates.__globals__["normalize_bbox_for_protocol"] = (
        normalize_bbox_for_protocol
    )

    candidates = build_blob_candidates(FakeImg())

    assert candidates[0][0] == "Red"
    assert candidates[0][1] == 160
    assert candidates[0][2] == 40
    assert candidates[0][3] == 220


def test_protocol_frame_uses_normalized_bbox_coordinates() -> None:
    """最终串口帧必须发送归一化后的协议坐标."""
    blob_rect_to_bbox, normalize_bbox_for_protocol, format_vision_frame = (
        load_functions(
            "blob_rect_to_bbox", "normalize_bbox_for_protocol", "format_vision_frame"
        )
    )

    left, top, right, bottom = blob_rect_to_bbox((100, 20, 40, 70))
    left, top, right, bottom = normalize_bbox_for_protocol(
        left, top, right, bottom, img_height=240
    )

    frame = format_vision_frame(left, top, right, bottom)

    assert frame == "left=100,top=150,right=140,bottom=220"


def test_choose_best_candidate_prefers_center_bottom() -> None:
    """目标选择优先靠近中线且靠近底边."""
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [
        ("Red", 40, 120, 200),
        ("Green", 158, 180, 230),
        ("Red", 220, 160, 220),
    ]
    best = choose_best_candidate(candidates, cx_screen=160, img_height=240)
    assert best == ("Green", 158, 180, 230)


def test_choose_best_candidate_prefers_bbox_bottom_over_centroid_y() -> None:
    """上下颠倒安装时,翻转后的正向图像应以 bbox bottom 靠近地板为准."""
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [
        ("Red", 160, 220, 180),
        ("Green", 160, 120, 235),
    ]
    best = choose_best_candidate(candidates, cx_screen=160, img_height=240)
    assert best == ("Green", 160, 120, 235)


def test_should_send_throttle() -> None:
    """发送节流应避免过密上报."""
    (should_send,) = load_functions("should_send")
    assert should_send(now_ms=100, last_send_ms=None, interval_ms=30) is True
    assert should_send(now_ms=120, last_send_ms=100, interval_ms=30) is False
    assert should_send(now_ms=131, last_send_ms=100, interval_ms=30) is True


def test_write_line_appends_crlf() -> None:
    """串口发送必须按协议追加 CRLF 结尾."""
    sleep_short, write_line = load_functions("sleep_short", "write_line")

    class FakeUart:
        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload)

    namespace_uart = FakeUart()
    globals_dict = write_line.__globals__
    globals_dict["sleep_short"] = sleep_short
    write_line(namespace_uart, "left=100,top=20,right=140,bottom=90")
    assert namespace_uart.writes == ["left=100,top=20,right=140,bottom=90\r\n"]


def test_write_line_retries_until_full_line_is_sent() -> None:
    """串口部分写出时, 仍必须把整行协议完整发完."""
    sleep_short, write_line = load_functions("sleep_short", "write_line")

    class PartialUart:
        def __init__(self):
            self.writes = []
            self._responses = [10, 8, 999]

        def write(self, payload):
            self.writes.append(payload)
            return self._responses.pop(0)

    uart = PartialUart()
    globals_dict = write_line.__globals__
    globals_dict["sleep_short"] = sleep_short

    write_line(uart, "camera_id=cam_a,frame_id=1,frame_end=1")

    assert uart.writes == [
        "camera_id=cam_a,frame_id=1,frame_end=1\r\n",
        "cam_a,frame_id=1,frame_end=1\r\n",
        "ame_id=1,frame_end=1\r\n",
    ]


def test_send_startup_reset_emits_reset_frame() -> None:
    """启动时初始化复位应发送 reset=1."""
    sleep_short, write_line, send_startup_reset = load_functions(
        "sleep_short", "write_line", "send_startup_reset"
    )

    class FakeUart:
        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload)

    namespace_uart = FakeUart()
    write_line.__globals__["sleep_short"] = sleep_short
    send_startup_reset.__globals__["write_line"] = write_line
    send_startup_reset(namespace_uart)
    assert namespace_uart.writes == ["reset=1\r\n"]
