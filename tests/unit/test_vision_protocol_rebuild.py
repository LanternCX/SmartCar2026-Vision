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


def test_format_vision_frame_outputs_only_bbox_fields() -> None:
    """视觉帧只能输出完整识别框的四个边界键."""
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(100, 20, 140, 90)
    assert frame == "left=100,top=20,right=140,bottom=90"


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
    """最终串口帧必须在视觉端内部完成归一化后再发送."""
    blob_rect_to_bbox, normalize_bbox_for_protocol, format_vision_frame = (
        load_functions(
            "blob_rect_to_bbox", "normalize_bbox_for_protocol", "format_vision_frame"
        )
    )

    left, top, right, bottom = blob_rect_to_bbox((100, 20, 40, 70))
    raw_frame = format_vision_frame(left, top, right, bottom)
    left, top, right, bottom = normalize_bbox_for_protocol(
        left, top, right, bottom, img_height=240
    )

    frame = format_vision_frame(left, top, right, bottom)

    assert raw_frame == "left=100,top=20,right=140,bottom=90"
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


def test_should_send_throttle_boundary() -> None:
    """发送节流边界必须做到不早发也不漏发."""
    (should_send,) = load_functions("should_send")
    assert should_send(now_ms=100, last_send_ms=None, interval_ms=30) is True
    assert should_send(now_ms=129, last_send_ms=100, interval_ms=30) is False
    assert should_send(now_ms=130, last_send_ms=100, interval_ms=30) is True
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


def test_send_startup_reset_emits_exactly_one_reset_line() -> None:
    """启动复位只能发送一条 reset=1 协议行."""
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


def test_choose_best_candidate_prefers_smaller_score() -> None:
    """目标选择应优先返回分值更小的候选."""
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [("Red", 40, 0, 200), ("Green", 158, 0, 230)]
    best = choose_best_candidate(candidates, cx_screen=160, img_height=240)
    assert best == ("Green", 158, 0, 230)


def test_choose_best_candidate_follows_task3_spec_by_keeping_first_candidate_on_tie() -> (
    None
):
    """任务 3 明确要求同分取第一个候选,因此这里锁死输入顺序语义."""
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [("Red", 150, 0, 230), ("Green", 170, 0, 230)]
    expected_by_task3_spec = candidates[0]
    best = choose_best_candidate(candidates, cx_screen=160, img_height=240)
    assert best == expected_by_task3_spec


def test_run_sends_single_startup_reset_and_stays_silent_without_candidates() -> None:
    """开启启动复位后,运行周期内最多只发一次 reset 且无目标时不发观测帧."""
    (run,) = load_functions("run")

    class StopRun(Exception):
        pass

    class FakeClock:
        @staticmethod
        def ticks_ms():
            return 0

    class FakeImg:
        def lens_corr(self, strength, zoom):
            return None

    class FakeUart:
        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload)

    class FakeSensor:
        def __init__(self):
            self.calls = 0

        def snapshot(self):
            if self.calls < 2:
                self.calls += 1
                return FakeImg()
            raise StopRun()

    namespace_uart = FakeUart()
    run.__globals__["init_uart"] = lambda: namespace_uart
    run.__globals__["init_sensor"] = lambda: 160
    run.__globals__["send_startup_reset"] = lambda uart: uart.write("reset=1\r\n")
    run.__globals__["sensor"] = FakeSensor()
    run.__globals__["build_blob_candidates"] = lambda img: []
    run.__globals__["choose_best_candidate"] = (
        lambda candidates, cx_screen, img_height: None
    )
    run.__globals__["blob_rect_to_bbox"] = lambda rect: rect
    run.__globals__["normalize_bbox_for_protocol"] = (
        lambda left, top, right, bottom, img_height: (
            left,
            top,
            right,
            bottom,
        )
    )
    run.__globals__["write_line"] = lambda uart, line: uart.write(line + "\r\n")
    run.__globals__["format_vision_frame"] = lambda left, top, right, bottom: "unused"
    run.__globals__["should_send"] = lambda now_ms, last_send_ms, interval_ms: True
    run.__globals__["time"] = FakeClock()
    run.__globals__["STARTUP_RESET"] = True
    run.__globals__["SEND_INTERVAL_MS"] = 40

    try:
        run()
    except StopRun:
        pass

    assert namespace_uart.writes == ["reset=1\r\n"]
