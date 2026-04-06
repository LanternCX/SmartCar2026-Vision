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
    """有目标时视觉帧必须切到最小协议字段集合."""
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(seq=7, valid=1, err_x=12, err_y=20)
    assert frame == "v=1,s=7,x=12,y=20"


def test_format_vision_frame_outputs_compact_invalid_frame_without_bbox() -> None:
    """无目标时视觉帧必须只发送版本位和序号."""
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(seq=8, valid=0, err_x=0, err_y=0)
    assert frame == "v=0,s=8"


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
    """最终串口帧必须只保留归一化后算出的像素差值语义."""
    blob_rect_to_bbox, normalize_bbox_for_protocol, format_vision_frame = (
        load_functions(
            "blob_rect_to_bbox", "normalize_bbox_for_protocol", "format_vision_frame"
        )
    )

    left, top, right, bottom = blob_rect_to_bbox((100, 20, 40, 70))
    raw_frame = format_vision_frame(
        seq=1,
        valid=1,
        err_x=0,
        err_y=150,
    )
    left, top, right, bottom = normalize_bbox_for_protocol(
        left, top, right, bottom, img_height=240
    )

    frame = format_vision_frame(
        seq=2,
        valid=1,
        err_x=0,
        err_y=20,
    )

    assert raw_frame == "v=1,s=1,x=0,y=150"
    assert frame == "v=1,s=2,x=0,y=20"


def test_compute_protocol_errors_uses_center_x_and_bottom_gap() -> None:
    """协议误差必须使用横向中心偏差和相对抓取位置的纵向差值."""
    (compute_protocol_errors,) = load_functions("compute_protocol_errors")
    compute_protocol_errors.__globals__["TARGET_ERR_Y"] = 60
    err_x, err_y = compute_protocol_errors(
        blob_cx=150,
        normalized_bottom=220,
        cx_screen=160,
        img_height=240,
    )
    assert (err_x, err_y) == (-10, 40)


def test_compute_protocol_errors_returns_negative_y_before_target_distance() -> None:
    """目标底边未达到期望抓取位置时, y 必须为负."""
    (compute_protocol_errors,) = load_functions("compute_protocol_errors")
    compute_protocol_errors.__globals__["TARGET_ERR_Y"] = 60
    err_x, err_y = compute_protocol_errors(
        blob_cx=170,
        normalized_bottom=150,
        cx_screen=160,
        img_height=240,
    )
    assert (err_x, err_y) == (10, -30)


def test_compute_protocol_errors_returns_zero_y_at_target_distance() -> None:
    """目标底边到达期望抓取位置时, y 必须为零."""
    (compute_protocol_errors,) = load_functions("compute_protocol_errors")
    compute_protocol_errors.__globals__["TARGET_ERR_Y"] = 60
    _, err_y = compute_protocol_errors(
        blob_cx=160,
        normalized_bottom=180,
        cx_screen=160,
        img_height=240,
    )
    assert err_y == 0


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
    write_line(namespace_uart, "v=1,s=1,x=0,y=20")
    assert namespace_uart.writes == ["v=1,s=1,x=0,y=20\r\n"]


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


def test_run_sends_single_startup_reset_and_reports_invalid_frame_without_candidates() -> (
    None
):
    """开启启动复位后,无目标时也必须按节流发送最小无效帧."""
    format_vision_frame, run = load_functions("format_vision_frame", "run")

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
    run.__globals__["format_vision_frame"] = format_vision_frame
    run.__globals__["should_send"] = lambda now_ms, last_send_ms, interval_ms: True
    run.__globals__["time"] = FakeClock()
    run.__globals__["STARTUP_RESET"] = True
    run.__globals__["SEND_INTERVAL_MS"] = 40

    try:
        run()
    except StopRun:
        pass

    assert namespace_uart.writes == [
        "reset=1\r\n",
        "v=0,s=1\r\n",
        "v=0,s=2\r\n",
    ]


def test_run_sends_minimal_valid_frame_with_protocol_errors() -> None:
    """有目标时运行时发送必须只保留序号和像素差值."""
    (
        blob_rect_to_bbox,
        normalize_bbox_for_protocol,
        compute_protocol_errors,
        format_vision_frame,
        run,
    ) = load_functions(
        "blob_rect_to_bbox",
        "normalize_bbox_for_protocol",
        "compute_protocol_errors",
        "format_vision_frame",
        "run",
    )

    class StopRun(Exception):
        pass

    class FakeClock:
        @staticmethod
        def ticks_ms():
            return 0

    class FakeImg:
        def lens_corr(self, strength, zoom):
            return None

        def height(self):
            return 240

        def draw_rectangle(self, rect):
            return None

        def draw_cross(self, x, y):
            return None

    class FakeBlob:
        def rect(self):
            return (100, 20, 40, 70)

    class FakeUart:
        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload)

    class FakeSensor:
        def __init__(self):
            self.calls = 0

        def snapshot(self):
            if self.calls == 0:
                self.calls += 1
                return FakeImg()
            raise StopRun()

    namespace_uart = FakeUart()
    run.__globals__["init_uart"] = lambda: namespace_uart
    run.__globals__["init_sensor"] = lambda: 160
    run.__globals__["send_startup_reset"] = lambda uart: None
    run.__globals__["sensor"] = FakeSensor()
    run.__globals__["build_blob_candidates"] = lambda img: [
        ("red", 150, 40, 220, FakeBlob())
    ]
    run.__globals__["choose_best_candidate"] = (
        lambda candidates, cx_screen, img_height: candidates[0]
    )
    run.__globals__["blob_rect_to_bbox"] = blob_rect_to_bbox
    run.__globals__["normalize_bbox_for_protocol"] = normalize_bbox_for_protocol
    run.__globals__["compute_protocol_errors"] = compute_protocol_errors
    run.__globals__["TARGET_ERR_Y"] = 60
    compute_protocol_errors.__globals__["TARGET_ERR_Y"] = 60
    run.__globals__["write_line"] = lambda uart, line: uart.write(line + "\r\n")
    run.__globals__["format_vision_frame"] = format_vision_frame
    run.__globals__["should_send"] = lambda now_ms, last_send_ms, interval_ms: True
    run.__globals__["time"] = FakeClock()
    run.__globals__["STARTUP_RESET"] = False
    run.__globals__["SEND_INTERVAL_MS"] = 40

    try:
        run()
    except StopRun:
        pass

    assert namespace_uart.writes == ["v=1,s=1,x=-10,y=40\r\n"]
