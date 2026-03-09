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


def test_format_vision_frame_only_xy() -> None:
    """视觉帧必须严格只包含 x,y 两个键."""
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(120, 80)
    assert frame == "x=120,y=80"


def test_choose_best_candidate_prefers_center_bottom() -> None:
    """目标选择优先靠近中线且靠近底边."""
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [("Red", 40, 200), ("Green", 158, 230), ("Red", 220, 220)]
    best = choose_best_candidate(candidates, cx_screen=160, img_height=240)
    assert best == ("Green", 158, 230)


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
    write_line(namespace_uart, "x=120,y=80")
    assert namespace_uart.writes == ["x=120,y=80\r\n"]


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
