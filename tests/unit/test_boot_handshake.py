"""视觉启动预热与握手行为测试."""

# pyright: reportAttributeAccessIssue=false, reportOptionalSubscript=false

import pytest

from tests.test_support import decode_frame, encode_frame, load_role_main_module


class FakeLed:
    def __init__(self, led_id=0, events=None):
        self.led_id = int(led_id)
        self.events = events
        self.toggle_count = 0

    def on(self):
        if self.events is not None:
            self.events.append("led%d_on" % self.led_id)

    def off(self):
        if self.events is not None:
            self.events.append("led%d_off" % self.led_id)

    def toggle(self):
        self.toggle_count += 1
        if self.events is not None:
            self.events.append("led%d_toggle" % self.led_id)


class FakeUart:
    def __init__(self, on_write=None):
        self.incoming = bytearray()
        self.writes = []
        self.on_write = on_write

    def any(self):
        return len(self.incoming)

    def read(self, size):
        data = bytes(self.incoming[:size])
        del self.incoming[:size]
        return data

    def write(self, data):
        frame = bytes(data)
        self.writes.append(frame)
        if self.on_write is not None:
            self.on_write(self, frame)
        return len(frame)

    def push(self, data):
        self.incoming.extend(data)


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_boot_protocol_confirms_ready_bidirectionally(role) -> None:
    module = load_role_main_module(role, "vision_%s_boot_protocol" % role)
    module.reset_runtime_state()

    ready_ack = encode_frame(
        module.Mode.ACK,
        module.Topic.VISION_BOOT_READY,
        module.BOOT_READY_SEQ,
        b"",
    )
    assert module.handle_control_frame(ready_ack) is None
    assert module.state.boot_ready_acked is True

    confirm = encode_frame(
        module.Mode.TCP,
        module.Topic.VISION_BOOT_CONFIRM,
        9,
        b"",
    )
    reply = decode_frame(module.handle_control_frame(confirm))

    assert module.state.boot_confirmed is True
    assert reply["mode"] == module.Mode.ACK
    assert reply["topic"] == module.Topic.VISION_BOOT_CONFIRM
    assert reply["seq"] == 9


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_boot_ready_retries_until_confirmed(role) -> None:
    module = load_role_main_module(role, "vision_%s_boot_retry" % role)
    module.reset_runtime_state()
    clock = [0]
    module.default_now_ms = lambda: clock[0]
    module._sleep_ms = lambda delay_ms: clock.__setitem__(0, clock[0] + int(delay_ms))

    def respond_after_seven_ready_frames(uart, frame_bytes):
        frame = decode_frame(frame_bytes)
        if frame["topic"] != module.Topic.VISION_BOOT_READY:
            return
        if len(uart.writes) < 7:
            return
        uart.push(
            encode_frame(
                module.Mode.ACK,
                module.Topic.VISION_BOOT_READY,
                frame["seq"],
                b"",
            )
        )
        uart.push(
            encode_frame(
                module.Mode.TCP,
                module.Topic.VISION_BOOT_CONFIRM,
                9,
                b"",
            )
        )

    uart = FakeUart(on_write=respond_after_seven_ready_frames)
    module.state.uart_device = uart
    green = FakeLed()

    module.perform_boot_handshake(green)

    frames = [decode_frame(frame) for frame in uart.writes]
    ready_frames = [
        frame for frame in frames if frame["topic"] == module.Topic.VISION_BOOT_READY
    ]
    assert len(ready_frames) == 7
    assert green.toggle_count == 7
    assert clock[0] >= module.BOOT_READY_RESEND_INTERVAL_MS * 6


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_boot_skips_yolo_inference_during_warm_up(role, monkeypatch) -> None:
    module = load_role_main_module(role, "vision_%s_yolo_boot" % role)
    module.reset_runtime_state()
    module.OBJECT_DETECTION_USE_YOLO = True
    events = []

    class Image:
        def replace(self, **_kwargs):
            return self

        def lens_corr(self, **_kwargs):
            events.append("lens_corr")

    monkeypatch.setattr(module.sensor, "snapshot", lambda: Image())
    module.yolo_detect = lambda _img: events.append("yolo_detect")

    module.warm_up_detection()

    assert events == ["lens_corr"]


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_boot_ready_accepts_confirm_and_sends_ack(role) -> None:
    module = load_role_main_module(role, "vision_%s_boot_success" % role)
    module.reset_runtime_state()
    clock = [0]
    module.default_now_ms = lambda: clock[0]
    module._sleep_ms = lambda delay_ms: clock.__setitem__(0, clock[0] + int(delay_ms))

    def respond(uart, frame_bytes):
        frame = decode_frame(frame_bytes)
        if frame["mode"] != module.Mode.TCP:
            return
        if frame["topic"] != module.Topic.VISION_BOOT_READY:
            return
        uart.push(
            encode_frame(
                module.Mode.ACK,
                module.Topic.VISION_BOOT_READY,
                frame["seq"],
                b"",
            )
        )
        uart.push(
            encode_frame(
                module.Mode.TCP,
                module.Topic.VISION_BOOT_CONFIRM,
                11,
                b"",
            )
        )

    uart = FakeUart(on_write=respond)
    module.state.uart_device = uart
    green = FakeLed()

    module.perform_boot_handshake(green)

    frames = [decode_frame(frame) for frame in uart.writes]
    assert [(frame["mode"], frame["topic"]) for frame in frames] == [
        (module.Mode.TCP, module.Topic.VISION_BOOT_READY),
        (module.Mode.ACK, module.Topic.VISION_BOOT_CONFIRM),
    ]
    assert green.toggle_count == 1


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_prepares_lighting_and_model_before_opening_uart(role) -> None:
    module = load_role_main_module(role, "vision_%s_boot_order" % role)
    module.reset_runtime_state()
    module.MASTER_DEBUG_DISPLAY_ENABLED = False if role == "master" else getattr(
        module, "MASTER_DEBUG_DISPLAY_ENABLED", False
    )
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = False if role == "assistant" else getattr(
        module, "ASSISTANT_DEBUG_DISPLAY_ENABLED", False
    )
    events = []
    module.LED = lambda led_id: FakeLed(led_id, events)
    module.init_sensor = lambda: events.append("sensor") or (320, 240)
    module.load_yolo_model = lambda: events.append("model") or "net"
    module.warm_up_detection = lambda: events.append("warmup")
    module.init_uart = lambda: events.append("uart") or FakeUart()
    module.perform_boot_handshake = lambda _green: events.append("handshake")
    module.wait_for_first_task = lambda _blue: events.append("wait_task")

    module.prepare_runtime()

    assert events.index("led4_on") < events.index("model")
    assert events.index("model") < events.index("sensor")
    assert events.index("sensor") < events.index("warmup")
    assert events.index("warmup") < events.index("uart")
    assert events.index("uart") < events.index("handshake")
    assert events.index("handshake") < events.index("wait_task")


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_fatal_error_writes_latest_traceback_to_sd(role, monkeypatch) -> None:
    module = load_role_main_module(role, "vision_%s_error_log" % role)
    writes = []

    class LogFile:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def write(self, value):
            writes.append(value)

    opened = []
    module.open = lambda path, mode: opened.append((path, mode)) or LogFile()
    monkeypatch.setattr(
        module.sys,
        "print_exception",
        lambda error, file: file.write("RuntimeError: %s\n" % error),
        raising=False,
    )

    module.write_fatal_error(RuntimeError("boot failed"))

    assert opened == [(module.ERROR_LOG_PATH, "w")]
    assert writes == ["RuntimeError: boot failed\n"]


@pytest.mark.parametrize("role", ("master", "assistant"))
def test_visual_prepare_runtime_logs_fatal_error(role) -> None:
    module = load_role_main_module(role, "vision_%s_fatal_handler" % role)
    error = RuntimeError("sensor failed")
    logged = []
    module.LED = FakeLed
    module._sleep_ms = lambda _delay_ms: None
    module.init_sensor = lambda: (_ for _ in ()).throw(error)
    module.write_fatal_error = logged.append

    with pytest.raises(RuntimeError, match="sensor failed"):
        module.prepare_runtime()

    assert logged == [error]
