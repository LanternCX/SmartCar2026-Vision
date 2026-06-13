"""主车视觉 main_v2 行为测试."""

import pytest

from tests.test_support import (
    MODE_ACK,
    MODE_TCP,
    MODE_UDP,
    decode_frame,
    decode_master_vision_event_report_body,
    decode_velocity_body,
    encode_frame,
    load_role_entry_module,
)


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
YELLOW_PIXEL = (70, -20, 70)


def load_master_v2():
    module = load_role_entry_module("master", "main_v2.py", "vision_master_v2_test_module")
    module.reset_runtime_state()
    module.OBJECT_STABLE_FRAMES = 1
    module.state.yolo_net = "fake-net"
    return module


class FakeUART:
    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)


class FakeBlob:
    def __init__(self, left, top, width, height, area):
        self._rect = (left, top, width, height)
        self._area = area

    def rect(self):
        return self._rect

    def area(self):
        return self._area


class FakeImage:
    def __init__(self, detections=None, yellow_area_by_roi=None, pixels=None):
        self.detections = list(detections or ())
        self.yellow_area_by_roi = dict(yellow_area_by_roi or {})
        self.pixels = dict(pixels or {})
        self.copy_calls = []
        self.lens_calls = []
        self.rectangles = []
        self.strings = []
        self.crosses = []
        self.lines = []
        self.flush_count = 0

    def width(self):
        return IMAGE_WIDTH

    def height(self):
        return IMAGE_HEIGHT

    def copy(self, scale, flag):
        self.copy_calls.append((scale, flag))
        return self

    def lens_corr(self, strength, zoom):
        self.lens_calls.append((strength, zoom))
        return self

    def draw_rectangle(self, rect, color=None, thickness=1):
        self.rectangles.append((rect, color, thickness))

    def draw_string(self, x, y, text, color=None, scale=1, mono_space=False):
        self.strings.append((x, y, text, color, scale, mono_space))

    def draw_cross(self, x, y, color=None):
        self.crosses.append((x, y, color))

    def draw_line(self, x0, y0, x1, y1, color=None):
        self.lines.append((x0, y0, x1, y1, color))

    def flush(self):
        self.flush_count += 1

    def find_blobs(
        self,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge,
        roi=None,
        margin=None,
    ):
        _ = thresholds, pixels_threshold, area_threshold, merge, margin
        if roi is None:
            return []
        area = self.yellow_area_by_roi.get(tuple(roi), 0)
        if area <= 0:
            return []
        return [FakeBlob(roi[0], roi[1], roi[2], roi[3], area)]

    def get_pixel(self, x, y):
        return self.pixels.get((x, y), (0, 0, 0))


def task_sync_frame(module, seq=12, context_id=7, state=None, target=None, arg=None):
    if state is None:
        state = int(module.STATE_SEARCH_OBJECT)
    if target is None:
        target = int(module.TARGET_OBJECT)
    if arg is None:
        arg = int(module.MASTER_SEARCH_TASK_CONFIG_ID)
    return encode_frame(
        MODE_TCP,
        module.TOPIC_MASTER_VISION_TASK_SYNC,
        seq,
        module.encode_master_vision_task_sync_body(context_id, state, target, arg),
    )


def event_ack_frame(module, seq):
    return encode_frame(MODE_ACK, module.TOPIC_MASTER_VISION_EVENT_REPORT, seq, b"")


def search_aligned_detection():
    return (150.0 / 320.0, 30.0 / 240.0, 170.0 / 320.0, 50.0 / 240.0, 1, 0.95)


def transport_aligned_detection():
    return (150.0 / 320.0, 0.0, 170.0 / 320.0, 20.0 / 240.0, 1, 0.95)


def pixel_detection(left, top, right, bottom, label=1, score=0.95):
    return (
        float(left) / float(IMAGE_WIDTH),
        float(top) / float(IMAGE_HEIGHT),
        float(right) / float(IMAGE_WIDTH),
        float(bottom) / float(IMAGE_HEIGHT),
        label,
        score,
    )


def set_protocol_yellow(pixels, x, y):
    pixels[(IMAGE_WIDTH - 1 - int(x), IMAGE_HEIGHT - 1 - int(y))] = YELLOW_PIXEL


def build_return_line_pixels(y, start_x=90, end_x=230):
    pixels = {}
    for x in range(start_x, end_x + 1):
        set_protocol_yellow(pixels, x, y)
    return pixels


def build_return_line_band_pixels(top, bottom, left=130, right=190):
    pixels = {}
    for y in range(int(top), int(bottom) + 1):
        for x in range(int(left), int(right) + 1):
            set_protocol_yellow(pixels, x, y)
    return pixels


def latest_velocity(uart):
    frame = decode_frame(uart.writes[-1])
    assert frame is not None
    assert frame["mode"] == MODE_UDP
    return decode_velocity_body(frame["body"])


def latest_event(uart):
    for frame_bytes in reversed(uart.writes):
        frame = decode_frame(frame_bytes)
        if frame is None:
            continue
        if frame["mode"] == MODE_TCP and frame["topic"] == 0x12:
            return decode_master_vision_event_report_body(frame["body"])
    raise AssertionError("missing event frame")


class ReadWriteUART:
    def __init__(self, data):
        self._data = data
        self.writes = []
        self.topic = 0x12

    def any(self):
        return len(self._data)

    def read(self, size):
        data = self._data[:size]
        self._data = self._data[size:]
        return data

    def write(self, data):
        self.writes.append(data)
        return len(data)


class BadReadUART:
    def __init__(self):
        self.topic = 0x12

    def any(self):
        return 1

    def read(self, size):
        _ = size
        return b"\xff"

    def write(self, data):
        _ = data
        return 0


def build_search_observation(module, area, err_x=0.0, err_y=0.0):
    target_x, target_y = module.build_search_target_point(module.MASTER_SEARCH_TASK_CONFIG_ID)
    return module.build_observation(1, target_x + err_x, target_y + err_y, area)


def cache_yolo_candidates(module, img=None):
    if img is None:
        img = FakeImage()
    module.state.current_yolo_candidates = tuple(module.yolo_detect(img))
    return img


def run_frame(module, img):
    cache_yolo_candidates(module, img)
    module.process_task_frame(img)


def test_master_main_v2_replies_task_sync_ack_and_records_task() -> None:
    module = load_master_v2()

    reply = module.handle_control_frame(task_sync_frame(module, seq=12, context_id=7))

    assert decode_frame(reply) == {
        "mode": MODE_ACK,
        "topic": module.TOPIC_MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 8,
    }
    assert module.state.current_task["context_id"] == 7
    assert module.state.current_task["arg"] == module.MASTER_SEARCH_TASK_CONFIG_ID


def test_master_main_v2_runtime_state_uses_state_object() -> None:
    module = load_master_v2()

    assert hasattr(module, "state")
    module.reset_runtime_state(next_event_seq=9)

    assert module.state.current_task is None
    assert module.state.next_event_seq == 9
    assert not hasattr(module, "CURRENT_TASK")


def test_master_main_v2_process_uart_input_replies_ack_for_task_sync_frame() -> None:
    module = load_master_v2()
    uart = ReadWriteUART(task_sync_frame(module))
    module.state.uart_device = uart

    rx_buffer = module.process_uart_input(b"")

    assert rx_buffer == b""
    assert decode_frame(uart.writes[0]) == {
        "mode": MODE_ACK,
        "topic": module.TOPIC_MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 8,
    }
    assert module.state.current_task is not None


def test_master_main_v2_process_uart_input_resyncs_before_task_sync_frame() -> None:
    module = load_master_v2()
    uart = ReadWriteUART(b"\x02" + task_sync_frame(module))
    module.state.uart_device = uart

    rx_buffer = module.process_uart_input(b"")

    assert rx_buffer == b""
    assert decode_frame(uart.writes[0])["topic"] == module.TOPIC_MASTER_VISION_TASK_SYNC
    assert module.state.current_task is not None


def test_master_main_v2_process_uart_input_ignores_bad_decode() -> None:
    module = load_master_v2()
    module.state.uart_device = BadReadUART()

    rx_buffer = module.process_uart_input(b"partial")

    assert rx_buffer == b"partial\xff"
    assert module.state.current_task is None


def test_master_main_v2_repeated_task_sync_replies_ack_without_reapplying() -> None:
    module = load_master_v2()
    module.OBJECT_STABLE_FRAMES = 2

    reply = module.handle_control_frame(task_sync_frame(module))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())
    repeated_reply = module.handle_control_frame(task_sync_frame(module))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())

    assert decode_frame(reply)["seq"] == 12
    assert decode_frame(repeated_reply)["seq"] == 12
    assert module.state.stable_frame_count == 2
    assert module.state.pending_event is not None


def test_master_main_v2_non_new_context_does_not_override_active_task() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.handle_control_frame(
        task_sync_frame(module, seq=13, context_id=6, state=int(module.STATE_ORBITING), arg=9)
    )

    observation = build_search_observation(module, 180.0)

    assert module.state.current_task == {
        "context_id": 7,
        "state": module.STATE_SEARCH_OBJECT,
        "target": module.TARGET_OBJECT,
        "arg": module.MASTER_SEARCH_TASK_CONFIG_ID,
    }
    assert observation == (7, 0.0, 0.0, 180.0)


def test_master_main_v2_wrong_ack_does_not_clear_pending_event() -> None:
    module = load_master_v2()
    now_ms = [100]
    module.default_now_ms = lambda: now_ms[0]
    module.RELIABLE_RESEND_INTERVAL_MS = 20
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())

    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 12))
    now_ms[0] += 20
    second = module.next_event_frame()

    assert decode_frame(first)["seq"] == 1
    assert second == first


def test_master_main_v2_repeats_event_until_matching_ack() -> None:
    module = load_master_v2()
    now_ms = [100]
    module.default_now_ms = lambda: now_ms[0]
    module.RELIABLE_RESEND_INTERVAL_MS = 20
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())

    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 0))
    now_ms[0] += 20
    second = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 1))

    assert second == first
    assert module.next_event_frame() is None


def test_master_main_v2_creates_target_found_once_per_context() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())
    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 1))
    module.accept_observation(build_search_observation(module, 180.0), FakeImage())

    assert first is not None
    assert module.next_event_frame() is None


def test_master_main_v2_search_uses_yolo_candidates_for_velocity_and_target_found() -> None:
    module = load_master_v2()
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage(detections=[search_aligned_detection()])

    run_frame(module, img)
    run_frame(module, img)

    assert img.copy_calls == [
        (module.YOLO_IMAGE_COPY_SCALE, 1),
        (module.YOLO_IMAGE_COPY_SCALE, 1),
    ]
    velocity = latest_velocity(type("U", (), {"writes": [uart.writes[0]]})())
    assert velocity == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }
    assert latest_event(type("U", (), {"writes": [uart.writes[1]]})()) == {
        "context_id": 7,
        "event": module.EVENT_TARGET_FOUND,
        "value": 1,
    }


def test_master_main_v2_debug_display_draws_detected_box_and_flushes() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.state.yolo_net = "fake-net"
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)

    assert img.rectangles
    assert any(entry[2] == "red" for entry in img.strings)
    assert img.crosses
    assert img.flush_count == 1


def test_master_main_v2_debug_display_shows_preview_without_task_sync() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)

    assert img.rectangles
    assert any(entry[2] == "red" for entry in img.strings)
    assert img.crosses
    assert img.flush_count == 1
    assert uart.writes == []


def test_master_main_v2_build_observation_and_candidates_uses_cached_candidates_without_current_image() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    target_x, target_y = module.build_search_target_point(module.MASTER_SEARCH_TASK_CONFIG_ID)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_yolo_candidates = (("red", target_x, target_y, 400.0, blob),)

    observation, best_blob, task_name, candidates = module.build_observation_and_candidates()

    assert observation == (7, 0.0, 0.0, 400.0)
    assert best_blob is blob
    assert task_name == "red"
    assert candidates == (("red", target_x, target_y, 400.0, blob),)


def test_master_main_v2_missing_target_outputs_configured_search_velocity() -> None:
    module = load_master_v2()
    velocity = module.build_search_velocity_from_observation((7, 0.0, 0.0, 0.0), IMAGE_HEIGHT)

    assert velocity == (module.MASTER_MISSING_SEARCH_VX, module.MASTER_MISSING_SEARCH_VY)


def test_master_main_v2_search_velocity_deadzone_zeroes_each_axis() -> None:
    module = load_master_v2()
    observation = build_search_observation(
        module,
        150.0,
        err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX),
        err_y=float(module.MASTER_SEARCH_DEADZONE_Y_PX),
    )

    velocity = module.build_search_velocity_from_observation(observation, IMAGE_HEIGHT)

    assert velocity == (0.0, 0.0)


def test_master_main_v2_search_velocity_applies_min_speed_outside_deadzone() -> None:
    module = load_master_v2()
    positive = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1,
            err_y=float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1,
        ),
        IMAGE_HEIGHT,
    )
    negative = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=-(float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1),
            err_y=-(float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1),
        ),
        IMAGE_HEIGHT,
    )

    assert positive == (
        pytest.approx(module.MASTER_SEARCH_MIN_SPEED),
        pytest.approx(-module.MASTER_SEARCH_MIN_SPEED),
    )
    assert negative == (
        pytest.approx(-module.MASTER_SEARCH_MIN_SPEED),
        pytest.approx(module.MASTER_SEARCH_MIN_SPEED),
    )


def test_master_main_v2_search_velocity_clamps_vx_and_vy() -> None:
    module = load_master_v2()
    positive = module.build_search_velocity_from_observation((7, 999.0, 999.0, 300.0), IMAGE_HEIGHT)
    negative = module.build_search_velocity_from_observation((7, -999.0, -999.0, 300.0), IMAGE_HEIGHT)

    expected_positive_vx = (
        module.MASTER_SEARCH_MAX_VX
        if module.MASTER_SEARCH_KP_X > 0
        else -module.MASTER_SEARCH_MAX_VX
    )
    expected_positive_vy = (
        module.MASTER_SEARCH_MAX_VY
        if module.MASTER_SEARCH_KP_Y > 0
        else -module.MASTER_SEARCH_MAX_VY
    )

    assert positive == (expected_positive_vx, expected_positive_vy)
    assert negative == (-expected_positive_vx, -expected_positive_vy)


def test_master_main_v2_reference_fps_keeps_search_velocity_unchanged() -> None:
    module = load_master_v2()
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms()

    velocity = module.build_search_velocity_from_observation(
        (7, 100.0, -100.0, 300.0),
        IMAGE_HEIGHT,
    )

    assert velocity == pytest.approx((5.0, 2.0833333333333335))


def test_master_main_v2_reference_frame_interval_is_derived_from_fps() -> None:
    module = load_master_v2()
    module.VISION_REFERENCE_FPS = 25

    assert module.reference_frame_interval_ms() == pytest.approx(40.0)


def test_master_main_v2_slow_frame_interval_reduces_search_velocity() -> None:
    module = load_master_v2()
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms()
    reference_velocity = module.build_search_velocity_from_observation(
        (7, 100.0, -100.0, 300.0),
        IMAGE_HEIGHT,
    )
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms() * 2
    slow_velocity = module.build_search_velocity_from_observation(
        (7, 100.0, -100.0, 300.0),
        IMAGE_HEIGHT,
    )

    assert abs(slow_velocity[0]) < abs(reference_velocity[0])
    assert abs(slow_velocity[1]) < abs(reference_velocity[1])
    assert slow_velocity == pytest.approx((reference_velocity[0] * 0.5, reference_velocity[1] * 0.5))


def test_master_main_v2_search_y_velocity_decreases_when_target_gets_closer() -> None:
    module = load_master_v2()
    far_velocity = module.build_search_velocity_from_observation((7, 0.0, -200.0, 1000.0), IMAGE_HEIGHT)
    close_velocity = module.build_search_velocity_from_observation((7, 0.0, -100.0, 1000.0), IMAGE_HEIGHT)

    assert close_velocity[1] < far_velocity[1]


def test_master_main_v2_orbit_outputs_only_velocity_correction_without_event() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.MASTER_ORBIT_KP_X = 0.2
    module.MASTER_ORBIT_KP_Y = -0.3
    module.MASTER_ORBIT_MIN_SPEED = 0.0
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.STATE_ORBITING),
            arg=int(module.MASTER_ORBIT_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)

    velocity = latest_velocity(uart)
    assert velocity["vx"] == pytest.approx(0.0)
    assert velocity["vy"] == pytest.approx(0.0)
    assert len(uart.writes) == 1


def test_master_main_v2_transport_alignment_reports_aligned_event() -> None:
    module = load_master_v2()
    module.tf.detect = lambda net, img: [transport_aligned_detection()]
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=9,
            state=int(module.STATE_SEARCH_OBJECT),
            arg=int(module.MASTER_TRANSPORT_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)
    run_frame(module, img)

    assert latest_event(type("U", (), {"writes": [uart.writes[1]]})()) == {
        "context_id": 9,
        "event": module.EVENT_ALIGNED,
        "value": 400,
    }


def test_master_main_v2_candidate_selection_uses_configured_target_point() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.MASTER_SEARCH_TASK_CONFIG_ID)
    module.tf.detect = lambda net, img: [
        pixel_detection(target_x - 40, IMAGE_HEIGHT - target_y, target_x + 40, IMAGE_HEIGHT - target_y + 20),
        pixel_detection(
            target_x - 40,
            IMAGE_HEIGHT - (target_y + 20.0),
            target_x + 40,
            IMAGE_HEIGHT - (target_y + 20.0) + 20,
        ),
    ]
    module.handle_control_frame(task_sync_frame(module))

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation == (7, 0.0, 0.0, 1600.0)


def test_master_main_v2_transport_alignment_keeps_candidate_selection() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.MASTER_TRANSPORT_TASK_CONFIG_ID)
    module.tf.detect = lambda net, img: [
        pixel_detection(
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 10.0,
            IMAGE_HEIGHT - target_y,
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 30.0,
            IMAGE_HEIGHT - target_y + 20,
        )
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.STATE_SEARCH_OBJECT),
            target=int(module.TARGET_OBJECT),
            arg=int(module.MASTER_TRANSPORT_TASK_CONFIG_ID),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[3] == pytest.approx(400.0)


def test_master_main_v2_finish_accepts_x_outside_when_bottom_hits_target_window() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID)
    module.tf.detect = lambda net, img: [
        pixel_detection(
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 10.0,
            IMAGE_HEIGHT - target_y,
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 30.0,
            IMAGE_HEIGHT - target_y + 20,
        )
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.STATE_TRANSPORT_OBJECT),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[3] == pytest.approx(400.0)


def test_master_main_v2_finish_prefers_largest_area_after_bottom_filter() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID)
    module.tf.detect = lambda net, img: [
        pixel_detection(target_x - 10.0, IMAGE_HEIGHT - target_y, target_x + 10.0, IMAGE_HEIGHT - target_y + 20),
        pixel_detection(
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 10.0,
            IMAGE_HEIGHT - target_y,
            target_x + float(module.OBJECT_X_TOLERANCE_PX) + 50.0,
            IMAGE_HEIGHT - target_y + 20,
        ),
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.STATE_TRANSPORT_OBJECT),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[1] == pytest.approx(float(module.OBJECT_X_TOLERANCE_PX) + 30.0)
    assert observation[3] == pytest.approx(800.0)


def test_master_main_v2_finish_ignores_candidates_when_bottom_outside_target_window() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID)
    module.tf.detect = lambda net, img: [
        pixel_detection(
            target_x - 10.0,
            IMAGE_HEIGHT - (target_y + float(module.OBJECT_Y_TOLERANCE_PX) + 10.0),
            target_x + 10.0,
            IMAGE_HEIGHT - (target_y + float(module.OBJECT_Y_TOLERANCE_PX) + 10.0) + 20,
        )
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.STATE_TRANSPORT_OBJECT),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is None
    assert observation == (7, 0.0, 0.0, 0.0)


def test_master_main_v2_finish_uses_target_window_and_reports_arrived_after_contact_release() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.FINISH_HOOK_STABLE_FRAMES = 1
    module.tf.detect = lambda net, img: [
        (145.0 / 320.0, 0.0, 175.0 / 320.0, 20.0 / 240.0, 1, 0.96),
        (150.0 / 320.0, 40.0 / 240.0, 170.0 / 320.0, 60.0 / 240.0, 1, 0.97),
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=11,
            state=int(module.STATE_TRANSPORT_OBJECT),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    blob = module.YoloDetectionBlob(145.0, 0.0, 175.0, 20.0, 1, 0.96)
    rois, _ = module.build_finish_task_ring_rois(blob, IMAGE_WIDTH, IMAGE_HEIGHT)
    touch_areas = {tuple(roi): max(1, int(roi[2]) * int(roi[3])) for roi in rois}

    run_frame(module, FakeImage(yellow_area_by_roi=touch_areas))
    run_frame(module, FakeImage(yellow_area_by_roi={}))
    run_frame(module, FakeImage(yellow_area_by_roi={}))

    assert latest_event(uart) == {
        "context_id": 11,
        "event": module.EVENT_ARRIVED,
        "value": 0,
    }


def test_master_main_v2_return_retreat_reports_line_aligned() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=13,
            state=int(module.STATE_RETURN_GARAGE_RETREAT),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage(pixels=build_return_line_pixels(160))

    run_frame(module, img)
    run_frame(module, img)
    run_frame(module, img)

    assert latest_event(uart) == {
        "context_id": 13,
        "event": module.EVENT_RETURN_LINE_ALIGNED,
        "value": 160,
    }


def test_master_main_v2_return_line_y_uses_center_columns_bounds_average() -> None:
    module = load_master_v2()

    line_y = module.build_return_line_y_from_image(
        FakeImage(pixels=build_return_line_band_pixels(180, 200)),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert line_y == pytest.approx(190.0)


def test_master_main_v2_return_line_velocity_uses_y_target_only() -> None:
    module = load_master_v2()
    module.RETURN_GARAGE_LINE_TARGET_Y_PX = 90.0
    module.RETURN_GARAGE_LINE_DEADZONE_Y_PX = 2.0
    module.RETURN_GARAGE_LINE_KP_Y = -0.5
    module.RETURN_GARAGE_LINE_MAX_VY = 10.0
    module.RETURN_GARAGE_LINE_MIN_SPEED = 0.0

    assert module.build_return_line_velocity_from_y(None) == (0.0, 0.0)
    assert module.build_return_line_velocity_from_y(80.0) == pytest.approx((0.0, 5.0))
    assert module.build_return_line_velocity_from_y(90.0) == pytest.approx((0.0, 0.0))
    assert module.build_return_line_velocity_from_y(100.0) == pytest.approx((0.0, -5.0))


def test_master_main_v2_return_line_limits_wide_yellow_to_lower_30px() -> None:
    module = load_master_v2()
    module.RETURN_GARAGE_LINE_MAX_THICKNESS_PX = 30

    line_y = module.build_return_line_y_from_image(
        FakeImage(pixels=build_return_line_band_pixels(80, 200)),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert line_y == pytest.approx(185.0)


def test_master_main_v2_return_line_keeps_previous_when_horizontal_connected_is_too_short() -> None:
    module = load_master_v2()
    module.RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50

    line_y = module.build_return_line_y_from_image(
        FakeImage(pixels=build_return_line_band_pixels(180, 200, left=150, right=170)),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        188.0,
    )

    assert line_y == pytest.approx(188.0)


def test_master_main_v2_return_line_runtime_does_not_use_blob_detection() -> None:
    module = load_master_v2()
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=7,
            state=int(module.STATE_RETURN_GARAGE_LINE),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart

    class RawPixelForbiddenImage(FakeImage):
        def find_blobs(self, *args, **kwargs):
            raise AssertionError("回库黄线算法不应调用 find_blobs")

    run_frame(module, RawPixelForbiddenImage())

    assert len(uart.writes) == 1


def test_master_main_v2_return_line_reports_finished_after_x270_missing_for_five_frames() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.RETURN_LINE_MISSING_FINISH_FRAMES = 5
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=15,
            state=int(module.STATE_RETURN_GARAGE_LINE),
            target=int(module.TARGET_EDGE_LINE),
            arg=int(module.MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage(pixels=build_return_line_pixels(160))

    for _ in range(7):
        run_frame(module, img)

    assert latest_event(uart) == {
        "context_id": 15,
        "event": module.EVENT_RETURN_GARAGE_FINISHED,
        "value": 0,
    }


def test_master_main_v2_pending_event_blocks_non_return_velocity_until_ack() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(task_sync_frame(module, context_id=21))
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)
    run_frame(module, img)
    run_frame(module, img)

    assert decode_frame(uart.writes[-1])["topic"] == module.TOPIC_MASTER_VISION_EVENT_REPORT
    module.handle_control_frame(event_ack_frame(module, 1))
    run_frame(module, img)

    assert decode_frame(uart.writes[-1])["topic"] == module.TOPIC_LOCAL_VISION_VELOCITY


def test_master_main_v2_run_applies_lens_correction_and_uses_yolo_detect_before_processing() -> None:
    module = load_master_v2()

    class StopLoop(Exception):
        pass

    class SnapshotImage(FakeImage):
        def __init__(self):
            super().__init__()
            self.lens_corr_called = False

        def lens_corr(self, strength, zoom):
            super().lens_corr(strength, zoom)
            _ = strength, zoom
            self.lens_corr_called = True
            return self

    image = SnapshotImage()
    call_log = []

    class Sensor:
        def snapshot(self):
            call_log.append("snapshot")
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = lambda rx_buffer: rx_buffer

    def fake_yolo_detect(img):
        assert img is image
        assert image.lens_corr_called is True
        call_log.append("yolo_detect")
        return [("red", 160.0, 210.0, 300.0, None)]

    def stop_after_frame(current_img):
        assert current_img is image
        assert call_log == ["snapshot", "yolo_detect"]
        assert tuple(module.state.current_yolo_candidates) == (("red", 160.0, 210.0, 300.0, None),)
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()
