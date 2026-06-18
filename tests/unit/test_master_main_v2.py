"""主车视觉 main_v2 行为测试."""

import inspect
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


def set_current_image(module, img):
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()


def start_run_as_local_vision_paused(module):
    original_reset_runtime_state = module.reset_runtime_state

    def reset_as_paused():
        original_reset_runtime_state()
        module.state.local_vision_control_paused = True

    module.reset_runtime_state = reset_as_paused


def test_master_main_v2_default_configuration_reports_target_found_after_configured_observations() -> None:
    module = load_role_entry_module(
        "master",
        "main_v2.py",
        "vision_master_v2_default_stable_frames_test_module",
    )
    module.reset_runtime_state()
    module.state.yolo_net = "fake-net"
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    img = FakeImage()
    set_current_image(module, img)
    for _ in range(int(module.OBJECT_STABLE_FRAMES) - 1):
        module.accept_observation(build_search_observation(module, 180.0), img)
        assert module.next_event_frame() is None

    module.accept_observation(build_search_observation(module, 180.0), img)

    assert latest_event(type("U", (), {"writes": [module.next_event_frame()]})()) == {
        "context_id": 7,
        "event": module.Event.TARGET_FOUND,
        "value": 180,
    }


def test_master_main_v2_process_task_and_velocity_helpers_drop_frame_size_parameters() -> None:
    module = load_master_v2()

    assert tuple(inspect.signature(module.process_task_frame).parameters) == ("img",)
    assert tuple(inspect.signature(module.yolo_detect).parameters) == ("img",)
    assert tuple(inspect.signature(module.normalize_bbox_for_protocol).parameters) == (
        "left",
        "top",
        "right",
        "bottom",
    )
    assert tuple(inspect.signature(module.build_search_velocity_from_observation).parameters) == (
        "observation",
    )
    assert tuple(
        inspect.signature(module.build_orbit_correction_velocity_from_observation).parameters
    ) == ("observation",)
    assert tuple(inspect.signature(module.build_return_line_y_from_image).parameters) == (
        "img",
        "previous_line_y",
    )
    assert tuple(inspect.signature(module.build_finish_task_ring_rois).parameters) == ("blob", "img")
    assert tuple(inspect.signature(module.build_finish_task_fixed_object_roi).parameters) == ("img",)
    assert tuple(inspect.signature(module.build_finish_task_yellow_ratio_percent).parameters) == ("img", "blob")
    assert tuple(inspect.signature(module.draw_protocol_target_point_debug).parameters) == (
        "img",
        "target_x",
        "target_y",
    )
    assert tuple(inspect.signature(module.draw_return_line_debug).parameters) == (
        "img",
        "line_y",
        "velocity",
    )
    assert tuple(inspect.signature(module.draw_search_preview_debug).parameters) == ("img", "candidates")
    assert tuple(inspect.signature(module.draw_tracking_state_debug).parameters) == ("img",)
    assert tuple(inspect.signature(module._return_line_pixel_matches).parameters) == ("img", "x", "y")
    assert tuple(inspect.signature(module._return_line_has_horizontal_connected_at).parameters) == (
        "img",
        "x",
        "y",
        "required_connected",
    )
    assert tuple(inspect.signature(module._return_line_y_on_column).parameters) == ("img", "x")
    assert tuple(inspect.signature(module._count_yellow_pixels_in_roi).parameters) == ("img", "roi")
    assert tuple(inspect.signature(module.build_task_event_value).parameters) == (
        "img",
        "best_blob",
        "task_name",
    )
    assert tuple(inspect.signature(module.accept_observation).parameters) == (
        "observation",
        "img",
        "event_value",
    )


def test_master_main_v2_run_uses_yolo_gate_for_debug_preview_without_track() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True

    assert module.should_run_yolo_for_current_frame() is True


def test_master_main_v2_run_skips_yolo_gate_when_short_tracking_is_active() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    _seed_master_short_track(module)

    assert module.should_run_yolo_for_current_frame() is False


def test_master_main_v2_run_skips_yolo_after_entering_transport_finish() -> None:
    module = load_master_v2()
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )

    assert module.should_run_yolo_for_current_frame() is False


def test_master_main_v2_transport_finish_runtime_does_not_call_yolo_detect() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True

    class StopLoop(Exception):
        pass

    image = FakeImage(yellow_area_by_roi={})

    class Sensor:
        def snapshot(self):
            return image

    uart = FakeUART()
    module.sensor = Sensor()
    module.init_uart = lambda: uart
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)

    def prime_finish_task(rx_buffer):
        module.state.current_task = {
            "context_id": 25,
            "state": int(module.State.TRANSPORT_OBJECT),
            "target": int(module.Target.EDGE_LINE),
            "arg": int(module.Task.TRANSPORT_FINISH),
        }
        return rx_buffer

    def _forbidden_detect(net, img):
        raise AssertionError("TRANSPORT_OBJECT 不应调用 YOLO")

    module.tf.detect = _forbidden_detect
    module.process_uart_input = prime_finish_task

    def stop_after_velocity(frame_bytes):
        uart.write(frame_bytes)
        raise StopLoop()

    module.write_data_line = stop_after_velocity

    with pytest.raises(StopLoop):
        module.run()

    assert latest_velocity(uart) == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }


def test_master_main_v2_yolo_only_mode_runs_yolo_every_configured_interval() -> None:
    module = load_master_v2()
    module.MASTER_YOLO_ONLY_INTERVAL_FRAMES = 3
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.SEARCH_OBJECT),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )

    assert module.should_run_yolo_for_current_frame() is True

    module.record_yolo_frame_run()

    assert module.should_run_yolo_for_current_frame() is False
    assert module.should_run_yolo_for_current_frame() is False
    assert module.should_run_yolo_for_current_frame() is True


def test_master_main_v2_transport_align_ignores_blob_tracking_candidates() -> None:
    module = load_master_v2()
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.SEARCH_OBJECT),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )
    _seed_master_short_track(module)

    class BlobForbiddenImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("主车搬运前对正阶段不应调用色块识别")

    img = BlobForbiddenImage()
    candidates = module.build_object_candidates(img, ())

    assert candidates == ()
    assert module.state.current_detection_source == "yolo"


def test_master_main_v2_yolo_miss_waits_configured_retry_frames() -> None:
    module = load_master_v2()
    module.MASTER_YOLO_RETRY_SKIP_FRAMES = 2
    module.handle_control_frame(task_sync_frame(module))

    module.record_yolo_retry_miss()

    assert module.state.yolo_retry_skip_frames_remaining == 2
    assert module.should_run_yolo_for_current_frame() is False
    assert module.state.yolo_retry_skip_frames_remaining == 1
    assert module.should_run_yolo_for_current_frame() is False
    assert module.state.yolo_retry_skip_frames_remaining == 0

    assert module.should_run_yolo_for_current_frame() is True


def test_master_main_v2_run_skips_yolo_after_entering_orbit() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=11))
    _seed_master_short_track(module)
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.ORBITING),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.ORBIT),
        )
    )

    assert module.should_run_yolo_for_current_frame() is False


def test_master_main_v2_orbit_blob_only_preserves_previous_track() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=11))
    _seed_master_short_track(module)

    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.ORBITING),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.ORBIT),
        )
    )

    class OrbitBlob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 260.0

    class OrbitBlobImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(
            self,
            thresholds,
            pixels_threshold,
            area_threshold,
            merge,
            roi=None,
            margin=None,
        ):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [OrbitBlob()]

    assert module.should_run_yolo_for_current_frame() is False
    assert module.state.track_task_name == "red"

    candidates = module.build_object_candidates(OrbitBlobImage(), ())

    assert module.state.current_detection_source == "roi"
    assert candidates[0][:4] == ("red", 160.0, 210.0, 260.0)


def test_master_main_v2_object_task_config_keeps_only_filter_parameters() -> None:
    module = load_master_v2()

    assert module._object_task_config("red") == ("red", 3, 30, 70, 90, True)
    assert module._object_task_config("tennis") is None


def test_master_main_v2_object_task_config_accepts_legacy_threshold_layout() -> None:
    module = load_master_v2()
    module.OBJECT_TASKS = (("red", ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),)

    assert module._object_task_config("red") == ("red", 3, 30, 70, 90, True)


def test_master_main_v2_roi_tracking_uses_calibrated_object_threshold() -> None:
    module = load_master_v2()
    calibrated_threshold = (1, 2, 3, 4, 5, 6)
    stale_dynamic_threshold = (16, 51, 21, 84, -11, 52)
    module.OBJECT_TASKS = (("red", (calibrated_threshold,), 3, 30, 70, 90, True),)
    _seed_master_short_track(module)
    module.state.track_dynamic_threshold = stale_dynamic_threshold
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class Blob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 220.0

    img = DynamicThresholdRoiImage(calibrated_threshold, [Blob()])
    set_current_image(module, img)

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "roi"
    assert candidates[0][:4] == ("red", 160.0, 210.0, 220.0)
    assert img.find_blobs_calls[0][0] == calibrated_threshold


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


class DynamicThresholdCalibrationImage:
    def __init__(self, bbox, foreground, background, fragment=None):
        self.left, self.top, self.right, self.bottom = bbox
        self.foreground = foreground
        self.background = background
        self.fragment = fragment

    def width(self):
        return IMAGE_WIDTH

    def height(self):
        return IMAGE_HEIGHT

    def get_pixel(self, x, y):
        if self.fragment is not None:
            frag_left, frag_top, frag_right, frag_bottom, frag_color = self.fragment
            if frag_left <= x < frag_right and frag_top <= y < frag_bottom:
                return frag_color
        inner_left = self.left + 5
        inner_top = self.top + 5
        inner_right = self.right - 5
        inner_bottom = self.bottom - 5
        if inner_left <= x < inner_right and inner_top <= y < inner_bottom:
            return self.foreground
        return self.background


class DynamicThresholdRoiImage:
    def __init__(self, expected_threshold, blobs):
        self.expected_threshold = tuple(expected_threshold)
        self.blobs = list(blobs)
        self.find_blobs_calls = []

    def width(self):
        return IMAGE_WIDTH

    def height(self):
        return IMAGE_HEIGHT

    def find_blobs(
        self,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge,
        roi=None,
        margin=None,
    ):
        _ = (pixels_threshold, area_threshold, merge, margin)
        key = tuple(thresholds[0])
        self.find_blobs_calls.append((key, roi))
        if len(thresholds) == 1 and key == self.expected_threshold:
            return list(self.blobs)
        return []


def task_sync_frame(module, seq=12, context_id=7, state=None, target=None, arg=None):
    if state is None:
        state = int(module.State.SEARCH_OBJECT)
    if target is None:
        target = int(module.Target.OBJECT)
    if arg is None:
        arg = int(module.Task.SEARCH)
    return encode_frame(
        MODE_TCP,
        module.Topic.MASTER_VISION_TASK_SYNC,
        seq,
        module.encode_master_vision_task_sync_body(context_id, state, target, arg),
    )


def event_ack_frame(module, seq):
    return encode_frame(MODE_ACK, module.Topic.MASTER_VISION_EVENT_REPORT, seq, b"")


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
    target_x, target_y = module.build_search_target_point(module.Task.SEARCH)
    return module.build_observation(1, target_x + err_x, target_y + err_y, area)


def cache_yolo_candidates(module, img=None):
    if img is None:
        img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.state.current_yolo_candidates = tuple(module.yolo_detect(img))
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)
    return img


def run_frame(module, img):
    cache_yolo_candidates(module, img)
    module.process_task_frame(img)


def _seed_master_short_track(
    module,
    *,
    task_name="red",
    center_x=160.0,
    bottom_y=210.0,
    area=400.0,
    rect=(150, 10, 170, 30),
    vx=0.0,
    vy=0.0,
    frames_since_yolo=1,
    roi_failures=0,
):
    module.state.track_task_name = task_name
    module.state.track_center_x = float(center_x)
    module.state.track_bottom_y = float(bottom_y)
    module.state.track_area = float(area)
    module.state.track_rect = tuple(rect)
    module.state.track_velocity_x = float(vx)
    module.state.track_velocity_bottom_y = float(vy)
    module.state.track_source = "yolo"
    module.state.track_roi_success_frames = 0
    module.state.track_roi_failure_frames = int(roi_failures)
    module.state.track_frames_since_yolo = int(frames_since_yolo)
    module.state.track_dynamic_threshold = (16, 51, 21, 84, -11, 52)
    module.state.track_dynamic_threshold_rect = tuple(rect)


def test_master_main_v2_replies_task_sync_ack_and_records_task() -> None:
    module = load_master_v2()

    reply = module.handle_control_frame(task_sync_frame(module, seq=12, context_id=7))

    assert decode_frame(reply) == {
        "mode": MODE_ACK,
        "topic": module.Topic.MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 10,
    }
    assert module.state.current_task["context_id"] == 7
    assert module.state.current_task["arg"] == module.Task.SEARCH


def test_master_main_v2_runtime_state_uses_state_object() -> None:
    module = load_master_v2()

    assert hasattr(module, "state")
    assert hasattr(module, "State")
    assert hasattr(module, "Mode")
    assert hasattr(module, "Topic")
    assert hasattr(module, "Task")
    assert hasattr(module, "Event")
    assert hasattr(module, "Target")
    assert hasattr(module.state, "current_object_candidates")
    module.reset_runtime_state(next_event_seq=9)

    assert module.state.current_task is None
    assert module.state.next_event_seq == 9
    assert not hasattr(module, "CURRENT_TASK")
    assert module.State.SEARCH_OBJECT == 1
    assert module.Mode.UDP == 0x01
    assert module.Topic.MASTER_VISION_EVENT_REPORT == 0x12
    assert module.Task.SEARCH == 1
    assert module.Event.TARGET_FOUND == 6
    assert module.Target.OBJECT == 1
    assert not hasattr(module, "MODE_UDP")
    assert not hasattr(module, "TOPIC_MASTER_VISION_EVENT_REPORT")
    assert not hasattr(module, "STATE_SEARCH_OBJECT")
    assert not hasattr(module, "TaskConfig")
    assert not hasattr(module, "MASTER_SEARCH_TASK_CONFIG_ID")
    assert not hasattr(module, "EVENT_TARGET_FOUND")
    assert not hasattr(module, "TARGET_OBJECT")


def test_master_main_v2_process_uart_input_replies_ack_for_task_sync_frame() -> None:
    module = load_master_v2()
    uart = ReadWriteUART(task_sync_frame(module))
    module.state.uart_device = uart

    rx_buffer = module.process_uart_input(b"")

    assert rx_buffer == b""
    assert decode_frame(uart.writes[0]) == {
        "mode": MODE_ACK,
        "topic": module.Topic.MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 10,
    }
    assert module.state.current_task is not None


def test_master_main_v2_process_uart_input_resyncs_before_task_sync_frame() -> None:
    module = load_master_v2()
    uart = ReadWriteUART(b"\x02" + task_sync_frame(module))
    module.state.uart_device = uart

    rx_buffer = module.process_uart_input(b"")

    assert rx_buffer == b""
    assert decode_frame(uart.writes[0])["topic"] == module.Topic.MASTER_VISION_TASK_SYNC
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
    img = FakeImage()
    set_current_image(module, img)
    module.accept_observation(build_search_observation(module, 180.0), img)
    repeated_reply = module.handle_control_frame(task_sync_frame(module))
    module.accept_observation(build_search_observation(module, 180.0), img)

    assert decode_frame(reply)["seq"] == 12
    assert decode_frame(repeated_reply)["seq"] == 12
    assert module.state.stable_frame_count == 2
    assert module.state.pending_event is not None


def test_master_main_v2_non_new_context_does_not_override_active_task() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.handle_control_frame(
        task_sync_frame(module, seq=13, context_id=6, state=int(module.State.ORBITING), arg=9)
    )

    observation = build_search_observation(module, 180.0)

    assert module.state.current_task == {
        "context_id": 7,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    assert observation == (7, 0.0, 0.0, 180.0)


def test_master_main_v2_wrong_ack_does_not_clear_pending_event() -> None:
    module = load_master_v2()
    now_ms = [100]
    module.default_now_ms = lambda: now_ms[0]
    module.RELIABLE_RESEND_INTERVAL_MS = 20
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    img = FakeImage()
    set_current_image(module, img)
    module.accept_observation(build_search_observation(module, 180.0), img)

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
    img = FakeImage()
    set_current_image(module, img)
    module.accept_observation(build_search_observation(module, 180.0), img)

    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 0))
    now_ms[0] += 20
    second = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 1))

    assert second == first
    assert module.next_event_frame() is None


def test_master_main_v2_event_report_carries_current_dynamic_threshold() -> None:
    module = load_master_v2()
    threshold = (12, 80, -30, 40, -20, 60)
    module.state.track_dynamic_threshold = threshold

    module.create_pending_event(7, module.Event.TARGET_FOUND, 2)

    frame = decode_frame(module.next_event_frame())
    assert frame is not None
    event = decode_master_vision_event_report_body(frame["body"])
    assert event == {
        "context_id": 7,
        "event": module.Event.TARGET_FOUND,
        "value": 2,
        "threshold": threshold,
    }


def test_master_main_v2_creates_target_found_once_per_context() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    img = FakeImage()
    set_current_image(module, img)
    module.accept_observation(build_search_observation(module, 180.0), img)
    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 1))
    module.accept_observation(build_search_observation(module, 180.0), img)

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
        "event": module.Event.TARGET_FOUND,
        "value": 1,
    }


def test_master_main_v2_yolo_detect_filters_small_area_candidates() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.tf.detect = lambda net, img: [(0.25, 0.125, 0.28, 0.145, 1, 0.95)]
    img = FakeImage()
    set_current_image(module, img)

    assert module.yolo_detect(img) == []


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
    assert any("src=" in entry[2] for entry in img.strings)
    assert any("fail=" in entry[2] for entry in img.strings)
    assert any("fps t=" in entry[2] for entry in img.strings)
    assert any("roi ok=" in entry[2] for entry in img.strings)
    assert img.crosses
    assert img.flush_count == 1


def test_master_main_v2_process_task_frame_prints_object_debug_log_when_enabled() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.state.yolo_net = "fake-net"
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.state.uart_device = FakeUART()
    logs = []
    module.print = lambda *args: logs.append(" ".join(str(arg) for arg in args))
    img = FakeImage()

    module.state.current_detection_source = "yolo"
    run_frame(module, img)

    assert any("[master_v2][object]" in line for line in logs)
    assert any("src=yolo" in line for line in logs)
    assert any("cand=1" in line for line in logs)


def test_master_main_v2_debug_display_shows_finish_yellow_without_task_sync() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(img)
    img.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])

    run_frame(module, img)

    assert img.rectangles
    assert any("finish ratio=" in entry[2] for entry in img.strings)
    assert any("touch=" in entry[2] for entry in img.strings)
    assert any("stable=" in entry[2] for entry in img.strings)
    assert img.flush_count == 1
    assert uart.writes == []


def test_master_main_v2_debug_display_reports_finish_yellow_ratio_without_task_sync() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(img)
    img.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])
    set_current_image(module, img)

    module.process_task_frame(img)

    assert any("finish ratio=100.0" in entry[2] for entry in img.strings)
    assert any("touch=1" in entry[2] for entry in img.strings)
    assert img.flush_count == 1


def test_master_main_v2_build_observation_and_candidates_uses_cached_candidates_without_current_image() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    target_x, target_y = module.build_search_target_point(module.Task.SEARCH)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_yolo_candidates = ()
    module.state.current_object_candidates = (("red", target_x, target_y, 400.0, blob),)

    observation, best_blob, task_name, candidates = module.build_observation_and_candidates()

    assert observation == (7, 0.0, 0.0, 400.0)
    assert best_blob is blob
    assert task_name == "red"
    assert candidates == (("red", target_x, target_y, 400.0, blob),)


def test_master_main_v2_missing_target_outputs_configured_search_velocity() -> None:
    module = load_master_v2()
    module.state.current_image_height = IMAGE_HEIGHT
    velocity = module.build_search_velocity_from_observation((7, 0.0, 0.0, 0.0))

    assert velocity == (module.MASTER_MISSING_SEARCH_VX, module.MASTER_MISSING_SEARCH_VY)


def test_master_main_v2_search_velocity_deadzone_zeroes_each_axis() -> None:
    module = load_master_v2()
    observation = build_search_observation(
        module,
        150.0,
        err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX),
        err_y=float(module.MASTER_SEARCH_DEADZONE_Y_PX),
    )

    module.state.current_image_height = IMAGE_HEIGHT
    velocity = module.build_search_velocity_from_observation(observation)

    assert velocity == (0.0, 0.0)


def test_master_main_v2_search_velocity_applies_min_speed_outside_deadzone() -> None:
    module = load_master_v2()
    module.state.current_image_height = IMAGE_HEIGHT
    positive = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1,
            err_y=float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1,
        )
    )
    negative = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=-(float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1),
            err_y=-(float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1),
        )
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
    module.state.current_image_height = IMAGE_HEIGHT
    positive = module.build_search_velocity_from_observation((7, 999.0, 999.0, 300.0))
    negative = module.build_search_velocity_from_observation((7, -999.0, -999.0, 300.0))

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

    module.state.current_image_height = IMAGE_HEIGHT
    velocity = module.build_search_velocity_from_observation((7, 100.0, -100.0, 300.0))

    assert velocity == pytest.approx((5.0, 2.0833333333333335))


def test_master_main_v2_reference_frame_interval_is_derived_from_fps() -> None:
    module = load_master_v2()
    module.VISION_REFERENCE_FPS = 25

    assert module.reference_frame_interval_ms() == pytest.approx(40.0)


def test_master_main_v2_slow_frame_interval_reduces_search_velocity() -> None:
    module = load_master_v2()
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms()
    module.state.current_image_height = IMAGE_HEIGHT
    reference_velocity = module.build_search_velocity_from_observation((7, 100.0, -100.0, 300.0))
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms() * 2
    slow_velocity = module.build_search_velocity_from_observation((7, 100.0, -100.0, 300.0))

    assert abs(slow_velocity[0]) < abs(reference_velocity[0])
    assert abs(slow_velocity[1]) < abs(reference_velocity[1])
    assert slow_velocity == pytest.approx((reference_velocity[0] * 0.5, reference_velocity[1] * 0.5))


def test_master_main_v2_search_y_velocity_decreases_when_target_gets_closer() -> None:
    module = load_master_v2()
    module.state.current_image_height = IMAGE_HEIGHT
    far_velocity = module.build_search_velocity_from_observation((7, 0.0, -200.0, 1000.0))
    close_velocity = module.build_search_velocity_from_observation((7, 0.0, -100.0, 1000.0))

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
            state=int(module.State.ORBITING),
            arg=int(module.Task.ORBIT),
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
            state=int(module.State.SEARCH_OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()

    run_frame(module, img)
    run_frame(module, img)

    assert latest_event(type("U", (), {"writes": [uart.writes[1]]})()) == {
        "context_id": 9,
        "event": module.Event.ALIGNED,
        "value": 400,
    }


def test_master_main_v2_candidate_selection_uses_configured_target_point() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.Task.SEARCH)
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
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT)
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
            state=int(module.State.SEARCH_OBJECT),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[3] == pytest.approx(400.0)


def test_master_main_v2_finish_accepts_x_outside_when_bottom_hits_target_window() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
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
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[3] == pytest.approx(400.0)


def test_master_main_v2_finish_uses_flipped_candidate_bottom_for_target_window() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
    module.tf.detect = lambda net, img: [
        pixel_detection(
            target_x - 10.0,
            IMAGE_HEIGHT - target_y,
            target_x + 10.0,
            IMAGE_HEIGHT - target_y + 20,
        )
    ]
    module.handle_control_frame(
        task_sync_frame(
            module,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation[2] == pytest.approx(0.0)


def test_master_main_v2_finish_prefers_largest_area_after_bottom_filter() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
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
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
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
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
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
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
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
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    fixed_roi = module.build_finish_task_fixed_object_roi(FakeImage())
    touch_areas = {tuple(fixed_roi): max(1, int(fixed_roi[2]) * int(fixed_roi[3]))}

    run_frame(module, FakeImage(yellow_area_by_roi=touch_areas))
    run_frame(module, FakeImage(yellow_area_by_roi={}))
    run_frame(module, FakeImage(yellow_area_by_roi={}))

    assert latest_event(uart) == {
        "context_id": 11,
        "event": module.Event.ARRIVED,
        "value": 0,
    }


def test_master_main_v2_finish_observation_enters_contact_seen_and_pending_event() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.FINISH_HOOK_STABLE_FRAMES = 2
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=21,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()
    module.accept_observation((21, 0.0, 0.0, 1.0), img, event_value=100.0)
    assert module.state.finish_contact_seen is True
    assert module.state.stable_frame_count == 0
    assert module.state.pending_event is None

    module.accept_observation((21, 0.0, 0.0, 1.0), img, event_value=0.0)
    module.accept_observation((21, 0.0, 0.0, 1.0), img, event_value=0.0)

    assert module.state.pending_event is not None


def test_master_main_v2_finish_reports_arrived_after_contact_release_without_object_candidate() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.FINISH_HOOK_STABLE_FRAMES = 2
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=24,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    fixed_roi = module.build_finish_task_fixed_object_roi(FakeImage())
    touch_areas = {tuple(fixed_roi): max(1, int(fixed_roi[2]) * int(fixed_roi[3]))}
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
    module.tf.detect = lambda net, img: [
        pixel_detection(
            target_x - 10.0,
            IMAGE_HEIGHT - target_y,
            target_x + 10.0,
            IMAGE_HEIGHT - target_y + 20.0,
        )
    ]

    run_frame(module, FakeImage(yellow_area_by_roi=touch_areas))
    module.tf.detect = lambda net, img: []
    run_frame(module, FakeImage(yellow_area_by_roi={}))
    run_frame(module, FakeImage(yellow_area_by_roi={}))
    assert latest_velocity(uart) == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }
    run_frame(module, FakeImage(yellow_area_by_roi={}))

    assert latest_event(uart) == {
        "context_id": 24,
        "event": module.Event.ARRIVED,
        "value": 0,
    }


def test_master_main_v2_finish_debug_touch_uses_same_ratio_as_finish_acceptance() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.state.yolo_net = "fake-net"
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=22,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )
    module.state.uart_device = FakeUART()
    img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(img)
    img.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])
    module.state.current_detection_source = "roi"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_yolo_candidates = (("red", target_x, target_y, 400.0, blob),)
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)

    module.process_task_frame(img)

    assert module.state.finish_contact_seen is True
    assert any("touch=1" in entry[2] for entry in img.strings)


def test_master_main_v2_finish_process_task_frame_consumes_touch_into_finish_state() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.FINISH_HOOK_STABLE_FRAMES = 2
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=23,
            state=int(module.State.TRANSPORT_OBJECT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.TRANSPORT_FINISH),
        )
    )
    module.state.uart_device = FakeUART()
    img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(img)
    img.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])
    module.state.current_detection_source = "roi"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT_FINISH)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_yolo_candidates = (("red", target_x, target_y, 400.0, blob),)
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)

    module.process_task_frame(img)

    assert module.state.finish_contact_seen is True
    assert module.state.pending_event is None
    assert module.state.stable_frame_count == 0


def test_master_main_v2_debug_finish_without_task_sync_emits_local_pending_event_after_release() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.FINISH_HOOK_STABLE_FRAMES = 2
    module.state.uart_device = FakeUART()
    touch_img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(touch_img)
    touch_img.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])

    module.process_task_frame(touch_img)
    module.process_task_frame(FakeImage())
    module.process_task_frame(FakeImage())

    assert module.state.finish_contact_seen is True
    assert module.state.pending_event is not None

def test_master_main_v2_finish_yellow_ratio_uses_fixed_object_roi() -> None:
    module = load_master_v2()
    img = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(img)
    yellow_area = int(fixed_roi[2]) * int(fixed_roi[3])

    class FarObjectBlob:
        def rect(self):
            return (0, 0, 20, 20)

    img.yellow_area_by_roi[tuple(fixed_roi)] = yellow_area

    assert fixed_roi == (80, 0, 160, 80)
    assert module.build_finish_task_yellow_ratio_percent(img, FarObjectBlob()) == pytest.approx(100.0)


def test_master_main_v2_return_retreat_reports_line_aligned() -> None:
    module = load_master_v2()
    module.state.yolo_net = "fake-net"
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=13,
            state=int(module.State.RETURN_GARAGE_RETREAT),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.RETURN_GARAGE_LINE),
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
        "event": module.Event.RETURN_LINE_ALIGNED,
        "value": 160,
    }


def test_master_main_v2_return_line_y_uses_center_columns_bounds_average() -> None:
    module = load_master_v2()
    img = FakeImage(pixels=build_return_line_band_pixels(180, 200))
    module.state.current_image = img
    module.state.current_image_width = IMAGE_WIDTH
    module.state.current_image_height = IMAGE_HEIGHT

    line_y = module.build_return_line_y_from_image(img)

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
    img = FakeImage(pixels=build_return_line_band_pixels(80, 200))
    module.state.current_image = img
    module.state.current_image_width = IMAGE_WIDTH
    module.state.current_image_height = IMAGE_HEIGHT

    line_y = module.build_return_line_y_from_image(img)

    assert line_y == pytest.approx(185.0)


def test_master_main_v2_return_line_keeps_previous_when_horizontal_connected_is_too_short() -> None:
    module = load_master_v2()
    module.RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50
    img = FakeImage(pixels=build_return_line_band_pixels(180, 200, left=150, right=170))
    module.state.current_image = img
    module.state.current_image_width = IMAGE_WIDTH
    module.state.current_image_height = IMAGE_HEIGHT

    line_y = module.build_return_line_y_from_image(img, 188.0)

    assert line_y == pytest.approx(188.0)


def test_master_main_v2_return_line_runtime_does_not_use_blob_detection() -> None:
    module = load_master_v2()
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=7,
            state=int(module.State.RETURN_GARAGE_LINE),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.RETURN_GARAGE_LINE),
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
            state=int(module.State.RETURN_GARAGE_LINE),
            target=int(module.Target.EDGE_LINE),
            arg=int(module.Task.RETURN_GARAGE_LINE),
        )
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage(pixels=build_return_line_pixels(160))

    for _ in range(7):
        run_frame(module, img)

    assert latest_event(uart) == {
        "context_id": 15,
        "event": module.Event.RETURN_GARAGE_FINISHED,
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

    assert decode_frame(uart.writes[-1])["topic"] == module.Topic.MASTER_VISION_EVENT_REPORT
    module.handle_control_frame(event_ack_frame(module, 1))
    run_frame(module, img)

    assert decode_frame(uart.writes[-1])["topic"] == module.Topic.LOCAL_VISION_VELOCITY


def test_master_main_v2_run_applies_lens_correction_and_shows_finish_yellow_debug_without_task_sync() -> None:
    """主车调试模式下没有任务同步时显示搬运结束黄线判定结果."""

    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    logs = []
    module.print = lambda *args: logs.append(" ".join(str(arg) for arg in args))

    class StopLoop(Exception):
        pass

    class SnapshotImage(FakeImage):
        def __init__(self, pixels=None):
            super().__init__(pixels=pixels)
            self.lens_corr_called = False

        def lens_corr(self, strength, zoom):
            super().lens_corr(strength, zoom)
            _ = strength, zoom
            self.lens_corr_called = True
            return self

        def flush(self):
            super().flush()
            raise StopLoop()

    image = SnapshotImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(image)
    image.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])
    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = lambda rx_buffer: rx_buffer

    def fake_yolo_detect(img):
        _ = img
        raise AssertionError("上电黄线调试不应调用 yolo_detect")

    module.yolo_detect = fake_yolo_detect
    module.write_reliable_line = lambda _frame_bytes: (_ for _ in ()).throw(
        AssertionError("上电调试预览不应发送底盘暂停控制")
    )

    with pytest.raises(StopLoop):
        module.run()

    assert module.state.current_second_total_frames == 1
    assert module.state.current_second_yolo_frames == 0
    assert module.state.current_image is image
    assert image.lens_corr_called is True
    assert tuple(module.state.current_yolo_candidates) == ()
    assert tuple(module.state.current_object_candidates) == ()
    assert image.flush_count == 1
    assert any("finish ratio=" in entry[2] for entry in image.strings)
    assert any("touch=" in entry[2] for entry in image.strings)
    assert any("stable=" in entry[2] for entry in image.strings)
    assert any("[master_v2][boot]" in line for line in logs)
    assert any("debug=1" in line for line in logs)


def test_master_main_v2_finish_debug_shows_pending_event_when_contact_stabilizes() -> None:
    module = load_master_v2()
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.TRANSPORT_OBJECT,
        "target": module.Target.EDGE_LINE,
        "arg": module.Task.TRANSPORT_FINISH,
    }
    module.state.finish_contact_seen = True
    module.state.stable_frame_count = int(module.FINISH_HOOK_STABLE_FRAMES) - 1
    module.state.pending_event = {"event": module.Event.ARRIVED}

    class ImageWithYellow(FakeImage):
        def __init__(self):
            super().__init__()
            roi = module.build_finish_task_fixed_object_roi(self)
            self.yellow_area_by_roi[tuple(roi)] = int(roi[2]) * int(roi[3])

    image = ImageWithYellow()
    module.draw_finish_task_debug(image, None, 100.0)

    assert any("event=1" in entry[2] for entry in image.strings)
    assert any("touch=1" in entry[2] for entry in image.strings)


def test_master_main_v2_run_requests_reliable_pause_before_yolo() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True

    class StopLoop(Exception):
        pass

    image = FakeImage()

    class Sensor:
        def snapshot(self):
            return image

    uart = FakeUART()
    module.sensor = Sensor()
    module.init_uart = lambda: uart
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)

    def prime_search_task(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": module.State.SEARCH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.Task.SEARCH,
        }
        return rx_buffer

    module.process_uart_input = prime_search_task
    module.yolo_detect = lambda img: (_ for _ in ()).throw(AssertionError("暂停确认前不应运行 YOLO"))

    def stop_after_reliable(frame_bytes):
        uart.write(frame_bytes)
        raise StopLoop()

    module.write_reliable_line = stop_after_reliable

    with pytest.raises(StopLoop):
        module.run()

    frame = decode_frame(uart.writes[-1])
    assert frame is not None
    assert frame["mode"] == MODE_TCP
    assert frame["topic"] == module.Topic.LOCAL_VISION_CONTROL
    assert frame["body"][0] == module.LocalVisionControl.PAUSE


def test_master_main_v2_yolo_frame_suppresses_velocity_and_requests_resume() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    start_run_as_local_vision_paused(module)

    class StopLoop(Exception):
        pass

    class FakeBlob:
        def rect(self):
            return (150, 30, 20, 20)

    image = FakeImage()

    class Sensor:
        def snapshot(self):
            return image

    uart = FakeUART()
    module.sensor = Sensor()
    module.init_uart = lambda: uart
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)

    def prime_search_task(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": module.State.SEARCH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.Task.SEARCH,
        }
        return rx_buffer

    module.process_uart_input = prime_search_task
    module.yolo_detect = lambda img: [("red", 160.0, 210.0, 400.0, FakeBlob())]

    def fail_data_write(frame_bytes):
        _ = frame_bytes
        raise AssertionError("YOLO 暂停帧不应发送速度")

    def stop_after_resume(frame_bytes):
        uart.write(frame_bytes)
        raise StopLoop()

    module.write_data_line = fail_data_write
    module.write_reliable_line = stop_after_resume

    with pytest.raises(StopLoop):
        module.run()

    frame = decode_frame(uart.writes[-1])
    assert frame is not None
    assert frame["mode"] == MODE_TCP
    assert frame["topic"] == module.Topic.LOCAL_VISION_CONTROL
    assert frame["body"][0] == module.LocalVisionControl.RESUME
    assert module.state.current_detection_source == "yolo"


def test_master_main_v2_finish_debug_ignores_cached_object_tracking_without_task_sync() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    image = FakeImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(image)
    image.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])
    set_current_image(module, image)
    _seed_master_short_track(module)
    yolo_blob = module.YoloDetectionBlob(150.0, 30.0, 170.0, 50.0, 1, 0.95)
    module.state.current_detection_source = "yolo"
    module.state.current_yolo_candidates = (("red", 160.0, 210.0, 400.0, yolo_blob),)
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)

    module.process_task_frame(image)

    assert tuple(module.state.current_yolo_candidates[:1])[0][:4] == ("red", 160.0, 210.0, 400.0)
    assert tuple(module.state.current_object_candidates[:1])[0][:4] == ("red", 160.0, 210.0, 400.0)
    assert module.state.track_source == "yolo"
    assert any("finish ratio=" in entry[2] for entry in image.strings)
    assert image.flush_count == 1


def test_master_main_v2_run_shows_finish_debug_when_blob_tracking_is_active() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True

    class StopLoop(Exception):
        pass

    class SnapshotImage(FakeImage):
        def flush(self):
            super().flush()
            raise StopLoop()

    image = SnapshotImage()
    fixed_roi = module.build_finish_task_fixed_object_roi(image)
    image.yellow_area_by_roi[tuple(fixed_roi)] = int(fixed_roi[2]) * int(fixed_roi[3])

    class Sensor:
        def snapshot(self):
            return image

    def prime_preview_tracking(rx_buffer):
        _seed_master_short_track(module)
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = prime_preview_tracking
    yolo_call_count = 0

    def fake_yolo_detect(img):
        nonlocal yolo_call_count
        assert img is image
        yolo_call_count += 1
        return []

    module.yolo_detect = fake_yolo_detect

    with pytest.raises(StopLoop):
        module.run()

    assert tuple(module.state.current_yolo_candidates) == ()
    assert tuple(module.state.current_object_candidates) == ()
    assert any("finish ratio=" in entry[2] for entry in image.strings)
    assert yolo_call_count == 0


def test_master_main_v2_run_skips_yolo_in_return_line_task() -> None:
    module = load_master_v2()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    logs = []
    module.print = lambda *args: logs.append(" ".join(str(arg) for arg in args))

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

    class Sensor:
        def snapshot(self):
            return image

    def switch_to_return_line(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": module.State.RETURN_GARAGE_RETREAT,
            "target": module.Target.EDGE_LINE,
            "arg": module.Task.RETURN_GARAGE_LINE,
        }
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = switch_to_return_line

    def fake_yolo_detect(img):
        _ = img
        raise AssertionError("回库黄线任务不应调用 yolo_detect")

    def stop_after_frame(current_img):
        assert current_img is image
        assert image.lens_corr_called is True
        assert tuple(module.state.current_yolo_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()

    assert any("[master_v2][skip]" in line and "reason=return_line" in line for line in logs)


def test_master_main_v2_run_skips_yolo_when_blob_tracking_is_active() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3

    class StopLoop(Exception):
        pass

    class FakeBlob:
        def rect(self):
            return (150, 10, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    class SnapshotImage(FakeImage):
        def __init__(self):
            super().__init__()
            self.lens_corr_called = False

        def lens_corr(self, strength, zoom):
            super().lens_corr(strength, zoom)
            _ = strength, zoom
            self.lens_corr_called = True
            return self

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=None):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [FakeBlob()]

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    def prime_object_tracking(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": module.State.SEARCH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.Task.SEARCH,
        }
        _seed_master_short_track(module)
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = prime_object_tracking
    yolo_call_count = 0

    def fake_yolo_detect(img):
        nonlocal yolo_call_count
        assert img is image
        yolo_call_count += 1
        return []

    def stop_after_write(frame_bytes):
        _ = frame_bytes
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.write_data_line = stop_after_write

    with pytest.raises(StopLoop):
        module.run()

    assert tuple(module.state.current_yolo_candidates) == ()
    assert tuple(module.state.current_object_candidates) == (
        ("red", 160.0, 230.0, 400.0, module.state.current_object_candidates[0][4]),
    )
    assert module.state.current_detection_source == "roi"
    assert yolo_call_count == 0


def test_master_main_v2_run_falls_back_to_yolo_after_roi_failure() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 1
    start_run_as_local_vision_paused(module)

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

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=None):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return []

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    def prime_object_tracking(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": module.State.SEARCH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.Task.SEARCH,
        }
        _seed_master_short_track(module)
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = prime_object_tracking

    class FakeBlob:
        def rect(self):
            return (150, 10, 20, 20)

    yolo_blob = FakeBlob()
    module.yolo_detect = lambda img: [("red", 160.0, 210.0, 400.0, yolo_blob)]

    def stop_after_resume(frame_bytes):
        frame = decode_frame(frame_bytes)
        assert frame is not None
        assert frame["topic"] == module.Topic.LOCAL_VISION_CONTROL
        assert frame["body"][0] == module.LocalVisionControl.RESUME
        raise StopLoop()

    module.write_reliable_line = stop_after_resume

    with pytest.raises(StopLoop):
        module.run()

    assert tuple(candidate[:4] for candidate in module.state.current_yolo_candidates) == (
        ("red", 160.0, 210.0, 400.0),
    )
    assert tuple(candidate[:4] for candidate in module.state.current_object_candidates) == (
        ("red", 160.0, 210.0, 400.0),
    )
    assert module.state.current_object_candidates[0][4].rect() == (150, 30, 20, 20)


def test_master_main_v2_build_object_candidates_keeps_predicted_target_before_yolo_fallback() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(module, center_x=158.0, bottom_y=208.0, vx=2.0, vy=3.0)

    class PredictImage(FakeImage):
        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=None):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return []

    img = PredictImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert candidates[0][:4] == ("red", 160.0, 211.0, 400.0)


def test_master_main_v2_predict_frame_does_not_create_event() -> None:
    module = load_master_v2()
    module.handle_control_frame(task_sync_frame(module, context_id=12))

    class FakeBlob:
        def rect(self):
            return (150, 10, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage()
    set_current_image(module, img)
    module.state.current_yolo_candidates = ()
    module.state.current_object_candidates = ()

    def fake_build_object_candidates(current_img, yolo_candidates):
        assert current_img is img
        assert yolo_candidates == ()
        module.state.current_detection_source = "predict"
        return (("red", 160.0, 210.0, 400.0, FakeBlob()),)

    module.build_object_candidates = fake_build_object_candidates
    module.state.current_object_candidates = tuple(module.build_object_candidates(img, ()))

    module.process_task_frame(img)

    assert module.state.pending_event is None


def test_master_main_v2_yolo_relocation_prefers_candidate_near_tracked_target() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(module, center_x=100.0, bottom_y=210.0, frames_since_yolo=3)

    class FakeImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

    class FakeBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

    yolo_candidates = [
        ("red", 160.0, 210.0, 400.0, FakeBlob(150, 10, 20, 20)),
        ("red", 100.0, 210.0, 400.0, FakeBlob(90, 10, 20, 20)),
    ]
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, yolo_candidates)

    assert candidates[0][:4] == ("red", 100.0, 210.0, 400.0)


def test_master_main_v2_yolo_relocation_limits_large_position_jump() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(
        module,
        center_x=100.0,
        bottom_y=210.0,
        rect=(90, 10, 110, 30),
        frames_since_yolo=3,
    )

    class FakeImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (220, 10, 20, 20)

    yolo_candidates = [
        ("red", 230.0, 210.0, 400.0, FakeBlob()),
    ]
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, yolo_candidates)

    assert candidates[0][:4] == ("red", 130.0, 210.0, 400.0)


def test_master_main_v2_yolo_relocation_limits_large_area_jump() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(
        module,
        area=400.0,
        rect=(150, 10, 170, 30),
        frames_since_yolo=3,
    )

    class FakeImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (140, 0, 40, 40)

    yolo_candidates = [
        ("red", 160.0, 210.0, 1600.0, FakeBlob()),
    ]
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, yolo_candidates)

    assert candidates[0][:4] == ("red", 160.0, 210.0, 800.0)


def test_master_main_v2_track_state_records_object_id_and_confidence() -> None:
    module = load_master_v2()
    module.state.current_image_height = IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (150, 10, 20, 20)

    module.remember_object_tracking("red", FakeBlob(), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_object_id == 1
    assert module.state.track_confidence == 80


def test_master_main_v2_debug_counters_roll_per_second() -> None:
    module = load_master_v2()

    module.state.record_frame_source("yolo", now_ms=0)
    module.state.record_roi_attempt(True, now_ms=100)
    module.state.record_frame_source("roi", now_ms=100)
    module.state.record_roi_attempt(False, now_ms=100)
    module.state.record_roi_fallback(now_ms=100)
    module.state.record_frame_source("predict", now_ms=1100)

    assert module.state.last_second_total_frames == 2
    assert module.state.last_second_yolo_frames == 1
    assert module.state.last_second_roi_frames == 1
    assert module.state.last_second_predict_frames == 0
    assert module.state.last_second_roi_attempt_frames == 2
    assert module.state.last_second_roi_success_frames == 1
    assert module.state.last_second_roi_fallbacks == 1
    assert module.state.current_second_total_frames == 1


def test_master_main_v2_rejects_blob_candidate_outside_tracking_window() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(module)

    class FarBlob:
        def rect(self):
            return (220, 10, 20, 20)

        def cx(self):
            return 230.0

        def area(self):
            return 400.0

    class FakeImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=None):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [FarBlob()]

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert module.state.track_failure_reason == module.TrackFailureReason.OUT_OF_WINDOW
    assert candidates[0][:4] == ("red", 160.0, 210.0, 400.0)


def test_master_main_v2_rejects_blob_candidate_with_large_area_jump() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    _seed_master_short_track(module, area=400.0)

    class LargeBlob:
        def rect(self):
            return (140, 30, 40, 40)

        def cx(self):
            return 160.0

        def area(self):
            return 1600.0

    class FakeImage:
        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=None):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [LargeBlob()]

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert module.state.track_failure_reason == module.TrackFailureReason.AREA_JUMP
    assert candidates[0][:4] == ("red", 160.0, 210.0, 400.0)


def test_master_main_v2_builds_dynamic_threshold_from_yolo_and_uses_it_for_roi() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, calibration_img)
    module.remember_object_tracking("red", YoloBlob(), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold is not None

    class Blob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 180.0

    threshold = module.state.track_dynamic_threshold
    img = DynamicThresholdRoiImage(threshold, [Blob()])
    set_current_image(module, img)

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "roi"
    assert candidates[0][:4] == ("red", 160.0, 210.0, 180.0)
    assert img.find_blobs_calls[0][0] == tuple(threshold)


def test_master_main_v2_yolo_calibration_uses_foreground_area_for_tracking() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, calibration_img)
    module.remember_object_tracking("red", YoloBlob(), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold is not None
    assert module.state.track_area < 300.0

    class Blob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 180.0

    threshold = module.state.track_dynamic_threshold
    img = DynamicThresholdRoiImage(threshold, [Blob()])
    set_current_image(module, img)

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "roi"
    assert module.state.track_failure_reason == module.TrackFailureReason.NONE
    assert candidates[0][:4] == ("red", 160.0, 210.0, 180.0)


def test_master_main_v2_reuses_dynamic_threshold_when_center_stays_stable() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, first_img)
    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation

    second_img = DynamicThresholdCalibrationImage(
        (152, 32, 170, 50),
        (30, 40, 20),
        (60, 10, 10),
    )
    set_current_image(module, second_img)
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: (_ for _ in ()).throw(
        AssertionError("中心采样稳定时不应重算阈值")
    )

    module.remember_object_tracking("red", YoloBlob(152, 32, 18, 18), 161.0, 208.0, 324.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation


def test_master_main_v2_refreshes_dynamic_threshold_after_consecutive_unhealthy_yolo_frames() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, first_img)
    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation
    new_threshold = (85, 95, -5, 5, -5, 5)
    build_calls = []

    def fake_build_dynamic_threshold(_img, _blob):
        build_calls.append(1)
        return new_threshold

    module._build_dynamic_threshold_for_blob = fake_build_dynamic_threshold

    second_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (90, 0, 0),
        (80, 0, 0),
    )
    set_current_image(module, second_img)

    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_dynamic_threshold_health_failures == 1
    assert module.state.track_pending_dynamic_threshold is None
    assert build_calls == []

    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_pending_dynamic_threshold == new_threshold
    assert module.state.track_pending_dynamic_threshold_ok_frames == 1
    assert len(build_calls) == 1

    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == new_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation + 1
    assert module.state.track_dynamic_threshold_health_failures == 0


def test_master_main_v2_keeps_dynamic_threshold_in_non_search_state() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.DYNAMIC_THRESHOLD_REFRESH_FAILURE_FRAMES = 1
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.ORBITING,
        "target": module.Target.OBJECT,
        "arg": module.Task.ORBIT,
    }

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, first_img)
    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation
    second_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (90, 0, 0),
        (80, 0, 0),
    )
    set_current_image(module, second_img)
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: (_ for _ in ()).throw(
        AssertionError("非寻找态不应重复计算动态阈值")
    )

    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_dynamic_threshold_health_failures == 0
    assert module.state.track_pending_dynamic_threshold is None


def test_master_main_v2_recomputes_dynamic_threshold_after_track_reset() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
    )
    set_current_image(module, first_img)
    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    new_threshold = (85, 95, -5, 5, -5, 5)
    module.state.clear_track()
    second_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (90, 0, 0),
        (80, 0, 0),
    )
    set_current_image(module, second_img)
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: new_threshold

    module.remember_object_tracking("red", YoloBlob(150, 30, 20, 20), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == new_threshold
    assert module.state.track_dynamic_threshold_generation == 1


def test_master_main_v2_dynamic_threshold_keeps_center_connected_component_only() -> None:
    module = load_master_v2()
    module.image.rgb_to_lab = lambda pixel: pixel

    class YoloBlob:
        def rect(self):
            return (150, 30, 20, 20)

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 30, 170, 50),
        (30, 40, 20),
        (80, 0, 0),
        fragment=(150, 30, 155, 35, (32, 75, 60)),
    )
    set_current_image(module, calibration_img)

    threshold = module._build_dynamic_threshold_for_blob(calibration_img, YoloBlob())

    assert threshold is not None
    assert threshold[3] < 75
    assert threshold[5] < 60


def test_master_main_v2_calibrated_threshold_enables_roi_tracking_without_pixel_sampling() -> None:
    module = load_master_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class YoloBlob:
        def rect(self):
            return (150, 30, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    img = FakeImage()
    set_current_image(module, img)
    module.remember_object_tracking("red", YoloBlob(), 160.0, 210.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == module.OBJECT_TASKS[0][1][0]
    assert module.should_use_blob_tracking() is True
