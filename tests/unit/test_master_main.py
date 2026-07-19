"""主车视觉默认入口行为测试."""

# pyright: reportAttributeAccessIssue=false, reportOptionalSubscript=false

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
MASTER_LOG_PREFIX = "[master_" + "v" + "2]"


def load_master_main():
    module = load_role_entry_module("master", "run.py", "vision_master_test_module")
    module.reset_runtime_state()
    module.MASTER_DEBUG_DISPLAY_ENABLED = False
    module.OBJECT_STABLE_FRAMES = 1
    module.OBJECT_SELECTION_STABLE_FRAMES = 1
    module.state.yolo_net = "fake-net"
    module.prepare_runtime = module.init_status_lights
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


def test_master_main_default_configuration_reports_target_found_after_configured_observations() -> None:
    module = load_role_entry_module(
        "master",
        "run.py",
        "vision_master_default_stable_frames_test_module",
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


def test_master_main_process_task_and_velocity_helpers_drop_frame_size_parameters() -> None:
    module = load_master_main()

    assert tuple(inspect.signature(module.process_task_frame).parameters) == ("img",)
    assert tuple(inspect.signature(module.yolo_detect).parameters) == ("img",)
    assert tuple(inspect.signature(module.normalize_bbox_for_protocol).parameters) == (
        "left",
        "top",
        "right",
        "bottom",
    )


def test_master_capture_image_rotates_frame_once() -> None:
    module = load_master_main()

    class Image:
        def __init__(self):
            self.replace_calls = []

        def replace(self, **kwargs):
            self.replace_calls.append(kwargs)
            return self

    image = Image()
    module.sensor.snapshot = lambda: image

    assert module.capture_image() is image
    assert image.replace_calls == [
        {"vflip": True, "hmirror": True, "transpose": False}
    ]
    assert module.normalize_bbox_for_protocol(10, 20, 30, 40) == (10, 20, 30, 40)
    assert tuple(inspect.signature(module.build_search_velocity_from_observation).parameters) == (
        "observation",
    )
    assert tuple(
        inspect.signature(module.build_orbit_correction_velocity_from_observation).parameters
    ) == ("observation",)
    assert tuple(inspect.signature(module.draw_protocol_target_point_debug).parameters) == (
        "img",
        "target_x",
        "target_y",
    )
    assert tuple(inspect.signature(module.draw_search_preview_debug).parameters) == ("img", "candidates")
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


@pytest.mark.parametrize(
    ("task_state", "task_arg"),
    (
        (1, 1),
        (1, 2),
        (2, 4),
    ),
)
def test_master_main_yolo_mode_runs_every_object_stage_without_pause(
    task_state,
    task_arg,
) -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = True

    class StopLoop(Exception):
        pass

    class SnapshotImage(FakeImage):
        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("YOLO 模式不应调用物体色块识别")

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.tf.load = lambda _path, load_to_fb=False: "fake-net"

    def prime_object_task(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": task_state,
            "target": module.Target.OBJECT,
            "arg": task_arg,
        }
        return rx_buffer

    yolo_calls = []
    module.process_uart_input = prime_object_task
    module.yolo_detect = lambda current_img: yolo_calls.append(current_img) or ()
    module.write_reliable_line = lambda _frame: (_ for _ in ()).throw(
        AssertionError("YOLO 推理不应发送暂停或恢复控制")
    )
    module.write_data_line = lambda _frame: (_ for _ in ()).throw(StopLoop())

    with pytest.raises(StopLoop):
        module.run()

    assert yolo_calls == [image]
    assert tuple(module.state.current_object_candidates) == ()


@pytest.mark.parametrize(
    ("task_state", "task_arg"),
    (
        (1, 1),
        (1, 2),
        (2, 4),
    ),
)
def test_master_main_blob_mode_runs_every_object_stage(task_state, task_arg) -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = False
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL

    class StopLoop(Exception):
        pass

    class SnapshotImage(FakeImage):
        def __init__(self):
            super().__init__()
            self.object_blob_calls = 0

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            self.object_blob_calls += 1
            return []

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)

    def prime_object_task(rx_buffer):
        module.state.current_task = {
            "context_id": 12,
            "state": task_state,
            "target": module.Target.OBJECT,
            "arg": task_arg,
        }
        return rx_buffer

    module.process_uart_input = prime_object_task
    module.yolo_detect = lambda _img: (_ for _ in ()).throw(
        AssertionError("色块模式不应调用 YOLO")
    )
    module.write_data_line = lambda _frame: (_ for _ in ()).throw(StopLoop())

    with pytest.raises(StopLoop):
        module.run()

    assert image.object_blob_calls == 1
    assert tuple(module.state.current_object_candidates) == ()


def test_master_main_yolo_mode_keeps_category_without_tracking_box() -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }
    blob = module.YoloDetectionBlob(150.0, 20.0, 170.0, 40.0, 1, 0.95)
    module.state.current_detection_source = "yolo"
    module.state.current_object_candidates = (("red", 160.0, 220.0, 400.0, blob),)
    module.write_data_line = lambda _frame: None

    module.process_task_frame(FakeImage())

    assert module.state.object_task_name == "red"
    assert not hasattr(module.state, "track_rect")
    assert not hasattr(module.state, "track_dynamic_threshold")


def test_master_main_search_locks_category_after_three_stable_frames() -> None:
    module = load_master_main()
    module.OBJECT_STABLE_FRAMES = 3
    module.OBJECT_SELECTION_STABLE_FRAMES = 3
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.write_data_line = lambda _frame: None
    red = object_candidate(module, "red", 150, 190, 170, 210)
    brown = object_candidate(module, "brown", 150, 190, 170, 210)

    for candidate in (red, red, brown, brown):
        module.state.current_object_candidates = (candidate,)
        module.process_task_frame(FakeImage())
        assert module.state.object_task_name is None
        assert module.state.pending_event is None

    module.state.current_object_candidates = (brown,)
    module.process_task_frame(FakeImage())

    assert module.state.object_task_name == "brown"
    assert module.state.pending_event is not None


def test_master_main_new_search_task_clears_previous_yolo_class_lock() -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.handle_control_frame(task_sync_frame(module, context_id=1))
    red_blob = module.YoloDetectionBlob(0.0, 0.0, 20.0, 20.0, 1, 0.95)
    module.state.current_object_candidates = (("red", 10.0, 20.0, 400.0, red_blob),)
    module.write_data_line = lambda _frame: None

    module.process_task_frame(FakeImage())
    module.handle_control_frame(task_sync_frame(module, seq=13, context_id=2))

    green_blob = module.YoloDetectionBlob(150.0, 190.0, 170.0, 210.0, 2, 0.95)
    candidates = module.build_object_candidates(
        FakeImage(),
        (("green", 160.0, 210.0, 400.0, green_blob),),
    )

    assert module.state.object_task_name is None
    assert tuple(candidate[0] for candidate in candidates) == ("green",)


def test_master_main_search_lock_reselects_before_target_found_and_hardens_afterward() -> None:
    module = load_master_main()
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
    module.OBJECT_STABLE_FRAMES = 99
    module.OBJECT_SELECTION_STABLE_FRAMES = 3
    assert module.OBJECT_LOCK_MISS_FRAMES == 3
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.write_data_line = lambda _frame: None
    red = object_candidate(module, "red", 150, 20, 170, 40)
    brown = object_candidate(module, "brown", 150, 20, 170, 40)

    for _ in range(module.OBJECT_SELECTION_STABLE_FRAMES):
        module.state.current_object_candidates = (red,)
        module.process_task_frame(FakeImage())
    first_miss = module.build_object_candidates(FakeImage(), (brown,))
    second_miss = module.build_object_candidates(FakeImage(), (brown,))
    candidates = module.build_object_candidates(FakeImage(), (brown,))
    for _ in range(module.OBJECT_SELECTION_STABLE_FRAMES):
        module.state.current_object_candidates = candidates
        module.process_task_frame(FakeImage())

    assert first_miss == ()
    assert second_miss == ()
    assert tuple(candidate[0] for candidate in candidates) == ("brown",)
    assert module.state.object_task_name == "brown"
    assert module.state.stable_frame_count == 0

    module.state.last_event_context_id = 7

    assert module.build_object_candidates(FakeImage(), (red,)) == ()


@pytest.mark.parametrize("use_yolo", (True, False))
def test_master_main_single_frame_miss_does_not_reuse_previous_candidate(use_yolo) -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = use_yolo
    module.state.current_task = {
        "context_id": 12,
        "state": module.State.SEARCH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.Task.SEARCH,
    }

    class Blob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    class FrameImage(FakeImage):
        def __init__(self):
            super().__init__()
            self.blobs = [Blob()]

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            return self.blobs

    img = FrameImage()
    yolo_candidates = (("red", 160.0, 220.0, 400.0, Blob()),)
    first = module.build_object_candidates(img, yolo_candidates if use_yolo else ())
    img.blobs = []
    second = module.build_object_candidates(img, ())

    assert first
    assert second == ()
def test_master_main_disable_yolo_uses_blob_candidates_in_every_object_task() -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = False
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.SEARCH_OBJECT),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )

    class Blob:
        def rect(self):
            return (150, 10, 20, 20)

        def cx(self):
            return 160.0

        def area(self):
            return 400.0

    img = DynamicThresholdRoiImage(module.OBJECT_TASKS[0][1][0], [Blob()])

    candidates = module.build_object_candidates(img, ())

    assert tuple(candidate[:4] for candidate in candidates) == (("red", 160.0, 30, 400.0),)
    assert module.state.current_detection_source == "blob"


def test_master_main_yolo_mode_does_not_call_blob_detector() -> None:
    module = load_master_main()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.handle_control_frame(
        task_sync_frame(
            module,
            context_id=12,
            state=int(module.State.SEARCH_OBJECT),
            target=int(module.Target.OBJECT),
            arg=int(module.Task.TRANSPORT),
        )
    )
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


def test_master_main_object_task_config_keeps_only_filter_parameters() -> None:
    module = load_master_main()
    for task in module.OBJECT_TASKS:
        task_name = task[0]
        expected = (
            task[0],
            int(task[2]),
            int(task[3]),
            int(task[4]),
            max(0, int(task[5])),
            bool(task[6]),
        )

        assert module._object_task_config(task_name) == expected


def test_master_main_enables_all_yolo_object_classes() -> None:
    module = load_master_main()

    assert tuple(task[0] for task in module.OBJECT_TASKS) == (
        "red",
        "blue",
        "brown",
        "white",
        "green",
    )
    assert tuple(module.object_task_id(name) for name in module.YOLO_LABELS) == (
        5,
        1,
        2,
        3,
        4,
    )


def test_master_main_object_task_config_accepts_legacy_threshold_layout() -> None:
    module = load_master_main()
    module.OBJECT_TASKS = (("red", ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),)
    task = module.OBJECT_TASKS[0]
    expected = (
        task[0],
        int(task[2]),
        int(task[3]),
        int(task[4]),
        int(task[5]),
        bool(task[6]),
    )

    assert module._object_task_config(task[0]) == expected


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
    def __init__(self, detections=None):
        self.detections = list(detections or ())
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

    def replace(self, **_kwargs):
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
        _ = thresholds, pixels_threshold, area_threshold, merge, roi, margin
        return []


class FixedThresholdPreviewImage(FakeImage):
    def __init__(self, expected_threshold, blob=None):
        super().__init__()
        self.expected_threshold = tuple(expected_threshold)
        self.preview_blob = blob or FakeBlob(120, 30, 30, 20, 900)
        self.find_blobs_calls = []

    def find_blobs(
        self,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge,
        roi=None,
        margin=None,
    ):
        _ = pixels_threshold, area_threshold, merge, margin
        threshold = tuple(thresholds[0])
        self.find_blobs_calls.append((threshold, roi))
        if threshold == self.expected_threshold and roi is None:
            return [self.preview_blob]
        return []


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


def task_sync_frame(
    module,
    seq=12,
    context_id=7,
    state=None,
    target=None,
    arg=None,
):
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
    return (150.0 / 320.0, 190.0 / 240.0, 170.0 / 320.0, 210.0 / 240.0, 1, 0.95)


def transport_aligned_detection(module):
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT)
    return pixel_detection(target_x - 10.0, target_y - 20.0, target_x + 10.0, target_y)


def pixel_detection(left, top, right, bottom, label=1, score=0.95):
    return (
        float(left) / float(IMAGE_WIDTH),
        float(top) / float(IMAGE_HEIGHT),
        float(right) / float(IMAGE_WIDTH),
        float(bottom) / float(IMAGE_HEIGHT),
        label,
        score,
    )


def object_candidate(module, task_name, left, top, right, bottom):
    blob = module.YoloDetectionBlob(left, top, right, bottom, 0, 0.95)
    return task_name, blob.cx(), float(bottom), blob.area(), blob


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
    module.state.current_object_candidates = tuple(module.yolo_detect(img))
    return img


def run_frame(module, img):
    cache_yolo_candidates(module, img)
    module.process_task_frame(img)


def test_master_main_replies_task_sync_ack_and_records_task() -> None:
    module = load_master_main()

    reply = module.handle_control_frame(task_sync_frame(module, seq=12, context_id=7))

    assert decode_frame(reply) == {
        "mode": MODE_ACK,
        "topic": module.Topic.MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 10,
    }
    assert module.state.current_task["context_id"] == 7
    assert module.state.current_task["arg"] == module.Task.SEARCH


def test_master_main_runtime_state_uses_state_object() -> None:
    module = load_master_main()

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
    assert module.IS_FINAL_ROUND is True
    assert (
        module.FINAL_ROUND_SELECTION_MODE
        == module._ObjectSelectionMode.NEAREST_BOTTOM
    )
    assert module.RED_SELECTION_MODE == module._RedSelectionMode.FIRST
    assert not hasattr(module, "FINAL_RED_SELECTION_MODE")
    assert module.Event.TARGET_FOUND == 6
    assert module.Target.OBJECT == 1
    assert not hasattr(module, "MODE_UDP")
    assert not hasattr(module, "TOPIC_MASTER_VISION_EVENT_REPORT")
    assert not hasattr(module, "STATE_SEARCH_OBJECT")
    assert not hasattr(module, "TaskConfig")
    assert not hasattr(module, "MASTER_SEARCH_TASK_CONFIG_ID")
    assert not hasattr(module, "EVENT_TARGET_FOUND")
    assert not hasattr(module, "TARGET_OBJECT")


def test_master_main_process_uart_input_replies_ack_for_task_sync_frame() -> None:
    module = load_master_main()
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


def test_master_main_process_uart_input_resyncs_before_task_sync_frame() -> None:
    module = load_master_main()
    uart = ReadWriteUART(b"\x02" + task_sync_frame(module))
    module.state.uart_device = uart

    rx_buffer = module.process_uart_input(b"")

    assert rx_buffer == b""
    assert decode_frame(uart.writes[0])["topic"] == module.Topic.MASTER_VISION_TASK_SYNC
    assert module.state.current_task is not None


def test_master_main_process_uart_input_ignores_bad_decode() -> None:
    module = load_master_main()
    module.state.uart_device = BadReadUART()

    rx_buffer = module.process_uart_input(b"partial")

    assert rx_buffer == b"partial\xff"
    assert module.state.current_task is None
def test_master_main_repeated_task_sync_replies_ack_without_reapplying() -> None:
    module = load_master_main()
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


def test_master_main_non_new_context_does_not_override_active_task() -> None:
    module = load_master_main()
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


def test_master_main_wrong_ack_does_not_clear_pending_event() -> None:
    module = load_master_main()
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


def test_master_main_repeats_event_until_matching_ack() -> None:
    module = load_master_main()
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


def test_master_main_creates_target_found_once_per_context() -> None:
    module = load_master_main()
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    img = FakeImage()
    set_current_image(module, img)
    module.accept_observation(build_search_observation(module, 180.0), img)
    first = module.next_event_frame()
    module.handle_control_frame(event_ack_frame(module, 1))
    module.accept_observation(build_search_observation(module, 180.0), img)

    assert first is not None
    assert module.next_event_frame() is None


def test_master_main_search_uses_yolo_candidates_for_velocity_and_target_found() -> None:
    module = load_master_main()
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
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


def test_master_main_final_object_marker_keeps_target_found_event() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.LAST
    module.tf.detect = lambda net, img: [search_aligned_detection()]
    module.handle_control_frame(
        task_sync_frame(module, context_id=7, arg=0x101)
    )
    uart = FakeUART()
    module.state.uart_device = uart
    img = FakeImage(detections=[search_aligned_detection()])

    run_frame(module, img)
    run_frame(module, img)

    assert latest_event(uart) == {
        "context_id": 7,
        "event": module.Event.TARGET_FOUND,
        "value": 1,
    }


def test_master_main_edge_selection_prefers_previous_target_edge_among_two_outer_candidates() -> None:
    module = load_master_main()
    module.state.last_object_edge_group = module.object_edge_group("brown")
    red = object_candidate(module, "red", 10, 20, 30, 60)
    white = object_candidate(module, "white", 280, 20, 300, 60)
    blue = object_candidate(module, "blue", 80, 20, 100, 60)

    task_name, _, _, _, best_blob = module.choose_search_candidate(
        (red, white, blue)
    )

    assert best_blob is white[4]
    assert task_name == "white"


def test_master_main_first_search_prefers_left_group_after_red_filter() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    red = object_candidate(module, "red", 0, 20, 20, 60)
    blue = object_candidate(module, "blue", 20, 20, 40, 60)
    brown = object_candidate(module, "brown", 280, 20, 300, 60)
    module.state.current_object_candidates = (red, blue, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is blue[4]
    assert task_name == "blue"


def test_master_main_preliminary_mode_uses_nearest_target_candidate() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = False
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    red = object_candidate(module, "red", 10, 190, 30, 210)
    brown = object_candidate(module, "brown", 150, 190, 170, 210)
    module.state.current_object_candidates = (red, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is brown[4]
    assert task_name == "brown"


def test_master_main_final_mode_selects_candidate_nearest_bottom_edge() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    centered = object_candidate(module, "red", 150, 180, 170, 200)
    nearest_bottom = object_candidate(module, "brown", 20, 190, 40, 230)
    module.state.current_object_candidates = (centered, nearest_bottom)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is nearest_bottom[4]
    assert task_name == "brown"


def test_master_main_final_all_mode_keeps_red_in_selection() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.state.last_object_edge_group = 1
    red = object_candidate(module, "red", 10, 20, 30, 60)
    white = object_candidate(module, "white", 150, 20, 170, 60)
    module.state.current_object_candidates = (red, white)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is red[4]
    assert task_name == "red"


def test_master_main_final_red_last_mode_excludes_red_before_final_object() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.LAST
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    module.state.last_object_edge_group = 1
    red = object_candidate(module, "red", 10, 20, 30, 60)
    white = object_candidate(module, "white", 150, 20, 170, 60)
    module.state.current_object_candidates = (red, white)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is white[4]
    assert task_name == "white"


def test_master_main_final_red_last_mode_uses_nearest_candidate_on_final_object() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.LAST
    module.handle_control_frame(
        task_sync_frame(module, context_id=7, arg=0x101)
    )
    red = object_candidate(module, "red", 10, 20, 30, 60)
    brown = object_candidate(module, "brown", 150, 190, 170, 210)
    module.state.current_object_candidates = (red, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is brown[4]
    assert task_name == "brown"


def test_master_main_final_red_last_mode_accepts_final_object_without_red() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.LAST
    module.handle_control_frame(
        task_sync_frame(module, context_id=7, arg=0x101)
    )
    blue = object_candidate(module, "blue", 10, 20, 30, 60)
    brown = object_candidate(module, "brown", 150, 190, 170, 210)
    module.state.current_object_candidates = (blue, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is brown[4]
    assert task_name == "brown"


def test_master_main_final_no_red_mode_excludes_red_on_final_object() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.NEVER
    module.handle_control_frame(
        task_sync_frame(module, context_id=7, arg=0x101)
    )
    module.state.last_object_edge_group = 1
    red = object_candidate(module, "red", 10, 20, 30, 60)
    white = object_candidate(module, "white", 150, 20, 170, 60)
    module.state.current_object_candidates = (red, white)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is white[4]
    assert task_name == "white"


def test_master_main_final_red_first_mode_selects_nearest_red_on_first_object() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.FIRST
    module.handle_control_frame(
        task_sync_frame(module, context_id=7, arg=0x201)
    )
    far_red = object_candidate(module, "red", 10, 20, 30, 60)
    near_red = object_candidate(module, "red", 150, 190, 170, 210)
    brown = object_candidate(module, "brown", 155, 190, 175, 210)
    module.state.current_object_candidates = (far_red, near_red, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is near_red[4]
    assert task_name == "red"


def test_master_main_final_red_first_mode_excludes_red_after_first_object() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.RED_SELECTION_MODE = module._RedSelectionMode.FIRST
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    red = object_candidate(module, "red", 0, 20, 20, 60)
    blue = object_candidate(module, "blue", 20, 20, 40, 60)
    brown = object_candidate(module, "brown", 280, 20, 300, 60)
    module.state.current_object_candidates = (red, blue, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is blue[4]
    assert task_name == "blue"


@pytest.mark.parametrize(
    ("previous_task_name", "candidate_task_name", "left_x", "right_x", "expected_side"),
    (
        ("red", "green", 40, 300, "left"),
        ("brown", "green", 20, 280, "right"),
        ("green", "brown", 40, 300, "left"),
        ("green", "red", 20, 280, "right"),
    ),
)
def test_master_main_edge_selection_matches_candidate_side_to_target_edge(
    previous_task_name,
    candidate_task_name,
    left_x,
    right_x,
    expected_side,
) -> None:
    module = load_master_main()
    module.state.last_object_edge_group = module.object_edge_group(previous_task_name)
    left = object_candidate(module, candidate_task_name, left_x - 10, 20, left_x + 10, 60)
    right = object_candidate(module, candidate_task_name, right_x - 10, 20, right_x + 10, 60)

    _, _, _, _, best_blob = module.choose_search_candidate((left, right))

    expected = left if expected_side == "left" else right
    assert best_blob is expected[4]


def test_master_main_final_mode_selects_single_tennis_on_any_side() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.state.object_task_name = "red"
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    tennis = object_candidate(module, "green", 280, 20, 300, 60)
    module.state.current_object_candidates = (tennis,)

    observation, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert observation[3] == tennis[3]
    assert best_blob is tennis[4]
    assert task_name == "green"


def test_master_main_search_ignores_candidate_occluded_on_lower_center_line() -> None:
    module = load_master_main()
    module.IS_FINAL_ROUND = True
    module.state.object_task_name = "red"
    module.handle_control_frame(task_sync_frame(module, context_id=7))
    red = object_candidate(module, "red", 100, 120, 120, 200)
    brown = object_candidate(module, "brown", 80, 170, 115, 220)
    module.state.current_object_candidates = (red, brown)

    _, best_blob, task_name, _ = module.build_observation_and_candidates()

    assert best_blob is brown[4]
    assert task_name == "brown"


def test_master_main_yolo_detect_filters_small_area_candidates() -> None:
    module = load_master_main()
    module.state.yolo_net = "fake-net"
    module.tf.detect = lambda net, img: [(0.25, 0.125, 0.28, 0.145, 1, 0.95)]
    img = FakeImage()
    set_current_image(module, img)

    assert module.yolo_detect(img) == []


def test_master_main_debug_display_draws_detected_box_and_flushes() -> None:
    module = load_master_main()
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


def test_master_main_debug_mode_bypasses_communication_and_previews_yolo() -> None:
    module = load_master_main()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.OBJECT_DETECTION_USE_YOLO = True

    class StopLoop(Exception):
        pass

    img = FakeImage()
    blob = module.YoloDetectionBlob(150.0, 20.0, 170.0, 40.0, 1, 0.95)

    class Sensor:
        def snapshot(self):
            return img

    module.sensor = Sensor()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.init_uart = lambda: (_ for _ in ()).throw(
        AssertionError("调试模式不应初始化串口")
    )
    module.process_uart_input = lambda _buffer: (_ for _ in ()).throw(
        AssertionError("调试模式不应读取串口")
    )
    module.write_data_line = lambda _frame: (_ for _ in ()).throw(
        AssertionError("调试模式不应发送速度")
    )
    module.write_reliable_line = lambda _frame: (_ for _ in ()).throw(
        AssertionError("调试模式不应发送可靠消息")
    )
    module.tf.load = lambda _path, load_to_fb=False: "fake-net"
    module.yolo_detect = lambda current_img: (
        (("red", 160.0, 220.0, 400.0, blob),) if current_img is img else ()
    )
    draw_preview = module.draw_search_preview_debug

    def stop_after_preview(current_img, candidates):
        draw_preview(current_img, candidates)
        raise StopLoop()

    module.draw_search_preview_debug = stop_after_preview

    with pytest.raises(StopLoop):
        module.run()

    assert img.rectangles
    assert any(entry[2] == "red" for entry in img.strings)
    assert img.flush_count == 1
    assert module.state.uart_device is None


def test_master_main_blob_debug_preview_reports_detected_object() -> None:
    module = load_master_main()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.OBJECT_DETECTION_USE_YOLO = False
    brown_threshold = module.OBJECT_TASKS[0][1][0]
    img = FixedThresholdPreviewImage(brown_threshold)
    set_current_image(module, img)

    module._process_debug_preview_frame(img)

    assert any(call == (tuple(brown_threshold), None) for call in img.find_blobs_calls)
    assert any(entry[2] == "red" for entry in img.strings)
    assert not any("touch=" in entry[2] for entry in img.strings)
    assert img.flush_count == 1


def test_master_main_build_observation_and_candidates_uses_cached_candidates_without_current_image() -> None:
    module = load_master_main()
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
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
    module.state.current_object_candidates = (("red", target_x, target_y, 400.0, blob),)

    observation, best_blob, task_name, candidates = module.build_observation_and_candidates()

    assert observation == (7, 0.0, 0.0, 400.0)
    assert best_blob is blob
    assert task_name == "red"
    assert candidates == (("red", target_x, target_y, 400.0, blob),)


def test_master_main_missing_target_outputs_configured_search_velocity() -> None:
    module = load_master_main()
    module.state.current_image_height = IMAGE_HEIGHT
    velocity = module.build_search_velocity_from_observation((7, 0.0, 0.0, 0.0))

    assert velocity == (module.MASTER_MISSING_SEARCH_VX, module.MASTER_MISSING_SEARCH_VY)


def test_master_main_search_velocity_deadzone_zeroes_each_axis() -> None:
    module = load_master_main()
    observation = build_search_observation(
        module,
        150.0,
        err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX),
        err_y=float(module.MASTER_SEARCH_DEADZONE_Y_PX),
    )

    module.state.current_image_height = IMAGE_HEIGHT
    velocity = module.build_search_velocity_from_observation(observation)

    assert velocity == (0.0, 0.0)


def test_master_main_search_velocity_applies_min_speed_outside_deadzone(monkeypatch) -> None:
    module = load_master_main()
    module.state.current_image_height = IMAGE_HEIGHT
    y_error = float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1
    kp_sign = 1.0 if float(module.MASTER_SEARCH_KP_Y) > 0.0 else -1.0
    monkeypatch.setattr(
        module,
        "MASTER_SEARCH_KP_Y",
        kp_sign * float(module.MASTER_SEARCH_MIN_SPEED) / (2.0 * y_error),
    )
    positive = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1,
            err_y=y_error,
        )
    )
    negative = module.build_search_velocity_from_observation(
        build_search_observation(
            module,
            150.0,
            err_x=-(float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1),
            err_y=-y_error,
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


def test_master_main_search_velocity_clamps_vx_and_vy() -> None:
    module = load_master_main()
    module.state.current_image_height = IMAGE_HEIGHT
    clamp_x = abs(float(module.MASTER_SEARCH_MAX_VX) / float(module.MASTER_SEARCH_KP_X)) + 1.0
    clamp_y = abs(float(module.MASTER_SEARCH_MAX_VY) / float(module.MASTER_SEARCH_KP_Y)) + 1.0
    positive = module.build_search_velocity_from_observation((7, clamp_x, clamp_y, 300.0))
    negative = module.build_search_velocity_from_observation((7, -clamp_x, -clamp_y, 300.0))

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


def test_master_main_reference_frame_interval_is_derived_from_fps() -> None:
    module = load_master_main()
    module.VISION_REFERENCE_FPS = 25

    assert module.reference_frame_interval_ms() == pytest.approx(40.0)


def test_master_main_slow_frame_interval_reduces_search_velocity() -> None:
    module = load_master_main()
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms()
    module.state.current_image_height = IMAGE_HEIGHT
    reference_velocity = module.build_search_velocity_from_observation((7, 100.0, -100.0, 300.0))
    module.state.current_frame_interval_ms = module.reference_frame_interval_ms() * 2
    slow_velocity = module.build_search_velocity_from_observation((7, 100.0, -100.0, 300.0))

    assert abs(slow_velocity[0]) < abs(reference_velocity[0])
    assert abs(slow_velocity[1]) < abs(reference_velocity[1])
    assert slow_velocity == pytest.approx((reference_velocity[0] * 0.5, reference_velocity[1] * 0.5))


def test_master_main_search_y_velocity_decreases_when_target_gets_closer() -> None:
    module = load_master_main()
    module.state.current_image_height = IMAGE_HEIGHT
    close_error = -(
        max(
            float(module.MASTER_SEARCH_DEADZONE_Y_PX),
            float(module.MASTER_SEARCH_MIN_SPEED) / abs(float(module.MASTER_SEARCH_KP_Y)),
        )
        + 1.0
    )
    far_error = close_error - 1.0
    far_velocity = module.build_search_velocity_from_observation((7, 0.0, far_error, 1000.0))
    close_velocity = module.build_search_velocity_from_observation((7, 0.0, close_error, 1000.0))

    assert close_velocity[1] < far_velocity[1]
    assert close_velocity[1] == pytest.approx(close_error * module.MASTER_SEARCH_KP_Y)


def test_master_main_orbit_outputs_independent_xy_velocity_correction() -> None:
    module = load_master_main()
    module.MASTER_ORBIT_KP_X = 0.2
    module.MASTER_ORBIT_KP_Y = -0.3
    module.MASTER_ORBIT_MIN_SPEED = 0.0
    module.MASTER_ORBIT_MAX_VX = 9.0
    module.MASTER_ORBIT_MAX_VY = 9.0
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
    target_x, target_y = module.build_search_target_point(module.Task.ORBIT)
    blob = module.YoloDetectionBlob(
        target_x + 30.0,
        IMAGE_HEIGHT - target_y,
        target_x + 50.0,
        IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.state.current_object_candidates = (("red", target_x + 40.0, target_y - 20.0, 400.0, blob),)

    module.process_task_frame(img)

    velocity = latest_velocity(uart)
    expected_y = -20.0 * float(module.MASTER_ORBIT_KP_Y)
    assert velocity["vx"] == pytest.approx(40.0 * module.MASTER_ORBIT_KP_X)
    assert velocity["vy"] == pytest.approx(expected_y)
    assert len(uart.writes) == 1


def test_master_main_transport_alignment_reports_aligned_event() -> None:
    module = load_master_main()
    module.tf.detect = lambda net, img: [transport_aligned_detection(module)]
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


def test_master_main_transport_candidate_selection_uses_configured_target_point() -> None:
    module = load_master_main()
    module.state.yolo_net = "fake-net"
    target_x, target_y = module.build_search_target_point(module.Task.TRANSPORT)
    module.tf.detect = lambda net, img: [
        pixel_detection(target_x - 40, target_y - 20.0, target_x + 40, target_y),
        pixel_detection(
            target_x - 40,
            target_y,
            target_x + 40,
            target_y + 20.0,
        ),
    ]
    module.handle_control_frame(task_sync_frame(module, arg=int(module.Task.TRANSPORT)))

    cache_yolo_candidates(module)
    observation, best_blob, _, _ = module.build_observation_and_candidates()

    assert best_blob is not None
    assert observation == (7, 0.0, 0.0, 1600.0)


def test_master_main_transport_alignment_keeps_candidate_selection() -> None:
    module = load_master_main()
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
def test_master_main_pending_event_blocks_non_return_velocity_until_ack() -> None:
    module = load_master_main()
    module.RED_SELECTION_MODE = module._RedSelectionMode.ALL
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
def test_master_main_blob_debug_preview_replaces_previous_candidates() -> None:
    module = load_master_main()
    module.MASTER_DEBUG_DISPLAY_ENABLED = True
    module.OBJECT_DETECTION_USE_YOLO = False
    brown_threshold = module.OBJECT_TASKS[0][1][0]
    image = FixedThresholdPreviewImage(brown_threshold)
    set_current_image(module, image)
    yolo_blob = module.YoloDetectionBlob(150.0, 30.0, 170.0, 50.0, 1, 0.95)
    module.state.current_detection_source = "yolo"
    module.state.current_object_candidates = (("red", 160.0, 210.0, 400.0, yolo_blob),)

    module._process_debug_preview_frame(image)

    assert tuple(module.state.current_object_candidates[:1])[0][:4] == (
        "red",
        135.0,
        50,
        900.0,
    )
    assert any(call == (tuple(brown_threshold), None) for call in image.find_blobs_calls)
    assert any(entry[2] == "red" for entry in image.strings)
    assert image.flush_count == 1
