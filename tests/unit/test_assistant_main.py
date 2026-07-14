"""辅车视觉默认入口行为测试."""

# pyright: reportAttributeAccessIssue=false

import inspect
import pytest

from tests.test_support import load_role_entry_module
import tests.unit.assistant_object_approach_support as legacy_tests


ASSISTANT_LOG_PREFIX = "[assistant_" + "v" + "2]"


def load_assistant_main():
    """加载辅车视觉默认入口模块."""

    module = load_role_entry_module("assistant", "run.py", "assistant_main_test_module")
    module.reset_runtime_state()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = False
    module.state.yolo_net = "fake-yolo-net"
    module.prepare_runtime = module.init_status_lights
    return module


def start_run_as_local_vision_paused(module):
    original_reset_runtime_state = module.reset_runtime_state

    def reset_as_paused():
        original_reset_runtime_state()
        module.state.local_vision_control_paused = True

    module.reset_runtime_state = reset_as_paused


legacy_tests.load_assistant = load_assistant_main


def assistant_target_point(module, config_id=None):
    if config_id is None:
        config_id = module.Task.SEARCH
    return module.build_object_target_point(config_id)


@pytest.mark.parametrize(
    ("state_value", "config_id", "run_mode"),
    (
        (2, 1, "approach_object"),
        (2, 2, "approach_object"),
        (4, 2, "approach_object"),
        (3, 3, "orbit_object"),
        (3, 2, "approach_object"),
    ),
)
def test_assistant_main_yolo_mode_runs_every_object_stage_without_pause(
    state_value,
    config_id,
    run_mode,
) -> None:
    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = True

    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = (strength, zoom)
            return self

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("YOLO 模式不应调用物体色块识别")

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.load_yolo_model = lambda: "fake-yolo-net"

    def prime_object_task(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": state_value,
            "target": module.Target.OBJECT,
            "arg": module.pack_task_arg(config_id, 1),
        }
        module.state.mode = run_mode
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
    ("state_value", "config_id", "run_mode"),
    (
        (2, 1, "approach_object"),
        (2, 2, "approach_object"),
        (4, 2, "approach_object"),
        (3, 3, "orbit_object"),
        (3, 2, "approach_object"),
    ),
)
def test_assistant_main_blob_mode_runs_every_object_stage(
    state_value,
    config_id,
    run_mode,
) -> None:
    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = False

    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def __init__(self):
            self.object_blob_calls = 0

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = (strength, zoom)
            return self

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            self.object_blob_calls += 1
            return []

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)

    def prime_object_task(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": state_value,
            "target": module.Target.OBJECT,
            "arg": module.pack_task_arg(config_id, 1),
        }
        module.state.mode = run_mode
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


@pytest.mark.parametrize("use_yolo", (True, False))
def test_assistant_main_single_frame_miss_does_not_reuse_previous_candidate(use_yolo) -> None:
    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = use_yolo
    module.state.current_sync = {
        "reliable_seq": 12,
        "state": module.State.APPROACH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.pack_task_arg(module.Task.SEARCH, 1),
    }
    module.state.object_task_name = "red"

    class Blob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class FrameImage:
        def __init__(self):
            self.blobs = [Blob()]

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            return self.blobs

    img = FrameImage()
    yolo_candidates = (("red", 160.0, 30.0, 220.0, 400.0, Blob()),)
    first = module.build_object_candidates(img, yolo_candidates if use_yolo else ())
    img.blobs = []
    second = module.build_object_candidates(img, ())

    assert first
    assert second == ()


def test_assistant_main_transport_window_keeps_x_outside_deadzone_candidate() -> None:
    """辅车搬运候选窗口覆盖全图横向范围."""

    module = load_assistant_main()
    target_x, target_y = assistant_target_point(module, module.Task.TRANSPORT)
    candidate = (
        "red",
        target_x + float(module.OBJECT_APPROACH_DEADZONE_X_PX) + 1.0,
        0.0,
        target_y - 1.0,
        300.0,
        object(),
    )

    assert module.filter_candidates_in_target_window(
        [candidate],
        target_x,
        target_y,
        module.OBJECT_TRANSPORT_WINDOW_X_PX,
        module.OBJECT_TRANSPORT_WINDOW_Y_PX,
    ) == [candidate]


def test_assistant_main_transport_window_rejects_y_outside_candidate() -> None:
    """辅车搬运窗口必须过滤底边 Y 达到命中线外侧的候选."""

    module = load_assistant_main()
    target_x, target_y = assistant_target_point(module, module.Task.TRANSPORT)
    candidates = [
        (
            "red",
            target_x,
            0.0,
            target_y,
            300.0,
            object(),
        )
    ]

    assert module.filter_candidates_in_target_window(
        candidates,
        target_x,
        target_y,
        module.OBJECT_TRANSPORT_WINDOW_X_PX,
        module.OBJECT_TRANSPORT_WINDOW_Y_PX,
    ) == []


def test_assistant_main_transport_window_keeps_xy_inside_candidate() -> None:
    """辅车搬运窗口保留中心 X 与底边 Y 都命中的候选."""

    module = load_assistant_main()
    target_x, target_y = assistant_target_point(module, module.Task.TRANSPORT)
    candidate = ("red", target_x, 0.0, target_y - 1.0, 300.0, object())

    assert module.filter_candidates_in_target_window(
        [candidate],
        target_x,
        target_y,
        module.OBJECT_TRANSPORT_WINDOW_X_PX,
        module.OBJECT_TRANSPORT_WINDOW_Y_PX,
    ) == [candidate]


def test_assistant_main_transport_window_rejects_y_above_hit_line_candidate() -> None:
    """辅车搬运窗口按底边 Y 小于命中线判断纵向命中."""

    module = load_assistant_main()
    target_x, target_y = assistant_target_point(module, module.Task.TRANSPORT)
    candidate = ("red", target_x, 0.0, target_y + 20.0, 300.0, object())

    assert module.filter_candidates_in_target_window(
        [candidate],
        target_x,
        target_y,
        module.OBJECT_TRANSPORT_WINDOW_X_PX,
        module.OBJECT_TRANSPORT_WINDOW_Y_PX,
    ) == []


def test_assistant_main_object_candidates_use_yolo_when_enabled() -> None:
    """辅车 main 开启 YOLO 时使用模型结果生成找物体候选."""

    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = True

    class FakeYoloTf:
        def __init__(self):
            self.load_calls = []
            self.detect_calls = []

        def load(self, path, load_to_fb=False):
            self.load_calls.append((path, load_to_fb))
            return "fake-yolo-net"

        def detect(self, net, img):
            self.detect_calls.append((net, img))
            return [(0.25, 0.125, 0.75, 0.2083333333, 1, 0.95)]

    class FakeImage:
        def __init__(self):
            self.copy_calls = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def copy(self, scale, copy_to_fb):
            self.copy_calls.append((scale, copy_to_fb))
            return "detect-image"

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            raise AssertionError("main 找物体主线不应回退到色块识别")

    module.tf = FakeYoloTf()
    module.state.yolo_net = None
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    module.state.yolo_net = module.load_yolo_model()
    raw_yolo_candidates = module.yolo_detect(img)
    candidates = module.build_object_candidates(img, raw_yolo_candidates)

    assert module.tf.load_calls == [(module.YOLO_MODEL_PATH, True)]
    assert module.tf.detect_calls == [("fake-yolo-net", "detect-image")]
    assert img.copy_calls == [(module.YOLO_IMAGE_COPY_SCALE, 1)]
    assert candidates[0][0] == "red"
    assert candidates[0][1] == pytest.approx(160.0)
    assert candidates[0][3] == pytest.approx(210.0)
    assert candidates[0][4] == pytest.approx(3200.0)


def test_assistant_main_yolo_detect_filters_small_area_candidates() -> None:
    """辅车 YOLO 结果面积过小时不应进入候选."""

    module = load_assistant_main()

    class FakeYoloTf:
        def detect(self, net, img):
            _ = (net, img)
            return [(0.25, 0.125, 0.28, 0.145, 1, 0.95)]

    class FakeImage:
        def copy(self, scale, copy_to_fb):
            _ = (scale, copy_to_fb)
            return "detect-image"

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

    module.tf = FakeYoloTf()
    module.state.yolo_net = "cached"
    img = FakeImage()

    assert module.yolo_detect(img) == []


def test_assistant_main_exposes_master_style_runtime_api() -> None:
    """辅车 main 对外保留与主车一致的主流程入口名."""

    module = load_assistant_main()

    assert callable(module.format_search_velocity_frame)
    assert callable(module.parse_task_sync_packet)
    assert callable(module.parse_event_ack_packet)
    assert callable(module.build_search_target_point)
    assert callable(module.current_task_config_id)
    assert callable(module.build_observation)
    assert callable(module.build_observation_and_candidates)
    assert callable(module.build_search_velocity_from_observation)
    assert callable(module.build_orbit_correction_velocity_from_observation)
    assert callable(module.current_event_type)
    assert callable(module.required_stable_frames)
    assert callable(module.resolve_event_value)
    assert callable(module.create_pending_event)
    assert callable(module.handle_control_frame)
    assert callable(module.process_task_frame)
    assert callable(module.write_data_line)
    assert callable(module.write_reliable_line)
    assert hasattr(module.state, "current_object_candidates")
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
    assert tuple(inspect.signature(module.build_object_candidates).parameters) == ("img", "yolo_candidates")
    assert tuple(
        inspect.signature(module.build_object_approach_velocity_from_error).parameters
    ) == ("err_x", "err_y")
    assert tuple(
        inspect.signature(module.build_object_approach_velocity_from_observation).parameters
    ) == ("observation",)
    assert tuple(
        inspect.signature(module.build_object_orbit_velocity_from_observation).parameters
    ) == ("observation",)
    assert tuple(inspect.signature(module._draw_debug_protocol_point).parameters) == (
        "img",
        "x",
        "y",
    )
    assert tuple(inspect.signature(module.draw_selected_marker).parameters) == (
        "img",
        "blob",
        "pixel_x",
        "pixel_y",
    )
    assert not hasattr(module, "format_vision_frame")
    assert not hasattr(module, "parse_sync_packet")
    assert not hasattr(module, "parse_ack_packet")
    assert tuple(inspect.signature(module.build_object_blob_candidates).parameters) == ("img",)
    assert not hasattr(module, "process_frame")
    assert not hasattr(module, "process_follow_frame")
    assert not hasattr(module, "process_object_frame")
    assert not hasattr(module, "process_return_line_frame")


def test_assistant_main_build_object_candidates_matches_master_style_signature() -> None:
    """辅车 main 的找物体候选入口应显式接收模型候选."""

    module = load_assistant_main()

    with pytest.raises(TypeError):
        module.build_object_candidates(object())


def test_assistant_main_object_task_config_keeps_only_filter_parameters() -> None:
    module = load_assistant_main()
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


def test_assistant_main_object_task_config_accepts_legacy_threshold_layout() -> None:
    module = load_assistant_main()
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


def test_assistant_main_process_uart_input_matches_master_style_signature() -> None:
    """辅车 main 的串口轮询入口应和主车一样只接受全局缓冲区."""

    module = load_assistant_main()

    with pytest.raises(TypeError):
        module.process_uart_input(legacy_tests.FakeUART(), b"", module.state)


def test_assistant_main_exposes_master_style_yolo_detect_api() -> None:
    """辅车 main 提供与主车一致的 yolo_detect 入口."""

    module = load_assistant_main()

    class FakeYoloTf:
        def __init__(self):
            self.detect_calls = []

        def detect(self, net, img):
            self.detect_calls.append((net, img))
            return [(0.25, 0.125, 0.75, 0.2083333333, 1, 0.95)]

    class FakeImage:
        def __init__(self):
            self.copy_calls = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def copy(self, scale, copy_to_fb):
            self.copy_calls.append((scale, copy_to_fb))
            return "detect-image"

    module.tf = FakeYoloTf()
    module.state.yolo_net = "cached-yolo-net"
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    candidates = module.yolo_detect(img)

    assert module.tf.detect_calls == [("cached-yolo-net", "detect-image")]
    assert img.copy_calls == [(module.YOLO_IMAGE_COPY_SCALE, 1)]
    assert candidates[0][0] == "red"
    assert candidates[0][1] == pytest.approx(160.0)
    assert candidates[0][3] == pytest.approx(210.0)
    assert candidates[0][4] == pytest.approx(3200.0)


def test_assistant_main_run_applies_lens_correction_and_uses_yolo_detect_before_processing() -> None:
    """辅车 follow 模式运行循环应先做镜头校正, 且不进入物体 YOLO 链路."""

    module = load_assistant_main()

    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def __init__(self):
            self.lens_corr_called = False

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = strength, zoom
            self.lens_corr_called = True
            return self

    image = SnapshotImage()
    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.process_uart_input = lambda rx_buffer: rx_buffer

    def fake_yolo_detect(img):
        _ = img
        raise AssertionError("follow 模式不应调用 yolo_detect")

    def stop_after_frame(current_img):
        assert current_img is image
        assert image.lens_corr_called is True
        assert tuple(module.state.current_object_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()


def test_assistant_main_run_applies_lens_correction_and_uses_yolo_preview_without_task_sync() -> None:
    """辅车调试模式绕过通信并直接显示物体识别."""

    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True
    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def __init__(self):
            self.lens_corr_called = False
            self.rectangles = []
            self.strings = []
            self.crosses = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = strength, zoom
            self.lens_corr_called = True
            return self

        def draw_rectangle(self, rect, **_kwargs):
            self.rectangles.append(rect)

        def draw_string(self, x, y, text, **_kwargs):
            self.strings.append((x, y, text))

        def draw_cross(self, x, y, **_kwargs):
            self.crosses.append((x, y))

        def flush(self):
            raise StopLoop()

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

    module.sensor = Sensor()
    module.init_uart = lambda: (_ for _ in ()).throw(
        AssertionError("调试模式不应初始化串口")
    )
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.process_uart_input = lambda _buffer: (_ for _ in ()).throw(
        AssertionError("调试模式不应读取串口")
    )

    def fake_yolo_detect(img):
        assert img is image
        return [("red", 160.0, 30.0, 220.0, 400.0, FakeBlob())]

    module.yolo_detect = fake_yolo_detect
    module.write_data_line = lambda _frame_bytes: (_ for _ in ()).throw(
        AssertionError("上电调试预览不应发送速度")
    )
    module.write_reliable_line = lambda _frame_bytes: (_ for _ in ()).throw(
        AssertionError("上电调试预览不应发送可靠控制")
    )

    with pytest.raises(StopLoop):
        module.run()

    assert image.lens_corr_called is True
    assert image.rectangles
    assert any(entry[2] == "red" for entry in image.strings)
    assert image.crosses
    assert module.state.uart_device is None
class DynamicThresholdCalibrationImage:
    def __init__(self, bbox, foreground, background, fragment=None):
        self.left, self.top, self.right, self.bottom = bbox
        self.foreground = foreground
        self.background = background
        self.fragment = fragment

    def width(self):
        return legacy_tests.IMAGE_WIDTH

    def height(self):
        return legacy_tests.IMAGE_HEIGHT

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
        return legacy_tests.IMAGE_WIDTH

    def height(self):
        return legacy_tests.IMAGE_HEIGHT

    def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=0):
        _ = (pixels_threshold, area_threshold, merge, roi, margin)
        key = tuple(thresholds[0])
        self.find_blobs_calls.append((key, roi))
        if len(thresholds) == 1 and key == self.expected_threshold:
            return list(self.blobs)
        return []


def test_assistant_main_disable_yolo_uses_blob_candidates_in_every_object_task() -> None:
    """关闭 YOLO 后, 辅车物体阶段直接使用色块候选."""

    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = False
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    class Blob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    img = DynamicThresholdRoiImage(module.OBJECT_TASKS[0][1][0], [Blob()])

    candidates = module.build_object_candidates(img, ())

    assert tuple(candidate[:5] for candidate in candidates) == (("red", 160.0, 30.0, 220.0, 400.0),)
    assert module.state.current_detection_source == "blob"


def test_assistant_main_build_object_observation_and_candidates_uses_current_object_candidates() -> None:
    module = load_assistant_main()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_target_point(module)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        legacy_tests.IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        legacy_tests.IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_object_candidates = (("red", target_x, 30.0, target_y, 400.0, blob),)

    observation, best_blob, task_name, candidates = module.build_object_observation_and_candidates()

    assert observation == (0.0, 0.0, 400.0)
    assert best_blob is blob
    assert task_name == "red"
    assert candidates == [("red", target_x, 30.0, target_y, 400.0, blob)]


def test_assistant_main_yolo_mode_does_not_call_blob_detector() -> None:
    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    class BlobForbiddenImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, *args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("辅车搬运前对正阶段不应调用色块识别")

    candidates = module.build_object_candidates(BlobForbiddenImage(), ())

    assert candidates == ()
    assert module.state.current_detection_source == "yolo"


def test_assistant_main_new_task_sync_clears_previous_pending_event() -> None:
    module = load_assistant_main()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            11,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    module.create_pending_event(11, module.Event.ALIGNED, 300)

    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    assert module.state.has_pending_event() is False


def test_assistant_main_blob_mode_uses_full_image_blob_search() -> None:
    module = load_assistant_main()
    module.OBJECT_DETECTION_USE_YOLO = False
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            11,
            module.State.ORBIT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    class BlobImage:
        def __init__(self):
            self.find_blobs_calls = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            self.find_blobs_calls.append(roi)
            return []

    img = BlobImage()

    module.build_object_candidates(img, ())

    assert img.find_blobs_calls == [None]


def test_assistant_main_process_task_frame_uses_cached_yolo_candidates() -> None:
    """辅车 main 的全局 task 入口应消费预先缓存的 YOLO 候选."""

    module = load_assistant_main()
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_target_point(module)

    class FakeBlob:
        def rect(self):
            return (target_x - 10, legacy_tests.IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return legacy_tests.IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, x, y):
            _ = (x, y)

        def draw_rectangle(self, *args):
            _ = args

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

    state.current_object_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )

    def fail_build_object_candidates(current_img):
        raise AssertionError("process_task_frame 不应再次自己做候选检测")

    module.build_object_candidates = fail_build_object_candidates
    uart = legacy_tests.FakeUART()

    module.state = state
    module.state.uart_device = uart
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.process_task_frame(img)

    legacy_tests.assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)


def test_assistant_main_handle_control_frame_reuses_global_state() -> None:
    """辅车 main 可像主车一样通过全局入口处理控制帧."""

    module = load_assistant_main()

    reply = module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    legacy_tests.assert_assistant_ack(module, reply, 12)
    assert module.state.mode == module.RunMode.APPROACH_OBJECT
    assert module.state.current_object_config_id() == module.Task.SEARCH


def test_assistant_main_parse_task_sync_packet_matches_master_style_name() -> None:
    """辅车 main 使用主车同名入口解析本地任务同步帧."""

    module = load_assistant_main()

    packet = module.parse_task_sync_packet(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 2),
        )
    )

    assert packet == {
        "reliable_seq": 12,
        "state": module.State.APPROACH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": legacy_tests.pack_task_arg(module.Task.SEARCH, 2),
    }


def test_assistant_main_build_search_velocity_wrapper_matches_object_path() -> None:
    """辅车 main 的主车式搜索速度入口仍复用找物体速度逻辑."""

    module = load_assistant_main()
    observation = (20.0, -30.0, 300.0)

    module.state.current_image_height = legacy_tests.IMAGE_HEIGHT
    assert module.build_search_velocity_from_observation(
        observation,
    ) == module.build_object_approach_velocity_from_observation(observation)


def test_assistant_main_missing_target_uses_configured_search_velocity() -> None:
    """辅车找不到物体时使用配置的搜索速度."""

    module = load_assistant_main()
    module.OBJECT_MISSING_SEARCH_VX = -1.25
    module.OBJECT_MISSING_SEARCH_VY = 3.75

    assert module.build_search_velocity_from_observation((0.0, 0.0, 0.0)) == (
        module.OBJECT_MISSING_SEARCH_VX,
        module.OBJECT_MISSING_SEARCH_VY,
    )


def test_assistant_main_orbit_outputs_independent_xy_velocity_correction() -> None:
    """辅车绕行修正直接输出独立 vx/vy 平移修正."""

    module = load_assistant_main()
    module.OBJECT_ORBIT_KP_X = 0.2
    module.OBJECT_ORBIT_KP_Y = -0.3
    module.OBJECT_ORBIT_MIN_SPEED = 0.0
    module.OBJECT_ORBIT_MAX_VX = 9.0
    module.OBJECT_ORBIT_MAX_VY = 9.0
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.ORBIT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.ORBIT, 1),
        )
    )
    target_x, target_y = assistant_target_point(module, module.Task.ORBIT)

    class FakeBlob:
        def rect(self):
            return (target_x + 30.0, legacy_tests.IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x + 40.0

        def cy(self):
            return legacy_tests.IMAGE_HEIGHT - target_y + 10.0

        def area(self):
            return 300.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, x, y):
            _ = (x, y)

        def draw_rectangle(self, *args):
            _ = args

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

    uart = legacy_tests.FakeUART()
    module.state.uart_device = uart
    module.state.current_object_candidates = (
        (
            "red",
            target_x + 40.0,
            legacy_tests.IMAGE_HEIGHT - target_y + 10.0,
            target_y - 20.0,
            300.0,
            FakeBlob(),
        ),
    )
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    module.process_task_frame(img)

    frame = module.decode_frame(uart.writes[0])
    body = module.decode_velocity_body(frame["body"])
    expected_y = -20.0 * (
        float(module.OBJECT_ORBIT_MAX_VY)
        / abs(float(module.OBJECT_ORBIT_KP_Y))
        / float(legacy_tests.IMAGE_HEIGHT)
    ) * float(module.OBJECT_ORBIT_KP_Y)
    assert body["vx"] == pytest.approx(40.0 * module.OBJECT_ORBIT_KP_X)
    assert body["vy"] == pytest.approx(expected_y)


def test_assistant_main_object_target_helpers_drop_unused_size_parameters() -> None:
    """辅车 main 的目标点与观测 helper 不再暴露无意义尺寸参数."""

    module = load_assistant_main()

    assert tuple(inspect.signature(module.build_object_target_point).parameters) == ("config_id",)
    assert tuple(inspect.signature(module.build_object_observation).parameters) == (
        "valid",
        "center_x",
        "bottom_y",
        "area",
    )
    assert tuple(inspect.signature(module.build_object_observation_and_candidates).parameters) == ()


def test_assistant_main_process_task_frame_uses_global_object_pipeline() -> None:
    """辅车 main 的全局 task 入口应像主车一样消费全局候选和全局状态."""

    module = load_assistant_main()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_target_point(module)

    class FakeBlob:
        def rect(self):
            return (target_x - 10, legacy_tests.IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return legacy_tests.IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, x, y):
            _ = (x, y)

        def draw_rectangle(self, *args):
            _ = args

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

    uart = legacy_tests.FakeUART()
    module.state.uart_device = uart
    module.state.current_object_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.process_task_frame(img)

    legacy_tests.assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)


def test_assistant_main_process_task_frame_calls_master_style_velocity_wrapper() -> None:
    """辅车 main 的全局 task 入口应走主车式速度包装入口."""

    module = load_assistant_main()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_target_point(module)

    class FakeBlob:
        def rect(self):
            return (target_x - 10, legacy_tests.IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return legacy_tests.IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, x, y):
            _ = (x, y)

        def draw_rectangle(self, *args):
            _ = args

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

    uart = legacy_tests.FakeUART()
    module.state.uart_device = uart
    module.state.current_object_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )
    call_log = []

    def fake_build_search_velocity_from_observation(observation):
        call_log.append(observation)
        return 0.0, 0.0

    def fail_build_object_approach_velocity_from_observation(observation, image_height):
        _ = (observation, image_height)
        raise AssertionError("全局主流程应走主车式搜索速度入口")

    module.build_search_velocity_from_observation = fake_build_search_velocity_from_observation
    module.build_object_approach_velocity_from_observation = (
        fail_build_object_approach_velocity_from_observation
    )

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.process_task_frame(img)

    assert len(call_log) == 1


def test_assistant_main_single_arg_process_uart_input_uses_global_state() -> None:
    """辅车 main 的单参串口轮询入口应直接驱动全局状态机."""

    module = load_assistant_main()
    uart = legacy_tests.FakeUART(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    module.state.uart_device = uart

    remainder = module.process_uart_input(b"")

    assert remainder == b""
    legacy_tests.assert_assistant_ack(module, uart.writes[0], 12)
    assert module.state.mode == module.RunMode.APPROACH_OBJECT


def test_assistant_main_single_arg_process_uart_input_calls_master_style_handle_control_frame() -> None:
    """辅车 main 的单参串口轮询入口应走全局控制包入口."""

    module = load_assistant_main()
    frame = legacy_tests.assistant_sync_frame(
        12,
        module.State.APPROACH_OBJECT,
        module.Target.OBJECT,
        legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
    )
    uart = legacy_tests.FakeUART(frame)
    module.state.uart_device = uart
    call_log = []

    def fake_handle_control_frame(frame_bytes):
        call_log.append(frame_bytes)
        return module.format_ack_frame(12)

    module.handle_control_frame = fake_handle_control_frame

    remainder = module.process_uart_input(b"")

    assert remainder == b""
    assert call_log == [frame]
def test_assistant_main_process_uart_input_without_any_matches_master_behavior() -> None:
    """辅车 main 不再为缺失 any 的串口对象静默返回."""

    module = load_assistant_main()
    module.state.uart_device = object()

    with pytest.raises(AttributeError):
        module.process_uart_input(b"")


def test_assistant_main_process_uart_input_read_error_propagates() -> None:
    """辅车 main 不再吞掉串口读取异常."""

    module = load_assistant_main()

    class BrokenUART:
        def any(self):
            return 1

        def read(self, size):
            _ = size
            raise RuntimeError("uart read failed")

    module.state.uart_device = BrokenUART()

    with pytest.raises(RuntimeError, match="uart read failed"):
        module.process_uart_input(b"")


def test_assistant_main_event_helpers_reflect_global_state() -> None:
    """辅车 main 的主车式事件 helper 应直接反映全局状态."""

    module = load_assistant_main()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    assert module.current_event_type() == module.Event.TARGET_FOUND
    assert module.required_stable_frames() == module.state.required_stable_frames
    assert module.resolve_event_value(300.0, None) == 300

    module.create_pending_event(12, module.Event.TARGET_FOUND, 300)

    assert module.state._pending_event == {
        "reliable_seq": 12,
        "event": module.Event.TARGET_FOUND,
        "value": 300,
    }
def test_assistant_run_applies_lens_correction_before_processing() -> None:
    """辅车非物体任务先做镜头校正, 且不缓存物体候选."""

    module = load_assistant_main()

    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def __init__(self):
            self.lens_corr_called = False

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = strength, zoom
            self.lens_corr_called = True
            return self

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.process_uart_input = lambda rx_buffer: rx_buffer

    def fake_yolo_detect(img):
        _ = img
        raise AssertionError("follow 模式不应调用 yolo_detect")

    def stop_after_frame(current_img):
        assert current_img is image
        assert image.lens_corr_called is True
        assert tuple(module.state.current_object_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()
