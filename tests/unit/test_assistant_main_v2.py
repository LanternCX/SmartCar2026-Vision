"""辅车视觉 main_v2 行为测试."""

import inspect
import pytest

from tests.test_support import load_role_entry_module
import tests.unit.test_assistant_object_approach as legacy_tests


def load_assistant_v2():
    """加载辅车视觉 main_v2 入口模块."""

    module = load_role_entry_module("assistant", "main_v2.py", "assistant_main_v2_test_module")
    module.reset_runtime_state()
    module.state.yolo_net = "fake-yolo-net"
    return module


def start_run_as_local_vision_paused(module):
    original_reset_runtime_state = module.reset_runtime_state

    def reset_as_paused():
        original_reset_runtime_state()
        module.state.local_vision_control_paused = True

    module.reset_runtime_state = reset_as_paused


legacy_tests.load_assistant = load_assistant_v2


def assistant_v2_target_point(module, config_id=None):
    if config_id is None:
        config_id = module.Task.SEARCH
    return module.build_object_target_point(config_id)


def test_assistant_main_v2_object_candidates_use_yolo_by_default() -> None:
    """辅车 main_v2 默认使用 YOLO 生成找物体候选."""

    module = load_assistant_v2()

    class FakeYoloTf:
        def __init__(self):
            self.loaded_paths = []
            self.detect_calls = []

        def load(self, path):
            self.loaded_paths.append(path)
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
            raise AssertionError("main_v2 找物体主线不应回退到色块识别")

    module.tf = FakeYoloTf()
    module.state.yolo_net = None
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    module.load_yolo_model()
    raw_yolo_candidates = module.yolo_detect(img)
    candidates = module.build_object_candidates(img, raw_yolo_candidates)

    assert module.tf.loaded_paths == [module.YOLO_MODEL_PATH]
    assert module.tf.detect_calls == [("fake-yolo-net", "detect-image")]
    assert img.copy_calls == [(module.YOLO_IMAGE_COPY_SCALE, 1)]
    assert candidates[0][0] == "red"
    assert candidates[0][1] == pytest.approx(160.0)
    assert candidates[0][3] == pytest.approx(210.0)
    assert candidates[0][4] == pytest.approx(3200.0)


def test_assistant_main_v2_yolo_detect_filters_small_area_candidates() -> None:
    """辅车 YOLO 结果面积过小时不应进入候选."""

    module = load_assistant_v2()

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


def test_assistant_main_v2_exposes_master_style_runtime_api() -> None:
    """辅车 main_v2 对外保留与主车一致的主流程入口名."""

    module = load_assistant_v2()

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
    assert tuple(inspect.signature(module.build_return_line_y_from_image).parameters) == (
        "img",
        "previous_line_y",
    )
    assert tuple(inspect.signature(module.build_object_candidates).parameters) == ("img", "yolo_candidates")
    assert tuple(inspect.signature(module._return_line_pixel_matches).parameters) == ("img", "x", "y")
    assert tuple(inspect.signature(module._return_line_has_horizontal_connected_at).parameters) == (
        "img",
        "x",
        "y",
        "required_connected",
    )
    assert tuple(inspect.signature(module._return_line_y_on_column).parameters) == ("img", "x")
    assert tuple(inspect.signature(module._build_return_line_y_from_pixels).parameters) == (
        "img",
        "previous_line_y",
    )
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
    assert tuple(inspect.signature(module.draw_object_tracking_debug).parameters) == ("img",)
    assert tuple(inspect.signature(module.draw_assistant_return_line_debug).parameters) == (
        "img",
        "line_y",
        "vx",
        "vy",
    )
    assert not hasattr(module, "format_vision_frame")
    assert not hasattr(module, "parse_sync_packet")
    assert not hasattr(module, "parse_ack_packet")
    assert not hasattr(module, "build_object_blob_candidates")
    assert not hasattr(module, "process_frame")
    assert not hasattr(module, "process_follow_frame")
    assert not hasattr(module, "process_object_frame")
    assert not hasattr(module, "process_return_line_frame")


def test_assistant_main_v2_build_object_candidates_matches_master_style_signature() -> None:
    """辅车 main_v2 的找物体候选入口应显式接收模型候选."""

    module = load_assistant_v2()

    with pytest.raises(TypeError):
        module.build_object_candidates(object())


def test_assistant_main_v2_object_task_config_keeps_only_filter_parameters() -> None:
    module = load_assistant_v2()

    assert module._object_task_config("tennis") == ("tennis", 3, 30, 70, 90, True)
    assert module._object_task_config("red") == ("red", 3, 30, 70, 90, True)
    assert module._object_task_config("blue") == ("blue", 3, 30, 70, 90, True)
    assert module._object_task_config("brown") == ("brown", 3, 30, 70, 90, True)
    assert module._object_task_config("white") == ("white", 3, 30, 70, 90, True)


def test_assistant_main_v2_object_task_config_accepts_legacy_threshold_layout() -> None:
    module = load_assistant_v2()
    module.OBJECT_TASKS = (("red", ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),)

    assert module._object_task_config("red") == ("red", 3, 30, 70, 90, True)


def test_assistant_main_v2_process_uart_input_matches_master_style_signature() -> None:
    """辅车 main_v2 的串口轮询入口应和主车一样只接受全局缓冲区."""

    module = load_assistant_v2()

    with pytest.raises(TypeError):
        module.process_uart_input(legacy_tests.FakeUART(), b"", module.state)


def test_assistant_main_v2_exposes_master_style_yolo_detect_api() -> None:
    """辅车 main_v2 提供与主车一致的 yolo_detect 入口."""

    module = load_assistant_v2()

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


def test_assistant_main_v2_run_applies_lens_correction_and_uses_yolo_detect_before_processing() -> None:
    """辅车 follow 模式运行循环应先做镜头校正, 且不进入物体 YOLO 链路."""

    module = load_assistant_v2()

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
        assert tuple(module.state.current_yolo_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()


def test_assistant_main_v2_run_applies_lens_correction_and_uses_yolo_preview_without_task_sync() -> None:
    """辅车调试模式下即使没有任务同步也应先跑预览识别."""

    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True
    logs = []
    module.print = lambda *args: logs.append(" ".join(str(arg) for arg in args))

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

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.process_uart_input = lambda rx_buffer: rx_buffer

    def fake_yolo_detect(img):
        assert img is image
        return [("red", 160.0, 30.0, 220.0, 400.0, FakeBlob())]

    def stop_after_frame(current_img):
        assert current_img is image
        assert module.state.current_sync is None
        assert module.state.current_image is image
        assert image.lens_corr_called is True
        assert tuple(candidate[:5] for candidate in module.state.current_yolo_candidates) == (
            ("red", 160.0, 30.0, 220.0, 400.0),
        )
        assert module.state.current_detection_source == "yolo"
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame
    module.write_reliable_line = lambda _frame_bytes: (_ for _ in ()).throw(
        AssertionError("上电调试预览不应发送底盘暂停控制")
    )

    with pytest.raises(StopLoop):
        module.run()

    assert module.state.current_second_total_frames == 1
    assert module.state.current_second_yolo_frames == 1
    assert any("[assistant_v2][boot]" in line for line in logs)
    assert any("debug=1" in line for line in logs)


def test_assistant_main_v2_run_pauses_for_approach_object_yolo() -> None:
    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True

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

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return []

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    uart = legacy_tests.FakeUART()
    module.sensor = Sensor()
    module.init_uart = lambda: uart
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)

    def prime_search_sync(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": module.State.APPROACH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.pack_task_arg(module.Task.SEARCH, 1),
            "threshold": (16, 51, 21, 84, -11, 52),
        }
        module.state.mode = module.RunMode.APPROACH_OBJECT
        module.state.track_task_name = "red"
        module.state.track_object_id = 1
        module.state.track_dynamic_threshold = (16, 51, 21, 84, -11, 52)
        return rx_buffer

    module.process_uart_input = prime_search_sync
    module.yolo_detect = lambda img: (_ for _ in ()).throw(AssertionError("未暂停控制前不应运行 YOLO"))

    def stop_after_frame(current_img):
        _ = current_img
        raise AssertionError("YOLO 暂停请求帧不应处理任务")

    module.process_task_frame = stop_after_frame

    def stop_after_control_frame(frame_bytes):
        uart.write(frame_bytes)
        raise StopLoop()

    module.write_reliable_line = stop_after_control_frame

    with pytest.raises(StopLoop):
        module.run()

    assert len(uart.writes) == 1
    frame = module.decode_frame(uart.writes[0])
    assert frame is not None
    assert frame["mode"] == module.Mode.TCP
    assert frame["topic"] == module.Topic.LOCAL_VISION_CONTROL
    assert frame["body"][0] == module.LocalVisionControl.PAUSE


def test_assistant_main_v2_approach_object_yolo_requests_resume() -> None:
    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True
    start_run_as_local_vision_paused(module)

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

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_rectangle(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_cross(self, *args, **kwargs):
            _ = (args, kwargs)

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return []

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    uart = legacy_tests.FakeUART()
    module.sensor = Sensor()
    module.init_uart = lambda: uart
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)

    def prime_search_sync(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": module.State.APPROACH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": module.pack_task_arg(module.Task.SEARCH, 1),
            "threshold": (16, 51, 21, 84, -11, 52),
        }
        module.state.mode = module.RunMode.APPROACH_OBJECT
        module.state.track_task_name = "red"
        module.state.track_object_id = 1
        module.state.track_dynamic_threshold = (16, 51, 21, 84, -11, 52)
        return rx_buffer

    module.process_uart_input = prime_search_sync

    def fake_yolo_detect(current_img):
        assert current_img is image
        return ()

    module.yolo_detect = fake_yolo_detect

    def stop_after_frame(current_img):
        assert current_img is image
        assert tuple(module.state.current_yolo_candidates) == ()
        assert module.state.current_detection_source == "yolo"
        raise StopLoop()

    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()

    assert len(uart.writes) == 1
    frame = module.decode_frame(uart.writes[0])
    assert frame is not None
    assert frame["mode"] == module.Mode.TCP
    assert frame["topic"] == module.Topic.LOCAL_VISION_CONTROL
    assert frame["body"][0] == module.LocalVisionControl.RESUME


def test_assistant_main_v2_run_skips_yolo_in_return_line_mode() -> None:
    """辅车回库黄线模式不应进入物体 YOLO 链路."""

    module = load_assistant_v2()

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

    def switch_to_return_line(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": module.State.RETURN_FOLLOW,
            "target": module.Target.NONE,
            "arg": legacy_tests.pack_task_arg(module.Task.RETURN_GARAGE_LINE, 0),
        }
        module.state.mode = module.RunMode.RETURN_LINE
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
    module.process_uart_input = switch_to_return_line

    def fake_yolo_detect(img):
        _ = img
        raise AssertionError("return line 模式不应调用 yolo_detect")

    def stop_after_frame(current_img):
        assert current_img is image
        assert image.lens_corr_called is True
        assert module.state.mode == module.RunMode.RETURN_LINE
        assert tuple(module.state.current_yolo_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()


def _seed_assistant_short_track(
    module,
    *,
    task_name="red",
    center_x=160.0,
    center_y=30.0,
    bottom_y=220.0,
    area=400.0,
    rect=(150, 20, 170, 40),
    vx=0.0,
    vy=0.0,
    frames_since_yolo=1,
    roi_failures=0,
):
    module.state.track_task_name = task_name
    module.state.track_center_x = float(center_x)
    module.state.track_center_y = float(center_y)
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


def test_assistant_main_v2_build_object_candidates_prefers_blob_tracking_between_yolo_frames() -> None:
    """辅车物体任务在短期跟踪有效时应优先使用传统候选."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(module)

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return [FakeBlob()]

        def copy(self, scale, copy_to_fb):
            _ = (scale, copy_to_fb)
            raise AssertionError("短期跟踪命中时不应复制图像做 YOLO")

    module.yolo_detect = lambda img: (_ for _ in ()).throw(
        AssertionError("短期跟踪命中时不应调用 yolo_detect")
    )
    img = FakeImage()

    candidates = module.build_object_candidates(img, ())

    assert len(candidates) == 1
    assert candidates[0][:5] == ("red", 160.0, 30.0, 220.0, 400.0)
    assert candidates[0][5].rect() == (150, 20, 20, 20)


def test_assistant_main_v2_build_object_candidates_fallbacks_to_yolo_in_approach_object() -> None:
    """辅车接近物体态传统候选失败达到阈值时允许回退到 YOLO."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 1
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(module)

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return []

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

    img = FakeImage()

    candidates = module.build_object_candidates(
        img,
        (("red", 160.0, 30.0, 220.0, 400.0, FakeBlob()),),
    )

    assert tuple(candidate[:5] for candidate in candidates) == (("red", 160.0, 30.0, 220.0, 400.0),)
    assert module.state.current_detection_source == "yolo"
    assert module.state.track_failure_reason == module.TrackFailureReason.NONE


def test_assistant_main_v2_build_object_candidates_keeps_predicted_target_before_yolo_fallback() -> None:
    """辅车传统候选首次失手且未到阈值时应保留预测目标一帧."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(module, center_x=158.0, center_y=28.0, bottom_y=218.0, vx=2.0, vy=3.0)

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return []

    module.yolo_detect = lambda img: (_ for _ in ()).throw(
        AssertionError("未到回退阈值前不应调用 yolo_detect")
    )
    img = FakeImage()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert candidates[0][:5] == ("red", 160.0, 31.0, 221.0, 400.0)


def test_assistant_main_v2_predict_frame_does_not_create_event() -> None:
    """辅车预测帧可以继续输出控制, 但不能触发可靠事件."""

    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    module.state.required_stable_frames = 1

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

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

    module.state.current_detection_source = "predict"
    module.state.current_yolo_candidates = ()
    module.state.current_object_candidates = (
        ("red", 160.0, 30.0, 220.0, 400.0, FakeBlob()),
    )
    module.state.uart_device = legacy_tests.FakeUART()
    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()

    module.process_task_frame(img)

    assert module.state.has_pending_event() is False


def test_assistant_main_v2_build_object_observation_and_candidates_uses_current_object_candidates() -> None:
    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_v2_target_point(module)
    blob = module.YoloDetectionBlob(
        target_x - 10.0,
        legacy_tests.IMAGE_HEIGHT - target_y,
        target_x + 10.0,
        legacy_tests.IMAGE_HEIGHT - target_y + 20.0,
        1,
        0.95,
    )
    module.state.current_yolo_candidates = ()
    module.state.current_object_candidates = (("red", target_x, 30.0, target_y, 400.0, blob),)

    observation, best_blob, task_name, candidates = module.build_object_observation_and_candidates()

    assert observation == (0.0, 0.0, 400.0)
    assert best_blob is blob
    assert task_name == "red"
    assert candidates == [("red", target_x, 30.0, target_y, 400.0, blob)]


def test_assistant_main_v2_runs_yolo_after_entering_transport_align() -> None:
    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    assert module.should_run_yolo_for_current_frame() is True


def test_assistant_main_v2_approach_object_without_master_threshold_requests_yolo() -> None:
    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
            threshold=(0, 0, 0, 0, 0, 0),
        )
    )

    assert module.should_run_yolo_for_current_frame() is True


def test_assistant_main_v2_runs_yolo_once_after_entering_transport_object() -> None:
    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    assert module.should_run_yolo_for_current_frame() is True

    module.state.entry_yolo_pending = False

    assert module.should_run_yolo_for_current_frame() is False


def test_assistant_main_v2_runs_yolo_once_after_entering_orbit() -> None:
    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.ORBIT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.ORBIT, 1),
        )
    )

    assert module.should_run_yolo_for_current_frame() is True

    module.state.entry_yolo_pending = False

    assert module.should_run_yolo_for_current_frame() is False


def test_assistant_main_v2_run_skips_yolo_when_blob_tracking_is_active() -> None:
    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3

    class StopLoop(Exception):
        pass

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

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

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [FakeBlob()]

        def draw_cross(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_rectangle(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    def prime_object_tracking(rx_buffer):
        module.state.current_sync = {
            "reliable_seq": 12,
            "state": module.State.APPROACH_OBJECT,
            "target": module.Target.OBJECT,
            "arg": legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        }
        module.state.mode = module.RunMode.APPROACH_OBJECT
        _seed_assistant_short_track(module)
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
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
        ("red", 160.0, 30.0, 220.0, 400.0, module.state.current_object_candidates[0][5]),
    )
    assert module.state.current_detection_source == "roi"
    assert yolo_call_count == 0


def test_assistant_main_v2_debug_preview_uses_blob_tracking_over_cached_yolo_without_task_sync() -> None:
    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True

    class PreviewBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class SnapshotImage:
        def __init__(self):
            self.flush_count = 0

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [PreviewBlob()]

        def draw_cross(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_rectangle(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

        def flush(self):
            self.flush_count += 1

    image = SnapshotImage()
    module.state.uart_device = legacy_tests.FakeUART()
    module.state.current_image = image
    module.state.current_image_width = image.width()
    module.state.current_image_height = image.height()
    _seed_assistant_short_track(module)
    yolo_blob = module.YoloDetectionBlob(150.0, 20.0, 170.0, 40.0, 1, 0.95)
    module.state.current_detection_source = "yolo"
    module.state.current_yolo_candidates = (("red", 160.0, 30.0, 220.0, 400.0, yolo_blob),)
    module.state.current_object_candidates = tuple(module.build_object_candidates(image, ()))

    module.process_task_frame(image)

    assert tuple(module.state.current_yolo_candidates[:1])[0][:5] == ("red", 160.0, 30.0, 220.0, 400.0)
    assert tuple(module.state.current_object_candidates[:1])[0][:5] == ("red", 160.0, 30.0, 220.0, 400.0)
    assert module.state.current_detection_source == "roi"
    assert module.state.track_source == "roi"
    assert image.flush_count == 1
    assert module.state.uart_device.writes == []


def test_assistant_main_v2_debug_preview_handles_empty_candidates_without_crashing() -> None:
    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True
    logs = []
    module.print = lambda *args: logs.append(" ".join(str(arg) for arg in args))

    class SnapshotImage:
        def __init__(self):
            self.flush_count = 0

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_rectangle(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

        def flush(self):
            self.flush_count += 1

    image = SnapshotImage()
    module.state.uart_device = legacy_tests.FakeUART()
    module.state.current_sync = None
    module.state.current_image = image
    module.state.current_image_width = image.width()
    module.state.current_image_height = image.height()
    module.state.current_detection_source = "miss"
    module.state.current_object_candidates = ()

    module.process_task_frame(image)

    assert image.flush_count == 1
    assert any("[assistant_v2][preview]" in line for line in logs)
    assert any("cand=0" in line for line in logs)


def test_assistant_main_v2_run_skips_yolo_in_debug_preview_when_blob_tracking_is_active() -> None:
    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True

    class StopLoop(Exception):
        pass

    class PreviewBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class SnapshotImage:
        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, roi=None, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, roi, margin)
            return [PreviewBlob()]

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def lens_corr(self, strength, zoom):
            _ = (strength, zoom)
            return self

        def draw_cross(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_rectangle(self, *args, **kwargs):
            _ = (args, kwargs)

        def draw_string(self, *args, **kwargs):
            _ = (args, kwargs)

        def flush(self):
            raise StopLoop()

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    def prime_preview_tracking(rx_buffer):
        _seed_assistant_short_track(module)
        return rx_buffer

    module.sensor = Sensor()
    module.init_uart = lambda: legacy_tests.FakeUART()
    module.init_sensor = lambda: (legacy_tests.IMAGE_WIDTH, legacy_tests.IMAGE_HEIGHT)
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
    assert tuple(module.state.current_object_candidates) == (
        ("red", 160.0, 30.0, 220.0, 400.0, module.state.current_object_candidates[0][5]),
    )
    assert module.state.current_detection_source == "roi"
    assert yolo_call_count == 0


def test_assistant_main_v2_approach_object_uses_yolo_relocation_candidate_in_tracking_window() -> None:
    """辅车接近物体态达到重定位条件后可使用跟踪窗口内的 YOLO 候选."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(
        module,
        center_x=100.0,
        center_y=30.0,
        bottom_y=220.0,
        frames_since_yolo=3,
    )

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

    class FakeBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

    img = FakeImage()
    candidates = module.build_object_candidates(
        img,
        [
            ("red", 160.0, 30.0, 220.0, 400.0, FakeBlob(150, 20, 20, 20)),
            ("red", 100.0, 30.0, 220.0, 400.0, FakeBlob(90, 20, 20, 20)),
        ],
    )

    assert tuple(candidate[:5] for candidate in candidates) == (("red", 100.0, 30.0, 220.0, 400.0),)
    assert module.state.current_detection_source == "yolo"
    assert module.state.track_failure_reason == module.TrackFailureReason.NONE


def test_assistant_main_v2_yolo_relocation_position_jump_candidate_is_ignored() -> None:
    """辅车不使用位置跳变的 YOLO 重定位候选."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(
        module,
        center_x=100.0,
        center_y=30.0,
        bottom_y=220.0,
        rect=(90, 20, 110, 40),
        frames_since_yolo=3,
    )

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (220, 20, 20, 20)

    img = FakeImage()
    candidates = module.build_object_candidates(img, [("red", 230.0, 30.0, 220.0, 400.0, FakeBlob())])

    assert candidates == ()
    assert module.state.current_detection_source == "yolo"
    assert module.state.track_failure_reason == module.TrackFailureReason.NO_CANDIDATE


def test_assistant_main_v2_yolo_relocation_area_jump_candidate_is_ignored() -> None:
    """辅车不使用面积突变的 YOLO 重定位候选."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(
        module,
        area=400.0,
        rect=(150, 20, 170, 40),
        frames_since_yolo=3,
    )

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (140, 10, 40, 40)

    img = FakeImage()
    candidates = module.build_object_candidates(img, [("red", 160.0, 30.0, 220.0, 1600.0, FakeBlob())])

    assert candidates == ()
    assert module.state.current_detection_source == "yolo"
    assert module.state.track_failure_reason == module.TrackFailureReason.NO_CANDIDATE


def test_assistant_main_v2_track_state_records_object_id_and_confidence() -> None:
    """辅车短期跟踪状态应单独记录目标编号和可信度."""

    module = load_assistant_v2()
    module.state.current_image_height = legacy_tests.IMAGE_HEIGHT

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

    module.remember_object_tracking("red", FakeBlob(), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_object_id == 1
    assert module.state.track_confidence == 80


def test_assistant_main_v2_debug_counters_roll_per_second() -> None:
    """辅车应滚动记录每秒来源统计、ROI 成功情况和回退次数."""

    module = load_assistant_v2()

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


def test_assistant_main_v2_rejects_blob_candidate_outside_tracking_window() -> None:
    """辅车传统候选偏离预测窗口时应拒绝更新并转入预测帧."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(module)

    class FarBlob:
        def rect(self):
            return (220, 20, 20, 20)

        def cx(self):
            return 230.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return [FarBlob()]

    module.yolo_detect = lambda img: (_ for _ in ()).throw(
        AssertionError("越界候选首次失手时不应立刻回退到 yolo_detect")
    )
    img = FakeImage()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert module.state.track_failure_reason == module.TrackFailureReason.OUT_OF_WINDOW
    assert candidates[0][:5] == ("red", 160.0, 30.0, 220.0, 400.0)


def test_assistant_main_v2_rejects_blob_candidate_with_large_area_jump() -> None:
    """辅车传统候选面积突变时应拒绝更新并转入预测帧."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    _seed_assistant_short_track(module, area=400.0)

    class LargeBlob:
        def rect(self):
            return (140, 10, 40, 40)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 1600.0

    class FakeImage:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            return [LargeBlob()]

    module.yolo_detect = lambda img: (_ for _ in ()).throw(
        AssertionError("面积突变首次失手时不应立刻回退到 yolo_detect")
    )
    img = FakeImage()

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "predict"
    assert module.state.track_roi_failure_frames == 1
    assert module.state.track_failure_reason == module.TrackFailureReason.AREA_JUMP
    assert candidates[0][:5] == ("red", 160.0, 30.0, 220.0, 400.0)


def test_assistant_main_v2_builds_dynamic_threshold_from_yolo_and_uses_it_for_roi() -> None:
    """辅车在 YOLO 命中后应建立动态阈值并在 ROI 帧复用."""

    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = calibration_img
    module.state.current_image_width = calibration_img.width()
    module.state.current_image_height = calibration_img.height()
    module.remember_object_tracking("red", YoloBlob(), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold is not None

    class Blob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 180.0

    threshold = module.state.track_dynamic_threshold
    img = DynamicThresholdRoiImage(threshold, [Blob()])

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "roi"
    assert candidates[0][:5] == ("red", 160.0, 30.0, 220.0, 180.0)
    assert img.find_blobs_calls[0][0] == tuple(threshold)


def test_assistant_main_v2_yolo_calibration_uses_foreground_area_for_tracking() -> None:
    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 2
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = calibration_img
    module.state.current_image_width = calibration_img.width()
    module.state.current_image_height = calibration_img.height()
    module.remember_object_tracking("red", YoloBlob(), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold is not None
    assert module.state.track_area < 300.0

    class Blob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 180.0

    threshold = module.state.track_dynamic_threshold
    img = DynamicThresholdRoiImage(threshold, [Blob()])

    candidates = module.build_object_candidates(img, ())

    assert module.state.current_detection_source == "roi"
    assert module.state.track_failure_reason == module.TrackFailureReason.NONE
    assert candidates[0][:5] == ("red", 160.0, 30.0, 220.0, 180.0)


def test_assistant_main_v2_reuses_dynamic_threshold_when_center_stays_stable() -> None:
    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def cy(self):
            _left, top, _width, height = self._rect
            return float(top) + float(height) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = first_img
    module.state.current_image_width = first_img.width()
    module.state.current_image_height = first_img.height()
    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation

    second_img = DynamicThresholdCalibrationImage(
        (152, 22, 170, 40),
        (30, 40, 20),
        (60, 10, 10),
    )
    module.state.current_image = second_img
    module.state.current_image_width = second_img.width()
    module.state.current_image_height = second_img.height()
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: (_ for _ in ()).throw(
        AssertionError("中心采样稳定时不应重算阈值")
    )

    module.remember_object_tracking("red", YoloBlob(152, 22, 18, 18), 161.0, 31.0, 218.0, 324.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation


def test_assistant_main_v2_refreshes_dynamic_threshold_after_consecutive_unhealthy_yolo_frames() -> None:
    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def cy(self):
            _left, top, _width, height = self._rect
            return float(top) + float(height) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = first_img
    module.state.current_image_width = first_img.width()
    module.state.current_image_height = first_img.height()
    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation
    new_threshold = (85, 95, -5, 5, -5, 5)
    build_calls = []

    def fake_build_dynamic_threshold(_img, _blob):
        build_calls.append(1)
        return new_threshold

    module._build_dynamic_threshold_for_blob = fake_build_dynamic_threshold

    second_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (90, 0, 0),
        (80, 0, 0),
    )
    module.state.current_image = second_img
    module.state.current_image_width = second_img.width()
    module.state.current_image_height = second_img.height()

    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_dynamic_threshold_health_failures == 1
    assert module.state.track_pending_dynamic_threshold is None
    assert build_calls == []

    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_pending_dynamic_threshold == new_threshold
    assert module.state.track_pending_dynamic_threshold_ok_frames == 1
    assert len(build_calls) == 1

    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == new_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation + 1
    assert module.state.track_dynamic_threshold_health_failures == 0


def test_assistant_main_v2_keeps_dynamic_threshold_in_non_search_state() -> None:
    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.DYNAMIC_THRESHOLD_REFRESH_FAILURE_FRAMES = 1
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.ORBIT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def cy(self):
            _left, top, _width, height = self._rect
            return float(top) + float(height) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = first_img
    module.state.current_image_width = first_img.width()
    module.state.current_image_height = first_img.height()
    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    old_threshold = module.state.track_dynamic_threshold
    old_generation = module.state.track_dynamic_threshold_generation
    second_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (90, 0, 0),
        (80, 0, 0),
    )
    module.state.current_image = second_img
    module.state.current_image_width = second_img.width()
    module.state.current_image_height = second_img.height()
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: (_ for _ in ()).throw(
        AssertionError("非寻找态不应重复计算动态阈值")
    )

    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == old_threshold
    assert module.state.track_dynamic_threshold_generation == old_generation
    assert module.state.track_dynamic_threshold_health_failures == 0
    assert module.state.track_pending_dynamic_threshold is None


def test_assistant_main_v2_recomputes_dynamic_threshold_after_track_reset() -> None:
    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            left, _top, width, _height = self._rect
            return float(left) + float(width) / 2.0

        def cy(self):
            _left, top, _width, height = self._rect
            return float(top) + float(height) / 2.0

        def area(self):
            _left, _top, width, height = self._rect
            return float(width) * float(height)

    first_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
    )
    module.state.current_image = first_img
    module.state.current_image_width = first_img.width()
    module.state.current_image_height = first_img.height()
    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    new_threshold = (85, 95, -5, 5, -5, 5)
    module.state.clear_track()
    second_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (90, 0, 0),
        (80, 0, 0),
    )
    module.state.current_image = second_img
    module.state.current_image_width = second_img.width()
    module.state.current_image_height = second_img.height()
    module._build_dynamic_threshold_for_blob = lambda _img, _blob: new_threshold

    module.remember_object_tracking("red", YoloBlob(150, 20, 20, 20), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold == new_threshold
    assert module.state.track_dynamic_threshold_generation == 1


def test_assistant_main_v2_dynamic_threshold_keeps_center_connected_component_only() -> None:
    """辅车动态阈值反推只使用与中心采样区连通的前景."""

    module = load_assistant_v2()
    module.image.rgb_to_lab = lambda pixel: pixel

    class YoloBlob:
        def rect(self):
            return (150, 20, 20, 20)

    calibration_img = DynamicThresholdCalibrationImage(
        (150, 20, 170, 40),
        (30, 40, 20),
        (80, 0, 0),
        fragment=(150, 20, 155, 25, (32, 75, 60)),
    )
    module.state.current_image = calibration_img
    module.state.current_image_width = calibration_img.width()
    module.state.current_image_height = calibration_img.height()

    threshold = module._build_dynamic_threshold_for_blob(calibration_img, YoloBlob())

    assert threshold is not None
    assert threshold[3] < 75
    assert threshold[5] < 60


def test_assistant_main_v2_failed_yolo_calibration_disables_roi_tracking() -> None:
    """辅车未建立动态阈值时不应继续 ROI 跟踪."""

    module = load_assistant_v2()
    module.ROI_TRACKING_MAX_FRAMES = 3
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class YoloBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class ImageWithoutPixels:
        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

    img = ImageWithoutPixels()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.remember_object_tracking("red", YoloBlob(), 160.0, 30.0, 220.0, 400.0, "yolo")

    assert module.state.track_dynamic_threshold is None
    assert module.should_use_blob_tracking() is False
    assert module.build_object_candidates(img, ()) == ()


def test_assistant_main_v2_process_task_frame_uses_cached_yolo_candidates() -> None:
    """辅车 main_v2 的全局 task 入口应消费预先缓存的 YOLO 候选."""

    module = load_assistant_v2()
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_v2_target_point(module)

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

    state.current_yolo_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )
    state.current_object_candidates = tuple(state.current_yolo_candidates)

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


def test_assistant_main_v2_handle_control_frame_reuses_global_state() -> None:
    """辅车 main_v2 可像主车一样通过全局入口处理控制帧."""

    module = load_assistant_v2()

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


def test_assistant_main_v2_parse_task_sync_packet_matches_master_style_name() -> None:
    """辅车 main_v2 使用主车同名入口解析本地任务同步帧."""

    module = load_assistant_v2()

    threshold = (12, 80, -30, 40, -20, 60)
    packet = module.parse_task_sync_packet(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 2),
            threshold,
        )
    )

    assert packet == {
        "reliable_seq": 12,
        "state": module.State.APPROACH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": legacy_tests.pack_task_arg(module.Task.SEARCH, 2),
        "threshold": threshold,
    }


def test_assistant_main_v2_uses_synced_threshold_without_yolo() -> None:
    module = load_assistant_v2()
    threshold = (12, 80, -30, 40, -20, 60)
    reply = module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 2),
            threshold,
        )
    )

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class FakeImage:
        def __init__(self):
            self.find_blobs_calls = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0, roi=None):
            _ = (pixels_threshold, area_threshold, merge, margin, roi)
            self.find_blobs_calls.append(tuple(thresholds[0]))
            return [FakeBlob()]

    legacy_tests.assert_assistant_ack(module, reply, 12)
    module.yolo_detect = lambda img: (_ for _ in ()).throw(
        AssertionError("同步阈值后辅车不应调用 YOLO")
    )
    img = FakeImage()

    candidates = module.build_object_candidates(img, ())

    assert img.find_blobs_calls == [threshold]
    assert tuple(candidate[:5] for candidate in candidates) == (
        ("blue", 160.0, 30.0, 220.0, 400.0),
    )
    assert module.state.current_detection_source == "sync_threshold"


def test_assistant_main_v2_build_search_velocity_wrapper_matches_object_path() -> None:
    """辅车 main_v2 的主车式搜索速度入口仍复用找物体速度逻辑."""

    module = load_assistant_v2()
    observation = (20.0, -30.0, 300.0)

    module.state.current_image_height = legacy_tests.IMAGE_HEIGHT
    assert module.build_search_velocity_from_observation(
        observation,
    ) == module.build_object_approach_velocity_from_observation(observation)


def test_assistant_main_v2_object_target_helpers_drop_unused_size_parameters() -> None:
    """辅车 main_v2 的目标点与观测 helper 不再暴露无意义尺寸参数."""

    module = load_assistant_v2()

    assert tuple(inspect.signature(module.build_object_target_point).parameters) == ("config_id",)
    assert tuple(inspect.signature(module.build_object_observation).parameters) == (
        "valid",
        "center_x",
        "bottom_y",
        "area",
    )
    assert tuple(inspect.signature(module.build_object_observation_and_candidates).parameters) == ()


def test_assistant_main_v2_process_task_frame_uses_global_object_pipeline() -> None:
    """辅车 main_v2 的全局 task 入口应像主车一样消费全局候选和全局状态."""

    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_v2_target_point(module)

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
    module.state.current_yolo_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.process_task_frame(img)

    legacy_tests.assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)


def test_assistant_main_v2_process_task_frame_calls_master_style_velocity_wrapper() -> None:
    """辅车 main_v2 的全局 task 入口应走主车式速度包装入口."""

    module = load_assistant_v2()
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_v2_target_point(module)

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
    module.state.current_yolo_candidates = (
        ("red", target_x, legacy_tests.IMAGE_HEIGHT - target_y + 10, target_y, 300.0, FakeBlob()),
    )
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)
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


def test_assistant_main_v2_debug_display_draws_tracking_state() -> None:
    """辅车调试显示应绘制跟踪来源、失败原因和刷新图像."""

    module = load_assistant_v2()
    module.ASSISTANT_DEBUG_DISPLAY_ENABLED = True
    module.handle_control_frame(
        legacy_tests.assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            legacy_tests.pack_task_arg(module.Task.SEARCH, 1),
        )
    )

    class FakeBlob:
        def rect(self):
            return (150, 20, 20, 20)

        def cx(self):
            return 160.0

        def cy(self):
            return 30.0

        def area(self):
            return 400.0

    class FakeImage:
        def __init__(self):
            self.rectangles = []
            self.strings = []
            self.crosses = []
            self.flush_count = 0

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def draw_cross(self, x, y, color=None):
            self.crosses.append((x, y, color))

        def draw_rectangle(self, *args, **kwargs):
            self.rectangles.append((args, kwargs))

        def draw_string(self, *args, **kwargs):
            self.strings.append((args, kwargs))

        def flush(self):
            self.flush_count += 1

    module.state.track_predicted_center_x = 160.0
    module.state.track_predicted_bottom_y = 220.0
    module.state.track_predicted_roi = (140.0, 10.0, 180.0, 50.0)
    module.state.track_confidence = 80
    module.state.track_object_id = 1
    module.state.current_detection_source = "yolo"
    module.state.current_yolo_candidates = (
        ("red", 160.0, 30.0, 220.0, 400.0, FakeBlob()),
    )
    module.state.current_object_candidates = tuple(module.state.current_yolo_candidates)
    module.state.uart_device = legacy_tests.FakeUART()

    def fake_build_search_velocity_from_observation(observation):
        _ = observation
        return 0.0, 0.0

    module.build_search_velocity_from_observation = fake_build_search_velocity_from_observation

    img = FakeImage()
    module.state.current_image = img
    module.state.current_image_width = img.width()
    module.state.current_image_height = img.height()
    module.process_task_frame(img)

    assert img.rectangles
    assert any("src=" in args[2] for args, _kwargs in img.strings if len(args) >= 3)
    assert any("fail=" in args[2] for args, _kwargs in img.strings if len(args) >= 3)
    assert any("roi ok=" in args[2] for args, _kwargs in img.strings if len(args) >= 3)
    assert img.flush_count == 1


def test_assistant_main_v2_single_arg_process_uart_input_uses_global_state() -> None:
    """辅车 main_v2 的单参串口轮询入口应直接驱动全局状态机."""

    module = load_assistant_v2()
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


def test_assistant_main_v2_single_arg_process_uart_input_calls_master_style_handle_control_frame() -> None:
    """辅车 main_v2 的单参串口轮询入口应走全局控制包入口."""

    module = load_assistant_v2()
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


def test_assistant_main_v2_process_uart_input_without_any_matches_master_behavior() -> None:
    """辅车 main_v2 不再为缺失 any 的串口对象静默返回."""

    module = load_assistant_v2()
    module.state.uart_device = object()

    with pytest.raises(AttributeError):
        module.process_uart_input(b"")


def test_assistant_main_v2_process_uart_input_read_error_propagates() -> None:
    """辅车 main_v2 不再吞掉串口读取异常."""

    module = load_assistant_v2()

    class BrokenUART:
        def any(self):
            return 1

        def read(self, size):
            _ = size
            raise RuntimeError("uart read failed")

    module.state.uart_device = BrokenUART()

    with pytest.raises(RuntimeError, match="uart read failed"):
        module.process_uart_input(b"")


def test_assistant_main_v2_event_helpers_reflect_global_state() -> None:
    """辅车 main_v2 的主车式事件 helper 应直接反映全局状态."""

    module = load_assistant_v2()
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

    module = load_assistant_v2()

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
        assert tuple(module.state.current_yolo_candidates) == ()
        raise StopLoop()

    module.yolo_detect = fake_yolo_detect
    module.process_task_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()
