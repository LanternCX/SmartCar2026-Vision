"""辅车视觉找物体模式单元测试."""

# pyright: reportAttributeAccessIssue=false

import pytest

from tests.test_support import assistant_event_ack_frame, assistant_sync_frame, load_main_module


class FakeUART:
    """记录串口写入内容的测试桩."""

    def __init__(self, incoming=b""):
        self.incoming = incoming
        self.writes = []

    def any(self):
        return len(self.incoming)

    def read(self, size):
        data = self.incoming[:size]
        self.incoming = self.incoming[size:]
        return data

    def write(self, data):
        self.writes.append(data)
        return len(data)


class BadReadUART:
    """模拟输入字节无法解码的测试桩."""

    def any(self):
        return 1

    def read(self, size):
        return b"\xff"


class FakeYoloTf:
    """记录 YOLO 加载与检测调用的测试桩."""

    def __init__(self):
        self.load_calls = []
        self.detect_calls = []

    def load(self, path, load_to_fb=False):
        self.load_calls.append((path, load_to_fb))
        return "fake-yolo-net"

    def detect(self, net, img):
        self.detect_calls.append((net, img))
        return [(0.25, 0.7916666667, 0.75, 0.875, 1, 0.95)]


def load_assistant():
    """加载辅车视觉入口模块."""

    return load_main_module("assistant_object_approach_test_module")


def test_assistant_exposes_grouped_enum_constants() -> None:
    module = load_assistant()

    assert module.Mode.UDP == 0x01
    assert module.Topic.ASSISTANT_VISION_EVENT_REPORT == 0x13
    assert module.RunMode.FOLLOW == "follow"
    assert module.State.APPROACH_OBJECT == 2
    assert module.Target.OBJECT == 1
    assert module.Task.SEARCH == 1
    assert module.Event.TARGET_FOUND == 6
    assert not hasattr(module, "MODE_UDP")
    assert not hasattr(module, "TOPIC_ASSISTANT_VISION_EVENT_REPORT")
    assert not hasattr(module, "MODE_FOLLOW")
    assert not hasattr(module, "STATE_APPROACH_OBJECT")
    assert not hasattr(module, "TARGET_OBJECT")
    assert not hasattr(module, "OBJECT_APPROACH_CONFIG_ID")
    assert not hasattr(module, "EVENT_TARGET_FOUND")
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def pack_task_arg(config_id, object_id):
    packed = (int(config_id) & 0xFF) | ((int(object_id) & 0xFF) << 8)
    if packed >= 0x8000:
        packed -= 0x10000
    return packed


def assistant_target_point(module, config_id=None):
    """返回当前指定配置的找物体目标点."""

    if config_id is None:
        config_id = module.Task.SEARCH
    return module.build_object_target_point(IMAGE_WIDTH, IMAGE_HEIGHT, config_id)


def centered_object_observation(module, area):
    """构造命中当前目标点的观测."""

    target_x, target_y = assistant_target_point(module)
    return module.build_object_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def centered_transport_observation(module, area):
    """构造命中搬运入口目标点的观测."""

    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )
    return module.build_object_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.TRANSPORT,
    )


def centered_orbit_observation(module, area):
    """构造命中绕行修正目标点的观测."""

    target_x, target_y = assistant_target_point(
        module,
        module.Task.ORBIT,
    )
    return module.build_object_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.ORBIT,
    )


def aligned_object_image(module, config_id=None, area=300):
    """构造底边对齐当前目标点的测试图像."""

    target_x, target_y = assistant_target_point(module, config_id)

    class AlignedBlob:
        def rect(self):
            return (target_x - 10, IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10

        def area(self):
            return area

        def min_corners(self):
            left, top, width, height = self.rect()
            right = left + width
            bottom = top + height
            return ((left, top), (right, top), (right, bottom), (left, bottom))

    class AlignedImage:
        def __init__(self):
            self.crosses = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [AlignedBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    return AlignedImage()


def choose_outside_deadzone_offset(target, upper_bound, deadzone, clearance=1.0):
    """在图像范围内构造一个稳定超出死区的偏移量."""

    required = float(deadzone) + float(clearance)
    target = float(target)
    upper_bound = float(upper_bound)
    if upper_bound - target >= required:
        return required
    if target >= required:
        return -required
    raise AssertionError("configured target leaves no room for out-of-deadzone sample")


def expected_axis_velocity(error, kp, min_speed, limit):
    """按当前参数计算单轴期望速度."""

    value = float(error) * float(kp)
    limit = abs(float(limit))
    min_speed = min(abs(float(min_speed)), limit)
    if value > limit:
        value = limit
    if value < -limit:
        value = -limit
    if value == 0.0:
        return 0.0
    if 0.0 < value < min_speed:
        return min_speed
    if -min_speed < value < 0.0:
        return -min_speed
    return value


def expected_object_y_velocity(module, err_y):
    """按当前参数计算纵向找物体期望速度."""

    err_y = float(err_y)
    if abs(err_y) <= float(module.OBJECT_APPROACH_DEADZONE_Y_PX):
        return 0.0
    return expected_axis_velocity(
        err_y,
        module.OBJECT_APPROACH_KP_Y,
        module.OBJECT_APPROACH_MIN_SPEED,
        module.OBJECT_APPROACH_MAX_VY,
    )


def assert_assistant_ack(module, frame_bytes, seq):
    """断言辅车 ACK 帧."""

    assert frame_bytes == module.format_ack_frame(seq)


def assert_assistant_event(module, frame_bytes, seq, event, value):
    """断言辅车事件帧."""

    assert frame_bytes == module.format_event_frame(seq, event, value)


def assert_velocity_frame(module, frame_bytes, vx, vy):
    """断言辅车速度帧."""

    frame = module.decode_frame(frame_bytes)
    assert frame is not None
    assert frame["mode"] == module.Mode.UDP
    assert frame["topic"] == module.Topic.LOCAL_VISION_VELOCITY
    body = module.decode_velocity_body(frame["body"])
    assert body["vx"] == pytest.approx(vx)
    assert body["vy"] == pytest.approx(vy)
    assert body["omega"] == pytest.approx(0.0)
    assert body["has_omega"] is False


def test_assistant_defaults_to_follow_mode() -> None:
    """默认启动模式仍然是 follow."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert state.mode == module.RunMode.FOLLOW


def test_assistant_sync_packet_switches_to_object_mode_and_replies_ack() -> None:
    """本地同步包切换到找物体模式时必须回复 ACK."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(
                12,
                module.State.APPROACH_OBJECT,
                module.Target.OBJECT,
                pack_task_arg(1, 2),
            )
        ),
        12,
    )
    assert state.mode == module.RunMode.APPROACH_OBJECT
    assert state.current_object_config_id() == 1


def test_assistant_sync_packet_exposes_selected_object_id() -> None:
    module = load_assistant()
    state = module.AssistantVisionState()

    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(1, 2),
        )
    )

    assert state.current_object_id() == 2


def test_assistant_packed_approach_sync_still_emits_target_found_event() -> None:
    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=1)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.SEARCH, 2),
        )
    )

    state.accept_object_observation(centered_object_observation(module, 180))

    assert_assistant_event(
        module,
        state.next_event_frame(),
        12,
        module.Event.TARGET_FOUND,
        180,
    )


def test_assistant_packed_transport_sync_still_emits_aligned_event() -> None:
    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=1)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 2),
        )
    )

    state.accept_object_observation(centered_transport_observation(module, 180))

    assert_assistant_event(
        module,
        state.next_event_frame(),
        12,
        module.Event.ALIGNED,
        180,
    )


def test_assistant_sync_packet_switches_to_orbit_correction_mode() -> None:
    """绕行同步包切换到绕行修正模式并回复 ACK."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(
                12,
                module.State.ORBIT,
                module.Target.OBJECT,
                pack_task_arg(module.Task.ORBIT, 2),
            )
        ),
        12,
    )
    assert state.mode == module.RunMode.ORBIT_OBJECT


def test_process_object_frame_filters_candidates_by_selected_object_id() -> None:
    module = load_assistant()
    module.OBJECT_TASKS = (
        ("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),
        ("brown", ((7, 8, 9, 10, 11, 12),), 0, 1, 1, True),
    )
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.SEARCH, 2),
        )
    )

    target_x, target_y = assistant_target_point(module, module.Task.SEARCH)

    class RedBlob:
        def rect(self):
            return (target_x - 10, IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300

    class BrownBlob:
        def rect(self):
            return (target_x - 10, IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300

    class MixedImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            if thresholds == [(1, 2, 3, 4, 5, 6)]:
                return [RedBlob()]
            if thresholds == [(7, 8, 9, 10, 11, 12)]:
                return [BrownBlob()]
            return []

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

    uart = FakeUART()
    module.process_object_frame(uart, state, MixedImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    assert_velocity_frame(
        module,
        uart.writes[0],
        0.0,
        0.0,
    )
    assert state.current_object_config_id() == module.Task.SEARCH
    assert state.current_object_id() == 2


def test_process_object_frame_prefers_target_window_candidate_in_transport() -> None:
    """辅车推行阶段优先选择当前目标窗口内的候选框."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    class FakeBlob:
        def __init__(self, center_x, center_y, bottom_y, area):
            self._center_x = center_x
            self._center_y = center_y
            self._bottom_y = bottom_y
            self._area = area

        def rect(self):
            width = 20
            height = 20
            top = IMAGE_HEIGHT - self._bottom_y
            return (self._center_x - width / 2, top, width, height)

        def cx(self):
            return self._center_x

        def cy(self):
            return self._center_y

        def area(self):
            return self._area

    class FakeImage:
        def __init__(self, blobs):
            self._blobs = blobs

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return self._blobs

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

    uart = FakeUART()

    module.process_object_frame(
        uart,
        state,
        FakeImage(
            [
                FakeBlob(target_x - 30.0, 53, target_y - 35.0, 300),
                FakeBlob(target_x, 90, target_y, 300),
            ]
        ),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert_velocity_frame(
        module,
        uart.writes[0],
        0.0,
        0.0,
    )


def test_process_object_frame_transport_alignment_keeps_original_candidate_selection() -> None:
    """辅车搬运前对正阶段不使用目标窗口强过滤."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    class FakeBlob:
        def rect(self):
            bottom_y = target_y - float(module.OBJECT_Y_TOLERANCE_PX) - 20.0
            return (target_x - 10.0, IMAGE_HEIGHT - bottom_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            bottom_y = target_y - float(module.OBJECT_Y_TOLERANCE_PX) - 20.0
            return IMAGE_HEIGHT - bottom_y + 10.0

        def area(self):
            return 300

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return [FakeBlob()]

        def draw_cross(self, x, y, color=None):
            _ = (x, y, color)

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

        def draw_string(self, x, y, text, color=None):
            _ = (x, y, text, color)

    uart = FakeUART()

    module.process_object_frame(uart, state, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    frame = module.decode_frame(uart.writes[0])
    assert frame is not None
    body = module.decode_velocity_body(frame["body"])
    assert body["vy"] != pytest.approx(module.OBJECT_MISSING_SEARCH_VY)


def test_process_object_frame_marks_selected_object_and_target_point() -> None:
    """辅车找物体调试画面标出当前选中目标与当前目标点."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.SEARCH, 1),
        )
    )
    target_x, target_y = assistant_target_point(module)

    class FakeBlob:
        def rect(self):
            return (target_x - 10, IMAGE_HEIGHT - target_y, 20, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 300

    class FakeImage:
        def __init__(self):
            self.crosses = []
            self.labels = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return [FakeBlob()]

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

        def draw_cross(self, x, y, color=None):
            self.crosses.append((x, y, color))

        def draw_string(self, x, y, text, color=None):
            self.labels.append((x, y, text, color))

    uart = FakeUART()
    img = FakeImage()

    module.process_object_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    expected_target_cross = (
        IMAGE_WIDTH - 1 - int(target_x),
        IMAGE_HEIGHT - 1 - int(target_y),
    )
    assert any(entry[:2] == expected_target_cross for entry in img.crosses)
    assert any(entry[2] == "SELECT" for entry in img.labels)


def test_assistant_orbit_correction_uses_independent_velocity_params_without_event() -> None:
    """绕行修正模式使用独立速度参数且不产生可靠事件."""

    module = load_assistant()
    module.OBJECT_ORBIT_KP_X = 0.2
    module.OBJECT_ORBIT_KP_Y = -0.3
    module.OBJECT_ORBIT_MIN_SPEED = 0.0
    module.OBJECT_ORBIT_MAX_VX = 9.0
    module.OBJECT_ORBIT_MAX_VY = 9.0
    module.OBJECT_ORBIT_DEADZONE_X_PX = 3.0
    module.OBJECT_ORBIT_DEADZONE_Y_PX = 3.0
    state = module.AssistantVisionState(stable_frames=1)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.ORBIT,
            module.Target.OBJECT,
            module.Task.ORBIT,
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.ORBIT,
    )
    err_x = 10.0
    err_y = 12.0
    observation = module.build_object_observation(
        1,
        target_x + err_x,
        target_y + err_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.ORBIT,
    )

    velocity = module.build_object_orbit_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    )
    state.accept_object_observation(observation)

    expected_y = err_y * (
        float(module.OBJECT_ORBIT_MAX_VY)
        / abs(float(module.OBJECT_ORBIT_KP_Y))
        / float(IMAGE_HEIGHT)
    ) * float(module.OBJECT_ORBIT_KP_Y)
    assert velocity == pytest.approx((err_x * module.OBJECT_ORBIT_KP_X, expected_y))
    assert state.next_event_frame() is None


def test_assistant_orbit_correction_missing_target_outputs_zero_velocity() -> None:
    """绕行修正模式无目标时输出零速度."""

    module = load_assistant()
    observation = module.build_object_observation(
        0,
        0,
        0,
        0,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.ORBIT,
    )

    assert module.build_object_orbit_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    ) == (0.0, 0.0)


def test_assistant_orbit_correction_zero_kp_outputs_zero_velocity() -> None:
    """绕行修正增益为零时保持零修正."""

    module = load_assistant()
    module.OBJECT_ORBIT_KP_X = 0.0
    module.OBJECT_ORBIT_KP_Y = 0.0
    target_x, target_y = assistant_target_point(
        module,
        module.Task.ORBIT,
    )
    observation = module.build_object_observation(
        1,
        target_x + module.OBJECT_ORBIT_DEADZONE_X_PX + 10.0,
        target_y + module.OBJECT_ORBIT_DEADZONE_Y_PX + 10.0,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.ORBIT,
    )

    assert module.build_object_orbit_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    ) == (0.0, 0.0)


def test_assistant_older_sync_only_replies_ack_without_reverting_mode() -> None:
    """较早同步包只确认, 不回退当前模式."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    assert_assistant_ack(
        module,
        state.handle_control_line(assistant_sync_frame(11, 1, 0, 0)),
        11,
    )
    assert state.mode == module.RunMode.APPROACH_OBJECT


def test_assistant_repeated_sync_replies_ack_without_clearing_pending_event() -> None:
    """重复同步包只重复 ACK, 不清掉未确认事件."""

    module = load_assistant()
    now_ms = [100]
    state = module.AssistantVisionState(
        stable_frames=1,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    observation = centered_object_observation(module, 180)
    state.accept_object_observation(observation)
    first = state.next_event_frame()

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    now_ms[0] += 20

    assert_assistant_event(module, first, 12, module.Event.TARGET_FOUND, 180)
    assert state.next_event_frame() == first


def test_assistant_missing_target_outputs_zero_search_velocity() -> None:
    """无目标时找物体模式输出零搜索速度。"""

    module = load_assistant()
    observation = module.build_object_observation(0, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert module.build_object_approach_velocity_from_observation(
        observation, IMAGE_HEIGHT
    ) == (
        module.OBJECT_MISSING_SEARCH_VX,
        module.OBJECT_MISSING_SEARCH_VY,
    )


def test_assistant_object_target_point_generates_p_search_velocity() -> None:
    """有目标时找物体模式按当前目标点误差生成速度."""

    module = load_assistant()
    target_x, target_y = assistant_target_point(module)
    err_x = choose_outside_deadzone_offset(
        target_x,
        IMAGE_WIDTH,
        module.OBJECT_APPROACH_DEADZONE_X_PX,
        clearance=15.0,
    )
    err_y = choose_outside_deadzone_offset(
        target_y,
        IMAGE_HEIGHT,
        module.OBJECT_APPROACH_DEADZONE_Y_PX,
        clearance=15.0,
    )
    observation = module.build_object_observation(
        1,
        target_x + err_x,
        target_y + err_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    velocity = module.build_object_approach_velocity_from_observation(
        observation, IMAGE_HEIGHT
    )

    assert observation == pytest.approx((err_x, err_y, 300.0))
    assert velocity == pytest.approx(
        (
            expected_axis_velocity(
                err_x,
                module.OBJECT_APPROACH_KP_X,
                module.OBJECT_APPROACH_MIN_SPEED,
                module.OBJECT_APPROACH_MAX_VX,
            ),
            expected_object_y_velocity(module, err_y),
        )
    )


def test_assistant_transport_observation_uses_transport_target_point() -> None:
    """搬运入口配置的目标底边必须切到推行阶段目标点."""

    module = load_assistant()
    search_target_x, search_target_y = assistant_target_point(
        module,
        module.Task.SEARCH,
    )
    transport_target_x, transport_target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    search_observation = module.build_object_observation(
        1,
        search_target_x,
        search_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.SEARCH,
    )
    transport_observation = module.build_object_observation(
        1,
        transport_target_x,
        transport_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.TRANSPORT,
    )

    assert search_target_y == float(module.OBJECT_APPROACH_TARGET_Y_PX)
    assert transport_target_y == float(module.ASSISTANT_TRANSPORT_TARGET_Y_PX)
    assert search_observation == pytest.approx((0.0, 0.0, 300.0))
    assert transport_observation == pytest.approx((0.0, 0.0, 300.0))


def test_assistant_transport_config_treats_search_target_as_not_aligned() -> None:
    """搬运入口配置不能继续沿用寻找阶段的 210 目标点."""

    module = load_assistant()
    search_target_x, search_target_y = assistant_target_point(
        module,
        module.Task.SEARCH,
    )

    observation = module.build_object_observation(
        1,
        search_target_x,
        search_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.Task.TRANSPORT,
    )

    expected_y = float(module.OBJECT_APPROACH_TARGET_Y_PX) - float(
        module.ASSISTANT_TRANSPORT_TARGET_Y_PX
    )

    assert observation == pytest.approx((0.0, expected_y, 300.0))


def test_assistant_object_params_stay_within_qvga_bounds() -> None:
    """找物体像素参数保持在当前图像范围内."""

    module = load_assistant()

    assert 0.0 <= float(module.OBJECT_APPROACH_TARGET_X_PX) <= IMAGE_WIDTH
    assert 0.0 <= float(module.OBJECT_APPROACH_TARGET_Y_PX) <= IMAGE_HEIGHT
    assert 0.0 <= float(module.OBJECT_APPROACH_DEADZONE_X_PX) < IMAGE_WIDTH
    assert 0.0 <= float(module.OBJECT_APPROACH_DEADZONE_Y_PX) < IMAGE_HEIGHT
    assert 0.0 <= float(module.OBJECT_X_TOLERANCE_PX) <= IMAGE_WIDTH
    assert 0.0 <= float(module.OBJECT_Y_TOLERANCE_PX) <= IMAGE_HEIGHT
    assert float(module.OBJECT_MIN_AREA) >= 0.0
    assert int(module.OBJECT_STABLE_FRAMES) >= 1


def test_assistant_target_found_window_matches_approach_deadzone() -> None:
    """找物体命中窗口与停下修正的死区保持一致."""

    module = load_assistant()

    assert float(module.OBJECT_X_TOLERANCE_PX) == float(module.OBJECT_APPROACH_DEADZONE_X_PX)
    assert float(module.OBJECT_Y_TOLERANCE_PX) == float(module.OBJECT_APPROACH_DEADZONE_Y_PX)


def test_assistant_transport_candidate_window_covers_qvga_frame() -> None:
    """搬运候选筛选窗口覆盖当前 QVGA 图像范围."""

    module = load_assistant()

    assert float(module.OBJECT_TRANSPORT_WINDOW_X_PX) == float(IMAGE_WIDTH)
    assert float(module.OBJECT_TRANSPORT_WINDOW_Y_PX) == float(IMAGE_HEIGHT)


def test_assistant_transport_candidate_stays_visible_after_crossing_target_line() -> None:
    """搬运目标越过命中线后仍在配置窗口内时继续参与横向修正."""

    module = load_assistant()
    target_x, target_y = module.build_object_target_point(module.Task.TRANSPORT)
    window_y = float(module.OBJECT_TRANSPORT_WINDOW_Y_PX)
    candidate = ("red", target_x, 0.0, target_y + window_y / 2.0, 300.0, object())
    outside = ("red", target_x, 0.0, target_y + window_y + 1.0, 300.0, object())

    filtered = module.filter_candidates_in_target_window(
        (candidate, outside),
        target_x,
        target_y,
        module.OBJECT_TRANSPORT_WINDOW_X_PX,
        module.OBJECT_TRANSPORT_WINDOW_Y_PX,
    )

    assert filtered == [candidate]


def test_assistant_object_target_can_be_reconfigured(monkeypatch) -> None:
    """找物体目标点改动后, 候选选择和输出速度都要跟着变化."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART()
    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    monkeypatch.setattr(module, "OBJECT_APPROACH_TARGET_X_PX", 80.0)
    monkeypatch.setattr(module, "OBJECT_APPROACH_TARGET_Y_PX", 120.0)

    class FakeBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            return self._rect[0] + self._rect[2] / 2

        def cy(self):
            return self._rect[1] + self._rect[3] / 2

        def min_corners(self):
            left, top, width, height = self._rect
            right = left + width
            bottom = top + height
            return ((left, top), (right, top), (right, bottom), (left, bottom))

    class FakeImage:
        def __init__(self, blobs):
            self._blobs = blobs
            self.crosses = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return list(self._blobs)

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    first_blob = FakeBlob(70, 120, 20, 20)
    second_blob = FakeBlob(150, 40, 20, 20)
    img = FakeImage([second_blob, first_blob])

    module.process_object_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert len(uart.writes) == 1
    assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)


def test_assistant_hook_waits_for_stable_target_before_event() -> None:
    """目标稳定满足条件后才创建 TARGET_FOUND 事件."""

    module = load_assistant()
    state = module.AssistantVisionState(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=2,
    )

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    observation = centered_object_observation(module, 150)

    state.accept_object_observation(observation)
    assert state.next_event_frame() is None
    state.accept_object_observation(observation)

    assert_assistant_event(module, state.next_event_frame(), 12, module.Event.TARGET_FOUND, 150)


def test_assistant_target_found_sends_stable_zero_before_event() -> None:
    """稳定命中前先输出零速度, 下一拍再发可靠事件."""

    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=2)
    uart = FakeUART()
    img = aligned_object_image(module)
    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )

    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert len(uart.writes) == 3
    assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)
    assert_velocity_frame(module, uart.writes[1], 0.0, 0.0)
    assert_assistant_event(module, uart.writes[2], 12, module.Event.TARGET_FOUND, 300)


def test_assistant_pending_event_suppresses_velocity_between_retries() -> None:
    """可靠事件等待确认期间不继续输出速度流."""

    module = load_assistant()
    now_ms = [100]
    state = module.AssistantVisionState(
        stable_frames=1,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )
    uart = FakeUART()
    img = aligned_object_image(module)
    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )

    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)
    now_ms[0] += 20
    module.process_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert len(uart.writes) == 3
    assert_velocity_frame(module, uart.writes[0], 0.0, 0.0)
    assert_assistant_event(module, uart.writes[1], 12, module.Event.TARGET_FOUND, 300)
    assert_assistant_event(module, uart.writes[2], 12, module.Event.TARGET_FOUND, 300)


def test_assistant_hook_repeats_event_until_matching_ack() -> None:
    """事件确认前重复发送, 收到匹配 ACK 后停止发送."""

    module = load_assistant()
    now_ms = [100]
    state = module.AssistantVisionState(
        stable_frames=1,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
        ),
        12,
    )
    observation = centered_object_observation(module, 180)
    state.accept_object_observation(observation)
    first = state.next_event_frame()
    state.handle_control_line(assistant_event_ack_frame(11))
    now_ms[0] += 20
    second = state.next_event_frame()
    state.handle_control_line(assistant_event_ack_frame(12))

    assert_assistant_event(module, first, 12, module.Event.TARGET_FOUND, 180)
    assert second == first
    assert state.next_event_frame() is None


def test_assistant_transport_mode_emits_aligned_for_transport_config() -> None:
    """搬运入口配置稳定满足条件后回报 ALIGNED."""

    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=1)

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(
                12,
                int(module.State.APPROACH_OBJECT),
                1,
                int(module.Task.TRANSPORT),
            )
        ),
        12,
    )
    observation = centered_transport_observation(module, 180)
    state.accept_object_observation(observation)

    assert_assistant_event(module, state.next_event_frame(), 12, module.Event.ALIGNED, 180)


def test_assistant_orbit_state_transport_config_aligns_to_transport_target() -> None:
    """主车绕行后辅车跳过绕行时直接按搬运目标点对正."""

    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=1)

    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(
                12,
                int(module.State.ORBIT),
                int(module.Target.OBJECT),
                pack_task_arg(module.Task.TRANSPORT, 2),
            )
        ),
        12,
    )
    observation = centered_transport_observation(module, 180)
    state.accept_object_observation(observation)

    assert state.mode == module.RunMode.APPROACH_OBJECT
    assert state.current_object_config_id() == module.Task.TRANSPORT
    assert_assistant_event(module, state.next_event_frame(), 12, module.Event.ALIGNED, 180)


def test_assistant_transport_mode_keeps_object_velocity_output() -> None:
    """搬运入口配置继续输出物体视觉速度."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART()
    assert_assistant_ack(
        module,
        state.handle_control_line(
            assistant_sync_frame(
                12,
                int(module.State.APPROACH_OBJECT),
                1,
                int(module.Task.TRANSPORT),
            )
        ),
        12,
    )

    class FakeBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            return self._rect[0] + self._rect[2] / 2

        def cy(self):
            return self._rect[1] + self._rect[3] / 2

        def min_corners(self):
            left, top, width, height = self._rect
            right = left + width
            bottom = top + height
            return ((left, top), (right, top), (right, bottom), (left, bottom))

    class FakeImage:
        def __init__(self, blobs):
            self._blobs = blobs
            self.crosses = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return list(self._blobs)

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    img = FakeImage([FakeBlob(150, 150, 20, 20)])
    module.process_object_frame(uart, state, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    frame = module.decode_frame(uart.writes[0])
    assert frame is not None
    assert frame["mode"] == module.Mode.UDP
    assert frame["topic"] == module.Topic.LOCAL_VISION_VELOCITY


def test_process_object_frame_accepts_x_outside_when_bottom_hits_target_window_in_transport() -> None:
    """辅车推行阶段只要求候选框底边命中目标窗口."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    class FakeBlob:
        def rect(self):
            width = 20
            height = 20
            return (
                target_x + float(module.OBJECT_X_TOLERANCE_PX) + 10.0,
                IMAGE_HEIGHT - target_y,
                width,
                height,
            )

        def cx(self):
            return target_x + float(module.OBJECT_X_TOLERANCE_PX) + 20.0

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10.0

        def area(self):
            return 300

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return [FakeBlob()]

        def draw_cross(self, x, y, color=None):
            _ = (x, y, color)

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

        def draw_string(self, x, y, text, color=None):
            _ = (x, y, text, color)

    uart = FakeUART()

    module.process_object_frame(uart, state, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    assert_velocity_frame(
        module,
        uart.writes[0],
        expected_axis_velocity(
            float(module.OBJECT_X_TOLERANCE_PX) + 20.0,
            module.OBJECT_APPROACH_KP_X,
            module.OBJECT_APPROACH_MIN_SPEED,
            module.OBJECT_APPROACH_MAX_VX,
        ),
        0.0,
    )


def test_process_object_frame_prefers_largest_area_after_bottom_filter_in_transport() -> None:
    """辅车推行阶段在底边命中的候选中选择面积最大者."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    class FakeBlob:
        def __init__(self, center_x, area):
            self._center_x = center_x
            self._area = area

        def rect(self):
            width = 20
            height = 20
            return (self._center_x - 10.0, IMAGE_HEIGHT - target_y, width, height)

        def cx(self):
            return self._center_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10.0

        def area(self):
            return self._area

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return [
                FakeBlob(target_x, 300),
                FakeBlob(target_x + float(module.OBJECT_X_TOLERANCE_PX) + 20.0, 800),
            ]

        def draw_cross(self, x, y, color=None):
            _ = (x, y, color)

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

        def draw_string(self, x, y, text, color=None):
            _ = (x, y, text, color)

    uart = FakeUART()

    module.process_object_frame(uart, state, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    assert_velocity_frame(
        module,
        uart.writes[0],
        expected_axis_velocity(
            float(module.OBJECT_X_TOLERANCE_PX) + 20.0,
            module.OBJECT_APPROACH_KP_X,
            module.OBJECT_APPROACH_MIN_SPEED,
            module.OBJECT_APPROACH_MAX_VX,
        ),
        0.0,
    )


def test_assistant_object_candidates_use_yolo_when_flag_enabled() -> None:
    """打开开关后辅车找物体候选切换到 YOLO."""

    module = load_assistant()
    module.OBJECT_DETECTION_USE_YOLO = True
    module.tf = FakeYoloTf()

    class FakeImage:
        def __init__(self):
            self.copy_calls = []

        def width(self):
            return IMAGE_WIDTH

        def height(self):
            return IMAGE_HEIGHT

        def copy(self, scale, copy_to_fb):
            self.copy_calls.append((scale, copy_to_fb))
            return "detect-image"

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            raise AssertionError("打开 YOLO 后不应继续调用色块识别")

    img = FakeImage()

    candidates = module.build_object_blob_candidates(img)

    assert module.tf.load_calls == [(module.YOLO_MODEL_PATH, True)]
    assert module.tf.detect_calls == [("fake-yolo-net", "detect-image")]
    assert img.copy_calls == [(module.YOLO_IMAGE_COPY_SCALE, 1)]
    assert candidates[0][0] == "red"
    assert candidates[0][1] == pytest.approx(160.0)
    assert candidates[0][3] == pytest.approx(210.0)
    assert candidates[0][4] == pytest.approx(3200.0)


def test_process_object_frame_ignores_candidates_when_bottom_outside_target_window_in_transport() -> None:
    """辅车推行阶段忽略底边未命中目标窗口的候选框."""

    module = load_assistant()
    module.OBJECT_TASKS = (("red", ((1, 2, 3, 4, 5, 6),), 0, 1, 1, True),)
    state = module.AssistantVisionState(stable_frames=99)
    state.handle_control_line(
        assistant_sync_frame(
            12,
            module.State.TRANSPORT_OBJECT,
            module.Target.OBJECT,
            pack_task_arg(module.Task.TRANSPORT, 1),
        )
    )
    target_x, target_y = assistant_target_point(
        module,
        module.Task.TRANSPORT,
    )

    class FakeBlob:
        def rect(self):
            width = 20
            height = 20
            bottom_y = target_y + float(module.OBJECT_Y_TOLERANCE_PX) + 10.0
            return (
                target_x,
                IMAGE_HEIGHT - bottom_y,
                width,
                height,
            )

        def cx(self):
            return target_x

        def cy(self):
            bottom_y = target_y + float(module.OBJECT_Y_TOLERANCE_PX) + 10.0
            return IMAGE_HEIGHT - bottom_y + 10.0

        def area(self):
            return 300

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = thresholds
            _ = pixels_threshold
            _ = area_threshold
            _ = merge
            _ = margin
            return [FakeBlob()]

        def draw_cross(self, x, y, color=None):
            _ = (x, y, color)

        def draw_rectangle(self, x, y, w, h):
            _ = (x, y, w, h)

        def draw_string(self, x, y, text, color=None):
            _ = (x, y, text, color)

    uart = FakeUART()

    module.process_object_frame(uart, state, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    assert_velocity_frame(
        module,
        uart.writes[0],
        module.OBJECT_MISSING_SEARCH_VX,
        module.OBJECT_MISSING_SEARCH_VY,
    )


def test_assistant_process_uart_input_writes_local_ack() -> None:
    """串口输入路径收到同步包后必须回写本地 ACK."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART(
        assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
    )

    assert module.process_uart_input(uart, b"", state) == b""
    assert uart.writes == [module.format_ack_frame(12)]
def test_assistant_run_applies_lens_correction_before_processing() -> None:
    """辅车入口先校准翻转后图像再进入业务处理."""

    module = load_assistant()

    class StopLoop(Exception):
        pass

    class SnapshotImage:
        def __init__(self):
            self.lens_corr_called = False

        def replace(self, **_kwargs):
            return self

        def lens_corr(self, strength, zoom):
            _ = strength
            _ = zoom
            self.lens_corr_called = True

    image = SnapshotImage()

    class Sensor:
        def snapshot(self):
            return image

    module.sensor = Sensor()
    module.init_uart = lambda: FakeUART()
    module.init_sensor = lambda: (IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_uart_input = lambda uart, rx_buffer, state: rx_buffer

    def stop_after_frame(uart, state, img, image_width, image_height):
        _ = uart
        _ = state
        _ = image_width
        _ = image_height
        assert img is image
        assert image.lens_corr_called is True
        raise StopLoop()

    module.process_frame = stop_after_frame

    with pytest.raises(StopLoop):
        module.run()


def test_assistant_process_uart_input_skips_crc_invalid_false_sync_frame() -> None:
    """串口输入错位假同步包不能吞掉后续真实同步包."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART(
        b"\x02"
        + assistant_sync_frame(12, module.State.APPROACH_OBJECT, module.Target.OBJECT, 1)
    )

    assert module.process_uart_input(uart, b"", state) == b""
    assert uart.writes == [module.format_ack_frame(12)]
    assert state.mode == module.RunMode.APPROACH_OBJECT


def test_assistant_process_uart_input_ignores_bad_decode() -> None:
    """串口输入解码失败不能中断主循环."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert module.process_uart_input(BadReadUART(), b"partial", state) == b"partial\xff"
    assert state.mode == module.RunMode.FOLLOW
