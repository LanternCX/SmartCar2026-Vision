"""辅车视觉找物体模式单元测试."""

import pytest

from tests.test_support import load_main_module


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


def load_assistant():
    """加载辅车视觉入口模块."""

    return load_main_module("assistant_object_approach_test_module")


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def assistant_target_point(module, config_id=None):
    """返回当前指定配置的找物体目标点."""

    if config_id is None:
        config_id = module.OBJECT_APPROACH_CONFIG_ID
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
        module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID,
    )
    return module.build_object_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID,
    )


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
    scaled_error = err_y * (
        float(module.OBJECT_APPROACH_MAX_VY)
        / abs(float(module.OBJECT_APPROACH_KP_Y))
        / float(IMAGE_HEIGHT)
    )
    return expected_axis_velocity(
        scaled_error,
        module.OBJECT_APPROACH_KP_Y,
        module.OBJECT_APPROACH_MIN_SPEED,
        module.OBJECT_APPROACH_MAX_VY,
    )


def test_assistant_defaults_to_follow_mode() -> None:
    """默认启动模式仍然是 follow."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert state.mode == module.MODE_FOLLOW


def test_assistant_sync_packet_switches_to_object_mode_and_replies_ack() -> None:
    """本地同步包切换到找物体模式时必须回复 ACK."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    assert state.mode == module.MODE_APPROACH_OBJECT


def test_assistant_older_sync_only_replies_ack_without_reverting_mode() -> None:
    """较早同步包只确认, 不回退当前模式."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    assert state.handle_control_line("s,11,1,0,0") == "a,11"
    assert state.mode == module.MODE_APPROACH_OBJECT


def test_assistant_repeated_sync_replies_ack_without_clearing_pending_event() -> None:
    """重复同步包只重复 ACK, 不清掉未确认事件."""

    module = load_assistant()
    now_ms = [100]
    state = module.AssistantVisionState(
        stable_frames=1,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    observation = centered_object_observation(module, 180)
    state.accept_object_observation(observation)
    first = state.next_event_frame()

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    now_ms[0] += 20

    assert first == "r,12,6,180"
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
    assert module.OBJECT_MISSING_SEARCH_VY == 0.0


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
        module.OBJECT_APPROACH_CONFIG_ID,
    )
    transport_target_x, transport_target_y = assistant_target_point(
        module,
        module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID,
    )

    search_observation = module.build_object_observation(
        1,
        search_target_x,
        search_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.OBJECT_APPROACH_CONFIG_ID,
    )
    transport_observation = module.build_object_observation(
        1,
        transport_target_x,
        transport_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID,
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
        module.OBJECT_APPROACH_CONFIG_ID,
    )

    observation = module.build_object_observation(
        1,
        search_target_x,
        search_target_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID,
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


def test_assistant_object_target_can_be_reconfigured(monkeypatch) -> None:
    """找物体目标点改动后, 候选选择和输出速度都要跟着变化."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART()
    assert state.handle_control_line("s,12,2,1,1") == "a,12"
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

    assert uart.writes == ["v,0,0\r\n"]


def test_assistant_hook_waits_for_stable_target_before_event() -> None:
    """目标稳定满足条件后才创建 TARGET_FOUND 事件."""

    module = load_assistant()
    state = module.AssistantVisionState(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=2,
    )

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    observation = centered_object_observation(module, 150)

    state.accept_object_observation(observation)
    assert state.next_event_frame() is None
    state.accept_object_observation(observation)

    assert state.next_event_frame() == "r,12,6,150"


def test_assistant_hook_repeats_event_until_matching_ack() -> None:
    """事件确认前重复发送, 收到匹配 ACK 后停止发送."""

    module = load_assistant()
    now_ms = [100]
    state = module.AssistantVisionState(
        stable_frames=1,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    observation = centered_object_observation(module, 180)
    state.accept_object_observation(observation)
    first = state.next_event_frame()
    state.handle_control_line("a,11")
    now_ms[0] += 20
    second = state.next_event_frame()
    state.handle_control_line("a,12")

    assert first == "r,12,6,180"
    assert second == first
    assert state.next_event_frame() is None


def test_assistant_transport_mode_emits_aligned_for_transport_config() -> None:
    """搬运入口配置稳定满足条件后回报 ALIGNED."""

    module = load_assistant()
    state = module.AssistantVisionState(stable_frames=1)

    assert (
        state.handle_control_line(
            "s,12,%d,1,%d"
            % (
                int(module.STATE_APPROACH_OBJECT),
                int(module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID),
            )
        )
        == "a,12"
    )
    observation = centered_transport_observation(module, 180)
    state.accept_object_observation(observation)

    assert state.next_event_frame() == "r,12,7,180"


def test_assistant_transport_mode_keeps_object_velocity_output() -> None:
    """搬运入口配置继续输出物体视觉速度."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART()
    assert (
        state.handle_control_line(
            "s,12,%d,1,%d"
            % (
                int(module.STATE_APPROACH_OBJECT),
                int(module.ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID),
            )
        )
        == "a,12"
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

    assert uart.writes[0].startswith("v,")


def test_assistant_process_uart_input_writes_local_ack() -> None:
    """串口输入路径收到同步包后必须回写本地 ACK."""

    module = load_assistant()
    state = module.AssistantVisionState()
    uart = FakeUART(b"s,12,2,1,1\r\n")

    assert module.process_uart_input(uart, "", state) == ""
    assert uart.writes == ["a,12\r\n"]


def test_assistant_process_uart_input_ignores_bad_decode() -> None:
    """串口输入解码失败不能中断主循环."""

    module = load_assistant()
    state = module.AssistantVisionState()

    assert module.process_uart_input(BadReadUART(), "partial", state) == "partial"
    assert state.mode == module.MODE_FOLLOW
