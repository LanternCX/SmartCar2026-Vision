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
    observation = module.build_object_observation(1, 160, 240, 180, 320, 240)
    state.accept_object_observation(observation)
    first = state.next_event_frame()

    assert state.handle_control_line("s,12,2,1,1") == "a,12"
    now_ms[0] += 20

    assert first == "r,12,6,180"
    assert state.next_event_frame() == first


def test_assistant_missing_target_outputs_zero_search_velocity() -> None:
    """无目标时找物体模式输出零搜索速度。"""

    module = load_assistant()
    observation = module.build_object_observation(0, 0, 0, 0, 320, 240)

    assert module.build_object_approach_velocity_from_observation(observation, 240) == (
        module.OBJECT_MISSING_SEARCH_VX,
        module.OBJECT_MISSING_SEARCH_VY,
    )
    assert module.OBJECT_MISSING_SEARCH_VY == 0.0


def test_assistant_target_bottom_generates_p_search_velocity() -> None:
    """有目标时找物体模式使用中心和底边误差生成速度."""

    module = load_assistant()
    observation = module.build_object_observation(1, 190, 172, 300, 320, 240)
    velocity = module.build_object_approach_velocity_from_observation(observation, 240)

    assert observation == (30.0, -68.0, 300.0)
    expected_vx = max(
        30.0 * module.OBJECT_APPROACH_KP_X,
        float(module.OBJECT_APPROACH_MIN_SPEED),
    )
    scaled_y_error = observation[1] * (
        float(module.OBJECT_APPROACH_MAX_VY)
        / abs(float(module.OBJECT_APPROACH_KP_Y))
        / 240.0
    )
    expected_vy = scaled_y_error * float(module.OBJECT_APPROACH_KP_Y)
    expected_vy = max(
        -float(module.OBJECT_APPROACH_MAX_VY),
        min(float(module.OBJECT_APPROACH_MAX_VY), expected_vy),
    )
    if 0.0 < expected_vy < float(module.OBJECT_APPROACH_MIN_SPEED):
        expected_vy = float(module.OBJECT_APPROACH_MIN_SPEED)
    elif -float(module.OBJECT_APPROACH_MIN_SPEED) < expected_vy < 0.0:
        expected_vy = -float(module.OBJECT_APPROACH_MIN_SPEED)

    assert velocity[0] == pytest.approx(expected_vx)
    assert velocity[1] == pytest.approx(expected_vy)


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
    observation = module.build_object_observation(1, 160, 240, 150, 320, 240)

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
    observation = module.build_object_observation(1, 160, 240, 180, 320, 240)
    state.accept_object_observation(observation)
    first = state.next_event_frame()
    state.handle_control_line("a,11")
    now_ms[0] += 20
    second = state.next_event_frame()
    state.handle_control_line("a,12")

    assert first == "r,12,6,180"
    assert second == first
    assert state.next_event_frame() is None


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
