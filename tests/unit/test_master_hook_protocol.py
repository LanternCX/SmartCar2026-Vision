"""! @brief OpenART Vision master hook 协议测试"""

import pytest

from tests.test_support import load_role_main_module


class FakeUART:
    """! @brief 记录串口写入内容的测试桩"""

    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)


class BlockedUART:
    """! @brief 模拟串口无法继续写出的测试桩"""

    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return 0


class BadReadUART:
    """! @brief 模拟输入字节无法解码的测试桩"""

    def any(self):
        return 1

    def read(self, size):
        return b"\xff"


def load_master():
    """! @brief 加载主车视觉入口模块"""

    return load_role_main_module("master", "master_hook_protocol_test_module")


def test_master_sync_packet_records_context_and_replies_ack() -> None:
    """! @brief 主车视觉同步包使用可靠序号确认, 使用上下文编号建业务上下文"""

    module = load_master()
    hook = module.MasterVisionHook()

    reply = hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    assert reply == "a,12"
    assert observation == (7, 0.0, 0.0, 180.0)


def test_master_repeated_sync_replies_ack_without_reapplying() -> None:
    """! @brief 重复同步包只重复确认, 不重复建立上下文"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=2, next_reliable_seq=30)

    assert hook.handle_control_line("s,12,7,1,1,1") == "a,12"
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)
    hook.accept_observation(observation)
    assert hook.handle_control_line("s,12,7,1,1,1") == "a,12"
    hook.accept_observation(observation)

    assert hook.next_event_frame() == "r,30,7,6,180"


def test_master_non_new_context_does_not_override_active_context() -> None:
    """! @brief 非新上下文同步包必须确认, 但不能覆盖已建立上下文"""

    module = load_master()
    hook = module.MasterVisionHook()

    assert hook.handle_control_line("s,12,7,1,1,1") == "a,12"
    assert hook.handle_control_line("s,13,6,2,1,9") == "a,13"
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    assert observation == (7, 0.0, 0.0, 180.0)


def test_master_reliable_seq_is_separate_from_context_id() -> None:
    """! @brief 同步确认和事件确认只匹配可靠序号, 不使用上下文编号代替"""

    module = load_master()
    now_ms = [100]
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    event_frame = hook.next_event_frame()
    hook.handle_control_line("a,12")
    now_ms[0] += 20

    assert event_frame == "r,30,7,6,180"
    assert hook.next_event_frame() == event_frame


def test_master_object_observation_uses_middle_and_image_bottom_target() -> None:
    """! @brief 物体观测误差使用画面中线和图像底边"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(1, 160, 240, 250, 320, 240)

    assert observation == (7, 0.0, 0.0, 250.0)


def test_master_missing_target_outputs_zero_observation() -> None:
    """! @brief 无目标时主车视觉输出同一上下文下的零观测"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(0, 0, 0, 0, 320, 240)

    assert observation == (7, 0.0, 0.0, 0.0)


def test_master_blob_candidates_report_area_as_value() -> None:
    """! @brief 候选物体强度使用 blob 面积"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            return (10, 20, 30, 40)

        def cx(self):
            return 25

        def cy(self):
            return 40

        def area(self):
            return 1234

    class FakeImage:
        def height(self):
            return 100

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    candidates = module.build_blob_candidates(FakeImage())

    assert candidates[0][3] == 1234


def test_master_hook_waits_for_stable_target_before_event() -> None:
    """! @brief hook 条件连续满足后才创建 TARGET_FOUND 事件"""

    module = load_master()
    hook = module.MasterVisionHook(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=2,
        next_reliable_seq=30,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 150, 320, 240)

    hook.accept_observation(observation)
    first_frame = hook.next_event_frame()
    hook.accept_observation(observation)
    second_frame = hook.next_event_frame()

    assert first_frame is None
    assert second_frame == "r,30,7,6,150"


def test_master_hook_does_not_event_when_condition_is_not_met() -> None:
    """! @brief 目标强度或误差不满足 hook 条件时不发送 TARGET_FOUND"""

    module = load_master()
    hook = module.MasterVisionHook(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=1,
        next_reliable_seq=30,
    )
    hook.handle_control_line("s,12,7,1,1,1")

    weak = hook.build_observation(1, 160, 240, 99, 320, 240)
    offset = hook.build_observation(1, 180, 240, 150, 320, 240)
    hook.accept_observation(weak)
    hook.accept_observation(offset)

    assert hook.next_event_frame() is None


def test_master_hook_does_not_event_for_unsupported_hook_config() -> None:
    """! @brief 未支持的 hook 配置不创建 TARGET_FOUND 事件"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=1, next_reliable_seq=30)
    hook.handle_control_line("s,12,7,1,1,99")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)

    assert hook.next_event_frame() is None


def test_master_hook_throttles_pending_event_retries() -> None:
    """! @brief 可靠事件按低频节奏重复发送, 不随每帧重复"""

    module = load_master()
    now_ms = [100]
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    first = hook.next_event_frame()
    second = hook.next_event_frame()
    now_ms[0] += 19
    third = hook.next_event_frame()
    now_ms[0] += 1
    fourth = hook.next_event_frame()

    assert first == "r,30,7,6,180"
    assert second is None
    assert third is None
    assert fourth == first


def test_master_hook_default_event_retry_interval_is_low_frequency() -> None:
    """! @brief 默认可靠事件重发间隔高于常见视觉单帧间隔"""

    module = load_master()
    now_ms = [100]
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
        now_ms=lambda: now_ms[0],
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    first = hook.next_event_frame()
    now_ms[0] += 20
    second = hook.next_event_frame()

    assert first == "r,30,7,6,180"
    assert second is None


def test_master_hook_repeats_event_until_matching_ack() -> None:
    """! @brief 事件确认前重复发送同一个可靠事件, 匹配确认后停止发送"""

    module = load_master()
    now_ms = [100]
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    first = hook.next_event_frame()
    hook.handle_control_line("a,29")
    now_ms[0] += 20
    second = hook.next_event_frame()
    now_ms[0] += 20
    third = hook.next_event_frame()
    hook.handle_control_line("a,30")

    assert first == "r,30,7,6,180"
    assert second == first
    assert third == first
    assert hook.next_event_frame() is None


def test_master_hook_keeps_unacked_event_after_new_context_sync() -> None:
    """! @brief 新上下文同步不能清除尚未确认的可靠事件"""

    module = load_master()
    now_ms = [100]
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
        now_ms=lambda: now_ms[0],
        event_resend_interval_ms=20,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    first = hook.next_event_frame()
    hook.handle_control_line("s,13,8,1,1,1")
    now_ms[0] += 20
    second = hook.next_event_frame()
    hook.handle_control_line("a,30")

    assert first == "r,30,7,6,180"
    assert second == first
    assert hook.next_event_frame() is None


def test_master_hook_creates_target_found_once_per_context() -> None:
    """! @brief 同一上下文只创建一次 TARGET_FOUND 事件"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=1, next_reliable_seq=30)
    hook.handle_control_line("s,12,7,1,1,1")
    observation = hook.build_observation(1, 160, 240, 180, 320, 240)

    hook.accept_observation(observation)
    assert hook.next_event_frame() == "r,30,7,6,180"
    hook.handle_control_line("a,30")
    hook.accept_observation(observation)

    assert hook.next_event_frame() is None


def test_process_uart_input_ignores_bad_decode() -> None:
    """! @brief 串口输入解码失败不能中断主循环"""

    module = load_master()
    hook = module.MasterVisionHook()

    rx_buffer = module.process_uart_input(BadReadUART(), "partial", hook)

    assert rx_buffer == "partial"
    assert not hook.has_context()


def test_data_stream_write_does_not_sleep(monkeypatch) -> None:
    """! @brief 数据流包发送不执行串口保护延时"""

    module = load_master()
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", lambda delay: sleeps.append(delay))
    uart = FakeUART()

    module.write_data_line(uart, "v,0,0")

    assert uart.writes == ["v,0,0\r\n"]
    assert sleeps == []


def test_reliable_write_sleeps_one_ms_before_and_after(monkeypatch) -> None:
    """! @brief 可靠包发送前后各执行一次 1 ms 延时"""

    module = load_master()
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", lambda delay: sleeps.append(delay))
    uart = FakeUART()

    assert module.write_reliable_line(uart, "a,12") is True

    assert uart.writes == ["a,12\r\n"]
    assert sleeps == [0.001, 0.001]


def test_reliable_write_reports_blocked_uart(monkeypatch) -> None:
    """! @brief 可靠包写出受阻时向调用方返回失败结果"""

    module = load_master()
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", lambda delay: sleeps.append(delay))
    uart = BlockedUART()

    assert module.write_reliable_line(uart, "a,12") is False

    assert uart.writes == ["a,12\r\n"]
    assert sleeps == [0.001, 0.001]


def test_master_formats_search_velocity_frame() -> None:
    """! @brief 主车搜索速度流使用 v 短包格式"""

    module = load_master()

    assert module.format_search_velocity_frame(-1.2, 0.0) == "v,-1.2,0"
    assert module.format_search_velocity_frame(0, 0) == "v,0,0"


def test_master_search_velocity_uses_observation_entry_only() -> None:
    """! @brief 主车搜索速度只保留运行路径使用的观测入口"""

    module = load_master()

    assert not hasattr(module, "build_search_velocity_command")


def test_master_missing_target_outputs_configured_search_velocity() -> None:
    """! @brief 无目标时主车搜索通过观测路径输出配置搜索速度"""

    module = load_master()

    class EmptyImage:
        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return []

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, EmptyImage(), 320, 240
    )
    velocity = module.build_search_velocity_from_observation(observation, 240)

    assert best_blob is None
    assert observation == (7, 0.0, 0.0, 0.0)
    assert velocity == (
        module.MASTER_MISSING_SEARCH_VX,
        module.MASTER_MISSING_SEARCH_VY,
    )


def test_master_target_bottom_generates_p_search_velocity() -> None:
    """! @brief 有目标时主车搜索通过物体底边到图像底边的误差生成 P 控制量"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            return (170, 68, 40, 62)

        def cx(self):
            return 190

        def cy(self):
            return 145

        def area(self):
            return 300

    class FakeImage:
        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), 320, 240
    )
    velocity = module.build_search_velocity_from_observation(observation, 240)

    assert best_blob is not None
    assert observation == (7, 30.0, -68.0, 300.0)
    expected_vx = max(
        30.0 * module.MASTER_SEARCH_KP_X,
        float(module.MASTER_SEARCH_MIN_SPEED),
    )
    scaled_y_error = observation[2] * (
        float(module.MASTER_SEARCH_MAX_VY)
        / abs(float(module.MASTER_SEARCH_KP_Y))
        / 240.0
    )
    expected_vy = scaled_y_error * float(module.MASTER_SEARCH_KP_Y)
    expected_vy = max(
        -float(module.MASTER_SEARCH_MAX_VY),
        min(float(module.MASTER_SEARCH_MAX_VY), expected_vy),
    )
    if 0.0 < expected_vy < float(module.MASTER_SEARCH_MIN_SPEED):
        expected_vy = float(module.MASTER_SEARCH_MIN_SPEED)
    elif -float(module.MASTER_SEARCH_MIN_SPEED) < expected_vy < 0.0:
        expected_vy = -float(module.MASTER_SEARCH_MIN_SPEED)

    assert velocity[0] == pytest.approx(expected_vx)
    assert velocity[1] == pytest.approx(expected_vy)


def test_master_search_y_velocity_decreases_when_target_gets_closer() -> None:
    """! @brief 主车目标接近时纵向搜索速度应变小"""

    module = load_master()

    class FarBlob:
        def rect(self):
            return (120, 100, 80, 40)

        def cx(self):
            return 160

        def cy(self):
            return 120

        def area(self):
            return 1000

    class CloseBlob:
        def rect(self):
            return (120, 20, 80, 80)

        def cx(self):
            return 160

        def cy(self):
            return 120

        def area(self):
            return 1000

    class FakeImage:
        def __init__(self, blob):
            self._blob = blob

        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [self._blob]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    far_observation, _ = module.build_observation_from_image(
        hook, FakeImage(FarBlob()), 320, 240
    )
    close_observation, _ = module.build_observation_from_image(
        hook, FakeImage(CloseBlob()), 320, 240
    )
    far_velocity = module.build_search_velocity_from_observation(far_observation, 240)
    close_velocity = module.build_search_velocity_from_observation(
        close_observation, 240
    )

    assert close_velocity[1] < far_velocity[1]


def test_master_search_velocity_deadzone_zeroes_each_axis() -> None:
    """! @brief 主车搜索通过图像观测路径对死区内轴输出零量"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            height = int(module.MASTER_SEARCH_DEADZONE_Y_PX)
            return (120, height, 80, height)

        def cx(self):
            return 160 + module.MASTER_SEARCH_DEADZONE_X_PX

        def cy(self):
            return 160

        def area(self):
            return 150

    class FakeImage:
        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), 320, 240
    )
    velocity = module.build_search_velocity_from_observation(observation, 240)

    assert best_blob is not None
    assert velocity == (0.0, 0.0)


def test_master_search_velocity_applies_min_speed_outside_deadzone() -> None:
    """! @brief 主车搜索误差超出死区时速度幅值不能低于最小速度"""

    module = load_master()

    positive = module.build_search_velocity_from_error(
        float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1,
        float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1,
        240,
    )
    negative = module.build_search_velocity_from_error(
        -(float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1),
        -(float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1),
        240,
    )

    assert positive == (
        pytest.approx(module.MASTER_SEARCH_MIN_SPEED),
        pytest.approx(-module.MASTER_SEARCH_MIN_SPEED),
    )
    assert negative == (
        pytest.approx(-module.MASTER_SEARCH_MIN_SPEED),
        pytest.approx(module.MASTER_SEARCH_MIN_SPEED),
    )


def test_master_search_velocity_clamps_vx_and_vy() -> None:
    """! @brief 主车搜索通过观测路径按轴限幅"""

    module = load_master()

    positive = module.build_search_velocity_from_observation((7, 999.0, 999.0, 300.0), 240)
    negative = module.build_search_velocity_from_observation((7, -999.0, -999.0, 300.0), 240)

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


def test_master_target_found_uses_bbox_center_and_bottom_error() -> None:
    """! @brief TARGET_FOUND 稳定判断使用色块中心 x 和底边 y 误差"""

    module = load_master()
    hook = module.MasterVisionHook(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=1,
        next_reliable_seq=30,
    )
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(1, 160, 240, 150, 320, 240)
    hook.accept_observation(observation)

    assert hook.next_event_frame() == "r,30,7,6,150"


def test_master_search_control_does_not_require_marker_span_or_min_corners() -> None:
    """! @brief 主车搜索控制不依赖最小外接旋转矩形或 marker_span"""

    module = load_master()

    class CenterOnlyBlob:
        def rect(self):
            return (120, 1, 80, 1)

        def cx(self):
            return 160

        def cy(self):
            return 160

        def area(self):
            return 4800

    class CenterOnlyImage:
        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [CenterOnlyBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, CenterOnlyImage(), 320, 240
    )
    velocity = module.build_search_velocity_from_observation(observation, 240)

    assert best_blob is not None
    assert velocity == (0.0, 0.0)


def test_master_search_frame_outputs_velocity_without_hook_context() -> None:
    """! @brief 主车速度流不依赖 hook 上下文, 直接按色块中心 x 和底边输出"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            return (190, 1, 20, 1)

        def cx(self):
            return 200

        def cy(self):
            return 160

        def area(self):
            return 500

        def min_corners(self):
            raise AssertionError("min_corners must not be used")

    class FakeImage:
        def __init__(self):
            self.crosses = []

        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    uart = FakeUART()
    hook = module.MasterVisionHook()
    img = FakeImage()

    module.process_search_frame(uart, hook, img, 320, 240)

    assert uart.writes == ["v,2,0\r\n"]
    assert img.crosses[-1] == (200, 160)
    assert hook.next_event_frame() is None


def test_master_blob_candidates_use_normalized_bottom() -> None:
    """! @brief 主车候选目标使用归一化后的色块底边 y"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            return (10, 20, 20, 30)

        def cx(self):
            return 20

        def cy(self):
            return 35

        def area(self):
            return 600

    class FakeImage:
        def height(self):
            return 100

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    candidates = module.build_blob_candidates(FakeImage())

    assert candidates[0][2] == 80


def test_master_search_frame_draws_blob_center_not_bottom() -> None:
    """! @brief 主车调试标记绘制目标中心, 不把底边当中心点"""

    module = load_master()

    class FakeBlob:
        def rect(self):
            return (190, 110, 20, 70)

        def cx(self):
            return 200

        def cy(self):
            return 145

        def area(self):
            return 500

    class FakeImage:
        def __init__(self):
            self.crosses = []

        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    uart = FakeUART()
    hook = module.MasterVisionHook()
    img = FakeImage()

    module.process_search_frame(uart, hook, img, 320, 240)

    assert img.crosses[-1] == (200, 145)


def test_master_hook_event_uses_image_bottom_not_blob_center_y() -> None:
    """! @brief TARGET_FOUND 经图像路径使用色块底边 y, 不使用中心 y"""

    module = load_master()
    hook = module.MasterVisionHook(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=1,
        next_reliable_seq=30,
    )
    hook.handle_control_line("s,12,7,1,1,1")

    class FakeBlob:
        def rect(self):
            return (120, 0, 80, 2)

        def cx(self):
            return 160

        def cy(self):
            return 210

        def area(self):
            return 4800

    class FakeImage:
        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    observation, _ = module.build_observation_from_image(hook, FakeImage(), 320, 240)
    hook.accept_observation(observation)

    assert observation == (7, 0.0, -0.0, 4800.0)
    assert hook.next_event_frame() == "r,30,7,6,4800"


def test_master_candidate_selection_uses_image_bottom_target() -> None:
    """! @brief 主车多候选选择使用图像底边作为 y 目标"""

    module = load_master()

    class HigherBottomBlob:
        def rect(self):
            return (120, 60, 80, 40)

        def cx(self):
            return 160

        def cy(self):
            return 120

        def area(self):
            return 1000

    class LowerBottomBlob:
        def rect(self):
            return (120, 0, 80, 20)

        def cx(self):
            return 160

        def cy(self):
            return 230

        def area(self):
            return 2000

    class FakeImage:
        def height(self):
            return 240

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [HigherBottomBlob(), LowerBottomBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), 320, 240
    )

    assert isinstance(best_blob, LowerBottomBlob)
    assert observation == (7, 0.0, -0.0, 2000.0)
