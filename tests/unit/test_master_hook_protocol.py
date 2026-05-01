"""! @brief OpenART Vision master hook 协议测试"""

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

    assert reply == "a,12"
    assert module.format_observation_frame(*observation) == "o,7,0,0,180"


def test_master_repeated_sync_replies_ack_without_reapplying() -> None:
    """! @brief 重复同步包只重复确认, 不重复建立上下文"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=2, next_reliable_seq=30)

    assert hook.handle_control_line("s,12,7,1,1,1") == "a,12"
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)
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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

    assert module.format_observation_frame(*observation) == "o,7,0,0,180"


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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

    hook.accept_observation(observation)
    event_frame = hook.next_event_frame()
    hook.handle_control_line("a,12")
    now_ms[0] += 20

    assert event_frame == "r,30,7,6,180"
    assert hook.next_event_frame() == event_frame


def test_master_object_observation_uses_middle_and_lower_third_target() -> None:
    """! @brief 物体观测误差使用画面中线和下三分之二点"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(1, 160, 160, 250, 320, 240)

    assert observation == (7, 0.0, 0.0, 250.0)
    assert module.format_observation_frame(*observation) == "o,7,0,0,250"


def test_master_missing_target_outputs_zero_observation() -> None:
    """! @brief 无目标时主车视觉输出同一上下文下的零观测"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(0, 0, 0, 0, 320, 240)

    assert observation == (7, 0.0, 0.0, 0.0)
    assert module.format_observation_frame(*observation) == "o,7,0,0,0"


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
    observation = hook.build_observation(1, 160, 160, 150, 320, 240)

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

    weak = hook.build_observation(1, 160, 160, 99, 320, 240)
    offset = hook.build_observation(1, 180, 160, 150, 320, 240)
    hook.accept_observation(weak)
    hook.accept_observation(offset)

    assert hook.next_event_frame() is None


def test_master_hook_does_not_event_for_unsupported_hook_config() -> None:
    """! @brief 未支持的 hook 配置不创建 TARGET_FOUND 事件"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=1, next_reliable_seq=30)
    hook.handle_control_line("s,12,7,1,1,99")
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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
    observation = hook.build_observation(1, 160, 160, 180, 320, 240)

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

    module.write_data_line(uart, "o,7,0,0,0")

    assert uart.writes == ["o,7,0,0,0\r\n"]
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
