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


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def master_target_point(module, config_id=None):
    """! @brief 返回当前主车指定配置的搜索目标点"""

    if config_id is None:
        config_id = module.MASTER_SEARCH_HOOK_CONFIG_ID
    return module.build_search_target_point(IMAGE_WIDTH, IMAGE_HEIGHT, config_id)


def centered_master_observation(module, hook, area):
    """! @brief 构造命中当前目标点的观测"""

    target_x, target_y = master_target_point(module)
    return hook.build_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def centered_master_transport_observation(module, hook, area):
    """! @brief 构造命中搬运入口目标点的观测"""

    target_x, target_y = master_target_point(
        module,
        module.MASTER_TRANSPORT_HOOK_CONFIG_ID,
    )
    return hook.build_observation(
        1,
        target_x,
        target_y,
        area,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def finish_hook_control_line(module):
    """! @brief 构造主车收尾 hook 的同步包"""

    return "s,12,7,%d,%d,%d" % (
        int(module.STATE_TRANSPORT_OBJECT),
        int(module.TARGET_EDGE_LINE),
        int(module.MASTER_TRANSPORT_FINISH_HOOK_CONFIG_ID),
    )


def orbit_hook_control_line(module):
    """! @brief 构造主车绕行修正 hook 的同步包"""

    return "s,12,7,%d,%d,%d" % (
        int(module.STATE_ORBITING),
        int(module.TARGET_OBJECT),
        int(module.MASTER_ORBIT_HOOK_CONFIG_ID),
    )


def choose_outside_deadzone_offset(target, upper_bound, deadzone, clearance=1.0):
    """! @brief 在图像范围内构造一个稳定超出死区的偏移量"""

    required = float(deadzone) + float(clearance)
    target = float(target)
    upper_bound = float(upper_bound)
    if upper_bound - target >= required:
        return required
    if target >= required:
        return -required
    raise AssertionError("configured target leaves no room for out-of-deadzone sample")


def expected_axis_velocity(error, kp, min_speed, limit):
    """! @brief 按当前参数计算单轴期望速度"""

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


def expected_master_y_velocity(module, err_y):
    """! @brief 按当前参数计算纵向搜索期望速度"""

    err_y = float(err_y)
    if abs(err_y) <= float(module.MASTER_SEARCH_DEADZONE_Y_PX):
        return 0.0
    scaled_error = err_y * (
        float(module.MASTER_SEARCH_MAX_VY)
        / abs(float(module.MASTER_SEARCH_KP_Y))
        / float(IMAGE_HEIGHT)
    )
    return expected_axis_velocity(
        scaled_error,
        module.MASTER_SEARCH_KP_Y,
        module.MASTER_SEARCH_MIN_SPEED,
        module.MASTER_SEARCH_MAX_VY,
    )


class FinishHookBlob:
    """! @brief 收尾 hook 测试使用的固定色块"""

    def __init__(self, left, top, width, height, area):
        self._rect = (left, top, width, height)
        self._area = area

    def rect(self):
        return self._rect

    def cx(self):
        return self._rect[0] + self._rect[2] / 2

    def cy(self):
        return self._rect[1] + self._rect[3] / 2

    def area(self):
        return self._area


class FinishHookImage:
    """! @brief 同时模拟红色主目标和黄色环带统计的图像桩"""

    def __init__(self, blob, yellow_area_by_roi):
        self._blob = blob
        self._yellow_area_by_roi = dict(yellow_area_by_roi)
        self.crosses = []

    def height(self):
        return IMAGE_HEIGHT

    def find_blobs(
        self,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge,
        roi=None,
    ):
        _ = pixels_threshold
        _ = area_threshold
        _ = merge
        if roi is None:
            return [self._blob]
        area = self._yellow_area_by_roi.get(tuple(roi), 0)
        if area <= 0:
            return []
        return [FinishHookBlob(roi[0], roi[1], roi[2], roi[3], area)]

    def draw_cross(self, x, y):
        self.crosses.append((x, y))


def finish_hook_blob_and_ring_areas():
    """! @brief 返回收尾 hook 使用的色块和裁剪后的环带 roi 面积"""

    blob = FinishHookBlob(left=150, top=0, width=20, height=20, area=400)
    return (
        blob,
        {
            (145, 20, 30, 5): 150,
            (145, 0, 5, 20): 100,
            (170, 0, 5, 20): 100,
        },
    )


def test_master_sync_packet_records_context_and_replies_ack() -> None:
    """! @brief 主车视觉同步包使用可靠序号确认, 使用上下文编号建业务上下文"""

    module = load_master()
    hook = module.MasterVisionHook()

    reply = hook.handle_control_line("s,12,7,1,1,1")
    observation = centered_master_observation(module, hook, 180)

    assert reply == "a,12"
    assert observation == (7, 0.0, 0.0, 180.0)


def test_master_orbit_sync_switches_to_orbit_correction_context() -> None:
    """! @brief 主车绕行同步建立只输出速度修正的上下文"""

    module = load_master()
    hook = module.MasterVisionHook()

    assert hook.handle_control_line(orbit_hook_control_line(module)) == "a,12"

    assert hook.is_orbit_correction_context()
    assert hook.current_target_config_id() == module.MASTER_ORBIT_HOOK_CONFIG_ID


def test_master_orbit_correction_uses_independent_velocity_params_without_event() -> None:
    """! @brief 主车绕行修正使用独立速度参数且不产生可靠事件"""

    module = load_master()
    module.MASTER_ORBIT_KP_X = 0.2
    module.MASTER_ORBIT_KP_Y = -0.3
    module.MASTER_ORBIT_MIN_SPEED = 0.0
    module.MASTER_ORBIT_MAX_VX = 9.0
    module.MASTER_ORBIT_MAX_VY = 9.0
    module.MASTER_ORBIT_DEADZONE_X_PX = 3.0
    module.MASTER_ORBIT_DEADZONE_Y_PX = 3.0
    hook = module.MasterVisionHook(stable_frames=1)
    hook.handle_control_line(orbit_hook_control_line(module))
    target_x, target_y = master_target_point(
        module,
        module.MASTER_ORBIT_HOOK_CONFIG_ID,
    )
    err_x = 10.0
    err_y = 12.0
    observation = hook.build_observation(
        1,
        target_x + err_x,
        target_y + err_y,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    velocity = module.build_orbit_correction_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    )
    hook.accept_observation(observation)

    expected_y = err_y * (
        float(module.MASTER_ORBIT_MAX_VY)
        / abs(float(module.MASTER_ORBIT_KP_Y))
        / float(IMAGE_HEIGHT)
    ) * float(module.MASTER_ORBIT_KP_Y)
    assert velocity == pytest.approx((err_x * module.MASTER_ORBIT_KP_X, expected_y))
    assert hook.next_event_frame() is None


def test_master_orbit_correction_missing_target_outputs_zero_velocity() -> None:
    """! @brief 主车绕行修正无目标时输出零修正速度"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line(orbit_hook_control_line(module))
    observation = hook.build_observation(0, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert module.build_orbit_correction_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    ) == (0.0, 0.0)


def test_master_orbit_correction_zero_kp_outputs_zero_velocity() -> None:
    """! @brief 主车绕行修正增益为零时保持零修正"""

    module = load_master()
    module.MASTER_ORBIT_KP_X = 0.0
    module.MASTER_ORBIT_KP_Y = 0.0
    target_x, target_y = master_target_point(
        module,
        module.MASTER_ORBIT_HOOK_CONFIG_ID,
    )
    observation = module.MasterVisionHook().build_observation(
        1,
        target_x + module.MASTER_ORBIT_DEADZONE_X_PX + 10.0,
        target_y + module.MASTER_ORBIT_DEADZONE_Y_PX + 10.0,
        300,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert module.build_orbit_correction_velocity_from_observation(
        observation,
        IMAGE_HEIGHT,
    ) == (0.0, 0.0)


def test_master_repeated_sync_replies_ack_without_reapplying() -> None:
    """! @brief 重复同步包只重复确认, 不重复建立上下文"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=2, next_reliable_seq=30)

    assert hook.handle_control_line("s,12,7,1,1,1") == "a,12"
    observation = centered_master_observation(module, hook, 180)
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
    observation = centered_master_observation(module, hook, 180)

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
    observation = centered_master_observation(module, hook, 180)

    hook.accept_observation(observation)
    event_frame = hook.next_event_frame()
    hook.handle_control_line("a,12")
    now_ms[0] += 20

    assert event_frame == "r,30,7,6,180"
    assert hook.next_event_frame() == event_frame


def test_master_object_observation_uses_configured_target_point() -> None:
    """! @brief 物体观测误差使用当前配置目标点"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")
    target_x, target_y = master_target_point(module)

    observation = hook.build_observation(
        1, target_x, target_y, 250, IMAGE_WIDTH, IMAGE_HEIGHT
    )

    assert observation == (7, 0.0, 0.0, 250.0)


def test_master_transport_observation_uses_transport_target_point() -> None:
    """! @brief 搬运入口配置的目标底边必须切到推行阶段目标点"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line(
        "s,12,7,%d,1,%d"
        % (
            int(module.STATE_SEARCH_OBJECT),
            int(module.MASTER_TRANSPORT_HOOK_CONFIG_ID),
        )
    )
    search_target_x, search_target_y = master_target_point(
        module,
        module.MASTER_SEARCH_HOOK_CONFIG_ID,
    )
    transport_target_x, transport_target_y = master_target_point(
        module,
        module.MASTER_TRANSPORT_HOOK_CONFIG_ID,
    )

    search_observation = hook.build_observation(
        1,
        search_target_x,
        search_target_y,
        250,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    transport_observation = hook.build_observation(
        1,
        transport_target_x,
        transport_target_y,
        250,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert search_target_y == 210.0
    assert transport_target_y == 240.0
    assert search_observation == (7, 0.0, -30.0, 250.0)
    assert transport_observation == (7, 0.0, 0.0, 250.0)


def test_master_search_params_stay_within_qvga_bounds() -> None:
    """! @brief 主车搜索像素参数保持在当前图像范围内"""

    module = load_master()

    assert 0.0 <= float(module.MASTER_SEARCH_TARGET_X_PX) <= IMAGE_WIDTH
    assert 0.0 <= float(module.MASTER_SEARCH_TARGET_Y_PX) <= IMAGE_HEIGHT
    assert 0.0 <= float(module.MASTER_SEARCH_DEADZONE_X_PX) < IMAGE_WIDTH
    assert 0.0 <= float(module.MASTER_SEARCH_DEADZONE_Y_PX) < IMAGE_HEIGHT
    assert 0.0 <= float(module.OBJECT_X_TOLERANCE_PX) <= IMAGE_WIDTH
    assert 0.0 <= float(module.OBJECT_Y_TOLERANCE_PX) <= IMAGE_HEIGHT
    assert float(module.OBJECT_MIN_AREA) >= 0.0
    assert int(module.OBJECT_STABLE_FRAMES) >= 1


def test_master_target_found_window_matches_search_deadzone() -> None:
    """! @brief 主车命中窗口与停下修正的死区保持一致"""

    module = load_master()

    assert float(module.OBJECT_X_TOLERANCE_PX) == float(module.MASTER_SEARCH_DEADZONE_X_PX)
    assert float(module.OBJECT_Y_TOLERANCE_PX) == float(module.MASTER_SEARCH_DEADZONE_Y_PX)


def test_master_search_target_can_be_reconfigured(monkeypatch) -> None:
    """! @brief 主车搜索目标点改动后, 候选选择和观测误差都要跟着变化"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")
    monkeypatch.setattr(module, "MASTER_SEARCH_TARGET_X_PX", 80.0)
    monkeypatch.setattr(module, "MASTER_SEARCH_TARGET_Y_PX", 120.0)

    class FakeBlob:
        def __init__(self, left, top, width, height):
            self._rect = (left, top, width, height)

        def rect(self):
            return self._rect

        def cx(self):
            return self._rect[0] + self._rect[2] / 2

    first_blob = FakeBlob(70, 120, 20, 20)
    second_blob = FakeBlob(150, 0, 20, 20)

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [second_blob, first_blob]

    observation, best_blob = module.build_observation_from_image(
        hook,
        FakeImage(),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert best_blob is first_blob
    assert observation == (7, 0.0, 0.0, 400.0)


def test_master_missing_target_outputs_zero_observation() -> None:
    """! @brief 无目标时主车视觉输出同一上下文下的零观测"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation = hook.build_observation(0, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)

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


def test_master_blob_candidates_use_runtime_task_config() -> None:
    """! @brief 主车候选提取直接使用当前配置的任务表"""

    module = load_master()
    module.TASKS = (("runtime_target", (9, 8, 7, 6, 5, 4)),)

    class FakeBlob:
        def rect(self):
            return (10, 20, 30, 40)

        def cx(self):
            return 25

        def area(self):
            return 1234

    class FakeImage:
        def __init__(self):
            self.calls = []

        def height(self):
            return 100

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            self.calls.append((thresholds, pixels_threshold, area_threshold, merge))
            return [FakeBlob()]

    img = FakeImage()
    candidates = module.build_blob_candidates(img)

    assert img.calls == [([(9, 8, 7, 6, 5, 4)], 200, 200, True)]
    assert candidates[0][0] == "runtime_target"


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
    observation = centered_master_observation(module, hook, 150)

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

    weak = centered_master_observation(module, hook, 99)
    target_x, target_y = master_target_point(module)
    offset_x = choose_outside_deadzone_offset(
        target_x,
        IMAGE_WIDTH,
        hook.tolerance_x,
        clearance=1.0,
    )
    offset = hook.build_observation(
        1,
        target_x + offset_x,
        target_y,
        150,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    hook.accept_observation(weak)
    hook.accept_observation(offset)

    assert hook.next_event_frame() is None


def test_master_hook_does_not_event_for_unsupported_hook_config() -> None:
    """! @brief 未支持的 hook 配置不创建 TARGET_FOUND 事件"""

    module = load_master()
    hook = module.MasterVisionHook(stable_frames=1, next_reliable_seq=30)
    hook.handle_control_line("s,12,7,1,1,99")
    observation = centered_master_observation(module, hook, 180)

    hook.accept_observation(observation)

    assert hook.next_event_frame() is None


def test_master_transport_hook_emits_aligned_for_transport_config() -> None:
    """! @brief 搬运入口 hook 配置稳定满足条件后回报 ALIGNED"""

    module = load_master()
    hook = module.MasterVisionHook(
        stable_frames=1,
        next_reliable_seq=30,
    )
    hook.handle_control_line(
        "s,12,7,%d,1,%d"
        % (
            int(module.STATE_SEARCH_OBJECT),
            int(module.MASTER_TRANSPORT_HOOK_CONFIG_ID),
        )
    )
    observation = centered_master_transport_observation(module, hook, 180)

    hook.accept_observation(observation)

    assert hook.next_event_frame() == "r,30,7,7,180"


def test_master_transport_hook_keeps_search_velocity_output() -> None:
    """! @brief 搬运入口 hook 配置继续输出视觉速度"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line(
        "s,12,7,%d,1,%d"
        % (
            int(module.STATE_SEARCH_OBJECT),
            int(module.MASTER_TRANSPORT_HOOK_CONFIG_ID),
        )
    )

    class FakeBlob:
        def rect(self):
            return (150, 150, 20, 20)

        def cx(self):
            return 160

        def cy(self):
            return 160

        def area(self):
            return 300

    class FakeImage:
        def __init__(self):
            self.crosses = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    uart = FakeUART()
    module.process_search_frame(uart, hook, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT)

    assert uart.writes[0].startswith("v,")


def test_master_transport_finish_hook_does_not_arrive_on_yellow_contact_only() -> None:
    """! @brief 收尾 hook 只接触黄线时不能直接回报 ARRIVED"""

    module = load_master()
    hook = module.MasterVisionHook(next_reliable_seq=30)
    hook.handle_control_line(finish_hook_control_line(module))
    blob, ring_roi_areas = finish_hook_blob_and_ring_areas()
    positive_yellow_pixels = int(
        module.FINISH_HOOK_YELLOW_RATIO_THRESHOLD
        * sum(ring_roi_areas.values())
    ) + 1
    img = FinishHookImage(
        blob,
        {
            (145, 20, 30, 5): positive_yellow_pixels,
        },
    )
    uart = FakeUART()

    module.process_search_frame(uart, hook, img, IMAGE_WIDTH, IMAGE_HEIGHT)
    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(
            blob,
            {
                (145, 20, 30, 5): positive_yellow_pixels,
            },
        ),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert uart.writes[0] == "v,0,0\r\n"
    assert uart.writes[1] == "v,0,0\r\n"
    assert uart.writes == ["v,0,0\r\n", "v,0,0\r\n"]


def test_master_transport_finish_hook_emits_arrived_after_yellow_contact_then_clear() -> None:
    """! @brief 收尾 hook 需要经历接触黄线后再次完全脱离才回报 ARRIVED"""

    module = load_master()
    hook = module.MasterVisionHook(next_reliable_seq=30)
    hook.handle_control_line(finish_hook_control_line(module))
    blob, ring_roi_areas = finish_hook_blob_and_ring_areas()
    positive_yellow_pixels = int(
        module.FINISH_HOOK_YELLOW_RATIO_THRESHOLD
        * sum(ring_roi_areas.values())
    ) + 1
    uart = FakeUART()

    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(
            blob,
            {
                (145, 20, 30, 5): positive_yellow_pixels,
            },
        ),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(blob, {}),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(blob, {}),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert uart.writes[0] == "v,0,0\r\n"
    assert uart.writes[1] == "v,0,0\r\n"
    assert uart.writes[2] == "v,0,0\r\n"
    assert uart.writes[3] == "r,30,7,8,0\r\n"


def test_master_transport_finish_hook_does_not_arrive_when_yellow_ratio_is_not_enough() -> None:
    """! @brief 收尾 hook 黄色占比不足时不能回报 ARRIVED"""

    module = load_master()
    hook = module.MasterVisionHook(next_reliable_seq=30)
    hook.handle_control_line(finish_hook_control_line(module))
    blob, ring_roi_areas = finish_hook_blob_and_ring_areas()
    insufficient_yellow_pixels = int(
        module.FINISH_HOOK_YELLOW_RATIO_THRESHOLD
        * sum(ring_roi_areas.values())
    )
    uart = FakeUART()

    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(
            blob,
            {
                (145, 20, 30, 5): insufficient_yellow_pixels,
            },
        ),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(
            blob,
            {
                (145, 20, 30, 5): insufficient_yellow_pixels,
            },
        ),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert uart.writes == ["v,0,0\r\n", "v,0,0\r\n"]


def test_master_transport_finish_hook_keeps_search_velocity_output() -> None:
    """! @brief 收尾 hook 配置继续保留现有速度输出主线"""

    module = load_master()
    hook = module.MasterVisionHook()
    hook.handle_control_line(finish_hook_control_line(module))
    blob, _ = finish_hook_blob_and_ring_areas()
    uart = FakeUART()

    module.process_search_frame(
        uart,
        hook,
        FinishHookImage(blob, {}),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    assert uart.writes == ["v,0,0\r\n"]


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
    observation = centered_master_observation(module, hook, 180)

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
    observation = centered_master_observation(module, hook, 180)

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
    observation = centered_master_observation(module, hook, 180)

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
    observation = centered_master_observation(module, hook, 180)

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
    observation = centered_master_observation(module, hook, 180)

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
        hook, EmptyImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    velocity = module.build_search_velocity_from_observation(
        observation, IMAGE_HEIGHT
    )

    assert best_blob is None
    assert observation == (7, 0.0, 0.0, 0.0)
    assert velocity == (
        module.MASTER_MISSING_SEARCH_VX,
        module.MASTER_MISSING_SEARCH_VY,
    )


def test_master_target_point_generates_p_search_velocity() -> None:
    """! @brief 有目标时主车搜索按当前目标点误差生成 P 控制量"""

    module = load_master()
    target_x, target_y = master_target_point(module)
    err_x = choose_outside_deadzone_offset(
        target_x,
        IMAGE_WIDTH,
        module.MASTER_SEARCH_DEADZONE_X_PX,
        clearance=15.0,
    )
    err_y = choose_outside_deadzone_offset(
        target_y,
        IMAGE_HEIGHT,
        module.MASTER_SEARCH_DEADZONE_Y_PX,
        clearance=15.0,
    )
    center_x = target_x + err_x
    normalized_bottom = target_y + err_y
    blob_top = IMAGE_HEIGHT - normalized_bottom

    class FakeBlob:
        def rect(self):
            return (center_x - 20, blob_top, 40, 20)

        def cx(self):
            return center_x

        def cy(self):
            return blob_top + 10

        def area(self):
            return 300

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    velocity = module.build_search_velocity_from_observation(
        observation, IMAGE_HEIGHT
    )

    assert best_blob is not None
    assert observation == pytest.approx((7.0, err_x, err_y, 300.0))
    assert velocity == pytest.approx(
        (
            expected_axis_velocity(
                err_x,
                module.MASTER_SEARCH_KP_X,
                module.MASTER_SEARCH_MIN_SPEED,
                module.MASTER_SEARCH_MAX_VX,
            ),
            expected_master_y_velocity(module, err_y),
        )
    )


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
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [self._blob]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    far_observation, _ = module.build_observation_from_image(
        hook, FakeImage(FarBlob()), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    close_observation, _ = module.build_observation_from_image(
        hook, FakeImage(CloseBlob()), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    far_velocity = module.build_search_velocity_from_observation(
        far_observation, IMAGE_HEIGHT
    )
    close_velocity = module.build_search_velocity_from_observation(
        close_observation, IMAGE_HEIGHT
    )

    assert close_velocity[1] < far_velocity[1]


def test_master_search_velocity_deadzone_zeroes_each_axis() -> None:
    """! @brief 主车搜索通过图像观测路径对当前死区边界输出零量"""

    module = load_master()
    target_x, target_y = master_target_point(module)
    normalized_bottom = target_y + float(module.MASTER_SEARCH_DEADZONE_Y_PX)
    blob_top = IMAGE_HEIGHT - normalized_bottom

    class FakeBlob:
        def rect(self):
            return (target_x - 40, blob_top, 80, 10)

        def cx(self):
            return target_x + float(module.MASTER_SEARCH_DEADZONE_X_PX)

        def cy(self):
            return blob_top + 5

        def area(self):
            return 150

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    velocity = module.build_search_velocity_from_observation(
        observation, IMAGE_HEIGHT
    )

    assert best_blob is not None
    assert velocity == (0.0, 0.0)


def test_master_search_velocity_applies_min_speed_outside_deadzone() -> None:
    """! @brief 主车搜索误差超出死区时速度幅值不能低于最小速度"""

    module = load_master()

    positive = module.build_search_velocity_from_error(
        float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1,
        float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1,
        IMAGE_HEIGHT,
    )
    negative = module.build_search_velocity_from_error(
        -(float(module.MASTER_SEARCH_DEADZONE_X_PX) + 0.1),
        -(float(module.MASTER_SEARCH_DEADZONE_Y_PX) + 0.1),
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


def test_master_search_velocity_clamps_vx_and_vy() -> None:
    """! @brief 主车搜索通过观测路径按轴限幅"""

    module = load_master()

    positive = module.build_search_velocity_from_observation(
        (7, 999.0, 999.0, 300.0), IMAGE_HEIGHT
    )
    negative = module.build_search_velocity_from_observation(
        (7, -999.0, -999.0, 300.0), IMAGE_HEIGHT
    )

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

    observation = centered_master_observation(module, hook, 150)
    hook.accept_observation(observation)

    assert hook.next_event_frame() == "r,30,7,6,150"


def test_master_search_control_does_not_require_marker_span_or_min_corners() -> None:
    """! @brief 主车搜索控制不依赖最小外接旋转矩形或 marker_span"""

    module = load_master()
    target_x, target_y = master_target_point(module)
    blob_top = IMAGE_HEIGHT - target_y

    class CenterOnlyBlob:
        def rect(self):
            return (target_x - 40, blob_top, 80, 1)

        def cx(self):
            return target_x

        def cy(self):
            return blob_top

        def area(self):
            return 4800

    class CenterOnlyImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [CenterOnlyBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, CenterOnlyImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    velocity = module.build_search_velocity_from_observation(
        observation, IMAGE_HEIGHT
    )

    assert best_blob is not None
    assert velocity == (0.0, 0.0)


def test_master_search_frame_outputs_velocity_without_hook_context() -> None:
    """! @brief 主车速度流不依赖 hook 上下文, 直接按当前目标点误差输出"""

    module = load_master()
    target_x, target_y = master_target_point(module)
    err_x = choose_outside_deadzone_offset(
        target_x,
        IMAGE_WIDTH,
        module.MASTER_SEARCH_DEADZONE_X_PX,
        clearance=25.0,
    )
    center_x = target_x + err_x
    blob_top = IMAGE_HEIGHT - target_y

    class FakeBlob:
        def rect(self):
            return (center_x - 10, blob_top, 20, 1)

        def cx(self):
            return center_x

        def cy(self):
            return blob_top

        def area(self):
            return 500

        def min_corners(self):
            raise AssertionError("min_corners must not be used")

    class FakeImage:
        def __init__(self):
            self.crosses = []

        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    uart = FakeUART()
    hook = module.MasterVisionHook()
    img = FakeImage()

    module.process_search_frame(uart, hook, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    expected_vx = expected_axis_velocity(
        err_x,
        module.MASTER_SEARCH_KP_X,
        module.MASTER_SEARCH_MIN_SPEED,
        module.MASTER_SEARCH_MAX_VX,
    )
    assert uart.writes == [f"v,{module.compact_number(expected_vx)},0\r\n"]
    assert img.crosses[-1] == (center_x, blob_top)
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
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

    uart = FakeUART()
    hook = module.MasterVisionHook()
    img = FakeImage()

    module.process_search_frame(uart, hook, img, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert img.crosses[-1] == (200, 145)


def test_master_hook_event_uses_configured_target_y_not_blob_center_y() -> None:
    """! @brief TARGET_FOUND 经图像路径使用当前目标点 y, 不使用中心 y"""

    module = load_master()
    hook = module.MasterVisionHook(
        min_area=100,
        tolerance_x=5,
        tolerance_y=5,
        stable_frames=1,
        next_reliable_seq=30,
    )
    hook.handle_control_line("s,12,7,1,1,1")
    target_x, target_y = master_target_point(module)
    blob_top = IMAGE_HEIGHT - target_y

    class FakeBlob:
        def rect(self):
            return (target_x - 40, blob_top, 80, 2)

        def cx(self):
            return target_x

        def cy(self):
            return target_y - 40

        def area(self):
            return 4800

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    observation, _ = module.build_observation_from_image(
        hook, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    hook.accept_observation(observation)

    assert observation == (7, 0.0, 0.0, 4800.0)
    assert hook.next_event_frame() == "r,30,7,6,4800"


def test_master_candidate_selection_uses_configured_target_point() -> None:
    """! @brief 主车多候选选择使用当前配置目标点"""

    module = load_master()
    target_x, target_y = master_target_point(module)
    far_offset = choose_outside_deadzone_offset(
        target_y,
        IMAGE_HEIGHT,
        0.0,
        clearance=20.0,
    )

    class HigherBottomBlob:
        def rect(self):
            return (target_x - 40, IMAGE_HEIGHT - target_y, 80, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - target_y + 10

        def area(self):
            return 1000

    class LowerBottomBlob:
        def rect(self):
            return (target_x - 40, IMAGE_HEIGHT - (target_y + far_offset), 80, 20)

        def cx(self):
            return target_x

        def cy(self):
            return IMAGE_HEIGHT - (target_y + far_offset) + 10

        def area(self):
            return 2000

    class FakeImage:
        def height(self):
            return IMAGE_HEIGHT

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [HigherBottomBlob(), LowerBottomBlob()]

    hook = module.MasterVisionHook()
    hook.handle_control_line("s,12,7,1,1,1")

    observation, best_blob = module.build_observation_from_image(
        hook, FakeImage(), IMAGE_WIDTH, IMAGE_HEIGHT
    )

    assert isinstance(best_blob, HigherBottomBlob)
    assert observation == (7, 0.0, 0.0, 1000.0)
