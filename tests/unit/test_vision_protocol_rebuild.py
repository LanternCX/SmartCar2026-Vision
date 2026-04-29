"""纯视觉上报重构的主机侧单元测试."""

import pytest

from tests.test_support import load_main_module


def build_align_x_sample(module):
    """构造一个稳定的横向阶段样例."""
    return module.build_follow_command(
        valid=1,
        err_x=float(module.FOLLOW_X_DEADZONE_PX) + 1.0,
        err_y=0.0,
    )


def build_hold_sample(module):
    """构造一个稳定的保持区样例."""
    return module.build_follow_command(
        valid=1,
        err_x=0.0,
        err_y=0.0,
    )


def build_align_y_sample(module):
    """构造一个稳定的纵向阶段样例."""
    return module.build_follow_command(
        valid=1,
        err_x=0.0,
        err_y=float(module.FOLLOW_Y_DEADZONE_PX) + 1.0,
    )


def build_reverse_align_y_sample(module):
    """构造一个反向纵向阶段样例."""
    return module.build_follow_command(
        valid=1,
        err_x=0.0,
        err_y=-(float(module.FOLLOW_Y_DEADZONE_PX) + 1.0),
    )


def build_dual_axis_sample(module):
    """构造一个横纵同时超出死区的样例."""
    return module.build_follow_command(
        valid=1,
        err_x=float(module.FOLLOW_X_DEADZONE_PX) + 1.0,
        err_y=float(module.FOLLOW_Y_DEADZONE_PX) + 1.0,
    )


def build_reverse_align_x_sample(module):
    """构造一个反向横向阶段样例."""
    return module.build_follow_command(
        valid=1,
        err_x=-(float(module.FOLLOW_X_DEADZONE_PX) + 5.0),
        err_y=0.0,
    )


def build_cross_direction_dual_axis_sample(module):
    """构造一个横纵方向相反的双轴偏差样例."""
    return module.build_follow_command(
        valid=1,
        err_x=-(float(module.FOLLOW_X_DEADZONE_PX) + 5.0),
        err_y=float(module.FOLLOW_Y_DEADZONE_PX) + 5.0,
    )


def test_follow_target_y_uses_calibrated_midpoint() -> None:
    """纵向目标尺度量使用现场标定的中点 45."""
    module = load_main_module("vision_main_test_module_unit")

    assert module.FOLLOW_TARGET_Y > 0


def test_compute_marker_span_uses_trapezoid_top_bottom_average() -> None:
    """距离环尺度量必须取梯形上底和下底的平均长度."""
    module = load_main_module("vision_main_test_module_unit")

    corners = ((10, 10), (30, 10), (36, 34), (4, 34))

    assert module.compute_marker_span(corners) == pytest.approx(26.0)


def test_get_marker_corners_prefers_min_area_rect_big_corners() -> None:
    """角点来源必须优先取最小外接旋转矩形的大角."""
    module = load_main_module("vision_main_test_module_unit")

    class FakeBlob:
        def corners(self):
            return ((11, 20), (18, 21), (30, 20), (35, 27))

        def min_corners(self):
            return ((10, 20), (30, 20), (36, 50), (4, 50))

    assert module.get_marker_corners(FakeBlob()) == (
        (10, 20),
        (30, 20),
        (36, 50),
        (4, 50),
    )


def test_compute_vertical_error_uses_target_span() -> None:
    """纵向误差必须相对目标尺度量计算."""
    module = load_main_module("vision_main_test_module_unit")

    assert (
        module.compute_vertical_error(
            marker_span=module.FOLLOW_TARGET_Y + 15,
            target_span=module.FOLLOW_TARGET_Y,
        )
        == 15
    )
    assert (
        module.compute_vertical_error(
            marker_span=module.FOLLOW_TARGET_Y - 15,
            target_span=module.FOLLOW_TARGET_Y,
        )
        == -15
    )


def test_build_blob_candidates_reports_corner_based_span() -> None:
    """候选目标必须携带由角点计算出的梯形尺度量."""
    module = load_main_module("vision_main_test_module_unit")

    class FakeBlob:
        def rect(self):
            return (10, 20, 20, 30)

        def corners(self):
            return ((11, 20), (18, 21), (30, 20), (35, 27))

        def min_corners(self):
            return ((10, 20), (30, 20), (36, 50), (4, 50))

        def cx(self):
            return 20

        def cy(self):
            return 35

    class FakeImage:
        def height(self):
            return 100

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return [FakeBlob()]

    candidates = module.build_blob_candidates(FakeImage())

    assert len(candidates) == 1
    candidate = candidates[0]
    pixel_x = candidate[1]
    pixel_y = candidate[2]
    bottom = candidate[3]
    marker_span = candidate[4]
    assert pixel_x == 20
    assert pixel_y == 35
    assert bottom == 80
    assert marker_span == pytest.approx(26.0)


def test_draw_selected_marker_draws_corners_without_bounding_box() -> None:
    """调试显示必须画四个角点而不是外接框."""
    module = load_main_module("vision_main_test_module_unit")

    class FakeBlob:
        def corners(self):
            return ((11, 20), (18, 21), (30, 20), (35, 27))

        def min_corners(self):
            return ((10, 20), (30, 20), (36, 50), (4, 50))

    class FakeImage:
        def __init__(self):
            self.crosses = []
            self.rectangles = []

        def draw_cross(self, x, y):
            self.crosses.append((x, y))

        def draw_rectangle(self, rect):
            self.rectangles.append(rect)

    img = FakeImage()
    blob = FakeBlob()

    module.draw_selected_marker(img=img, blob=blob, pixel_x=20, pixel_y=35)

    assert img.rectangles == []
    assert len(img.crosses) == 5
    assert (20, 35) in img.crosses
    for point in ((10, 20), (30, 20), (36, 50), (4, 50)):
        assert point in img.crosses




def test_format_vision_frame_outputs_short_velocity_packet_with_compact_numbers() -> None:
    """视觉主线必须输出紧凑的 v 短包文本."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=1.2, vy=0.0)
    assert frame == "v,1.2,0"
    assert frame.split(",") == ["v", "1.2", "0"]
    assert "=" not in frame


def test_format_vision_frame_outputs_zero_short_velocity_packet() -> None:
    """无目标或保持阶段时也必须输出零速度短包."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=0, vy=0)
    assert frame == "v,0,0"


def test_format_vision_frame_preserves_nonzero_vy_in_short_velocity_packet() -> None:
    """纵向速度非零时也必须输出完整的 v 短包文本."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=0, vy=0.6)
    assert frame == "v,0,0.6"


def test_build_follow_command_align_x_outputs_only_lateral_velocity_delta() -> None:
    """横向未对齐时只能输出横向速度量."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_align_x_sample(module)
    expected_vx = (float(module.FOLLOW_X_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_X
    )

    assert result["command_vx"] == pytest.approx(expected_vx)
    assert result["command_vy"] == pytest.approx(0.0)


def test_build_follow_command_align_x_preserves_reverse_direction() -> None:
    """横向偏差反向时也必须保留反向速度符号."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_reverse_align_x_sample(module)

    assert result["command_vx"] < 0
    assert result["command_vy"] == pytest.approx(0.0)


def test_build_follow_command_outputs_both_axes_when_both_errors_exist() -> None:
    """横纵都超出死区时必须同时输出两个方向的速度量."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_dual_axis_sample(module)
    expected_vx = (float(module.FOLLOW_X_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_X
    )
    expected_vy = (float(module.FOLLOW_Y_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_Y
    )

    assert result["command_vx"] == pytest.approx(expected_vx)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_area_deadzone_holds_after_x_aligned() -> None:
    """横向已对齐且纵向进入死区后必须输出零速度."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_hold_sample(module)

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(0.0)


def test_build_follow_command_exact_deadzone_boundary_still_holds() -> None:
    """刚好落在死区边界时也必须继续保持零速度."""
    module = load_main_module("vision_main_test_module_unit")
    result = module.build_follow_command(
        valid=1,
        err_x=float(module.FOLLOW_X_DEADZONE_PX),
        err_y=-float(module.FOLLOW_Y_DEADZONE_PX),
    )

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(0.0)


def test_build_follow_command_align_y_outputs_only_longitudinal_velocity_delta() -> None:
    """ALIGN_Y 阶段必须只输出纵向速度量."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_align_y_sample(module)
    expected_vy = (float(module.FOLLOW_Y_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_Y
    )

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_align_y_preserves_reverse_direction() -> None:
    """纵向面积误差反向时也必须保留反向速度符号."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_reverse_align_y_sample(module)
    expected_vy = -(float(module.FOLLOW_Y_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_Y
    )

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_dual_axis_keeps_axes_independent() -> None:
    """双轴同时偏差时, 横向反向不应把纵向输出一起压掉."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_cross_direction_dual_axis_sample(module)

    assert result["command_vx"] < 0
    assert result["command_vy"] != 0


def test_build_follow_command_marks_missing_target_as_zero_velocity_command() -> None:
    """没有有效目标时必须输出零速度命令."""
    module = load_main_module("vision_main_test_module_unit")
    result = module.build_follow_command(valid=0, err_x=30, err_y=40)

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(0.0)

def test_build_follow_command_keeps_phase_names_for_behavior_paths() -> None:
    """阶段名必须继续标识无目标、保持、单轴和双轴路径."""
    module = load_main_module("vision_main_test_module_unit")

    assert module.build_follow_command(valid=0, err_x=30, err_y=40)["phase"] == "MARKER_MISSING"
    assert build_hold_sample(module)["phase"] == "CENTER_HOLD"
    assert build_align_x_sample(module)["phase"] == "ALIGN_X"
    assert build_align_y_sample(module)["phase"] == "ALIGN_Y"
    assert build_dual_axis_sample(module)["phase"] == "ALIGN_XY"


def test_build_follow_command_keeps_longitudinal_velocity_limit() -> None:
    """纵向速度修正量必须继续受单帧上限约束."""
    module = load_main_module("vision_main_test_module_unit")

    forward = module.build_follow_command(valid=1, err_x=0, err_y=1000)
    backward = module.build_follow_command(valid=1, err_x=0, err_y=-1000)

    assert forward["phase"] == "ALIGN_Y"
    assert backward["phase"] == "ALIGN_Y"
    assert forward["command_vy"] == pytest.approx(-float(module.FOLLOW_CONTROL_MAX_Y))
    assert backward["command_vy"] == pytest.approx(float(module.FOLLOW_CONTROL_MAX_Y))


def test_follow_command_values_flow_into_short_packet_without_value_change() -> None:
    """控制结果进入短包时只能改变协议外壳, 不能改变速度数值."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_dual_axis_sample(module)

    frame = module.format_vision_frame(
        vx=result["command_vx"],
        vy=result["command_vy"],
    )

    assert result["command_vx"] == pytest.approx(0.64)
    assert result["command_vy"] == pytest.approx(-1.35)
    assert frame == "v,0.64,-1.35"


def test_write_line_appends_crlf_to_short_packet(monkeypatch) -> None:
    """串口发送必须继续使用 CRLF 作为单行结束符."""
    module = load_main_module("vision_main_test_module_unit")
    monkeypatch.setattr(module, "WRITE_DELAY_S", 0)

    class FakeUART:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)
            return len(data)

    uart = FakeUART()

    module.write_line(uart, module.format_vision_frame(vx=0, vy=0))

    assert uart.writes == ["v,0,0\r\n"]
