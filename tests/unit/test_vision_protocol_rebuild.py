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


def test_compute_vertical_error_uses_fixed_target_y() -> None:
    """纵向误差必须相对固定目标点 95 计算."""
    module = load_main_module("vision_main_test_module_unit")

    assert (
        module.compute_vertical_error(blob_cy=110, target_y=module.FOLLOW_TARGET_Y)
        == 15
    )
    assert (
        module.compute_vertical_error(blob_cy=80, target_y=module.FOLLOW_TARGET_Y)
        == -15
    )


def test_format_vision_frame_outputs_position_request_with_compact_numbers() -> None:
    """速度式主线必须输出紧凑的 vx/vy 文本."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=1.2, vy=0.0)
    assert frame == "vx=1.2,vy=0"


def test_format_vision_frame_outputs_zero_position_request() -> None:
    """无目标或保持阶段时也必须输出零速度请求."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=0, vy=0)
    assert frame == "vx=0,vy=0"


def test_format_vision_frame_preserves_nonzero_dy_in_position_request() -> None:
    """纵向速度非零时也必须输出完整的 vx/vy 文本格式."""
    module = load_main_module("vision_main_test_module_unit")
    frame = module.format_vision_frame(vx=0, vy=0.6)
    assert frame == "vx=0,vy=0.6"


def test_build_follow_command_align_x_outputs_only_lateral_position_delta() -> None:
    """横向未对齐时只能输出横向速度量."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_align_x_sample(module)
    expected_vx = (float(module.FOLLOW_X_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_X
    )

    assert result["phase"] == "ALIGN_X"
    assert result["command_vx"] == pytest.approx(expected_vx)
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

    assert result["phase"] == "ALIGN_XY"
    assert result["command_vx"] == pytest.approx(expected_vx)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_area_deadzone_holds_after_x_aligned() -> None:
    """横向已对齐且纵向进入死区后必须输出零速度."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_hold_sample(module)

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(0.0)


def test_build_follow_command_align_y_outputs_only_longitudinal_delta() -> None:
    """ALIGN_Y 阶段必须只输出纵向速度量."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_align_y_sample(module)
    expected_vy = (float(module.FOLLOW_Y_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_Y
    )

    assert result["phase"] == "ALIGN_Y"
    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_align_y_preserves_reverse_direction() -> None:
    """纵向面积误差反向时也必须保留反向速度符号."""
    module = load_main_module("vision_main_test_module_unit")
    result = build_reverse_align_y_sample(module)
    expected_vy = -(float(module.FOLLOW_Y_DEADZONE_PX) + 1.0) * float(
        module.FOLLOW_CONTROL_KP_Y
    )

    assert result["phase"] == "ALIGN_Y"
    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(expected_vy)


def test_build_follow_command_marks_missing_target_as_zero_position_command() -> None:
    """没有有效目标时必须输出零速度命令."""
    module = load_main_module("vision_main_test_module_unit")
    result = module.build_follow_command(valid=0, err_x=30, err_y=40)

    assert result["command_vx"] == pytest.approx(0.0)
    assert result["command_vy"] == pytest.approx(0.0)
