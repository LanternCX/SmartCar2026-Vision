"""main.py 纯视觉协议契约测试."""

import inspect

from tests.test_support import load_main_module


def test_main_formats_minimal_observation_frames() -> None:
    """main.py 必须按速度式协议输出最小文本帧."""
    module = load_main_module("vision_main_test_module_contract")

    assert module.format_vision_frame(vx=-1.2, vy=0.0) == "vx=-1.2,vy=0"
    assert module.format_vision_frame(vx=0, vy=0) == "vx=0,vy=0"


def test_main_formats_zeroish_values_as_formal_zero_frame() -> None:
    """协议输出必须把接近零的抖动收敛成正式零值文本."""
    module = load_main_module("vision_main_test_module_contract")

    assert module.format_vision_frame(vx=-0.0004, vy=0.0004) == "vx=0,vy=0"


def test_main_serialized_frame_keeps_only_current_velocity_keys() -> None:
    """正式主线文本只保留 vx/vy, 不把旧字段写回输出帧."""
    module = load_main_module("vision_main_test_module_contract")
    frame = module.format_vision_frame(vx=1.25, vy=-0.5)
    field_names = {part.split("=", 1)[0] for part in frame.split(",")}

    assert field_names == {"vx", "vy"}
    assert "dx=" not in frame
    assert "dy=" not in frame


def test_main_serialized_frame_does_not_require_follow_metadata() -> None:
    """当前正式主线文本只输出 vx/vy, 不附带 follow 协议元信息."""
    module = load_main_module("vision_main_test_module_contract")
    frame = module.format_vision_frame(vx=0, vy=0)

    assert frame == "vx=0,vy=0"
    assert "follow=" not in frame
    assert "seq=" not in frame
    assert "valid=" not in frame


def test_main_format_vision_frame_signature_keeps_only_minimal_inputs() -> None:
    """format_vision_frame 只应保留最小输入参数."""
    module = load_main_module("vision_main_test_module_contract")
    signature = inspect.signature(module.format_vision_frame)
    parameter_names = list(signature.parameters)

    assert "vx" in parameter_names
    assert "vy" in parameter_names
    assert len(parameter_names) == 2


def test_main_build_follow_command_returns_position_deltas() -> None:
    """build_follow_command 必须统一返回速度量字段."""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=0, err_x=0, err_y=0)

    assert "command_vx" in result
    assert "command_vy" in result
    assert "command_dx" not in result
    assert "command_dy" not in result


def test_main_missing_target_formats_formal_zero_velocity_frame() -> None:
    """无目标时主线输出必须落到正式零速度文本口径."""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=0, err_x=30, err_y=-40)

    assert (
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
        == "vx=0,vy=0"
    )


def test_main_deadzone_hold_formats_formal_zero_velocity_frame() -> None:
    """保持区输出必须继续使用正式零速度文本口径."""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=1, err_x=0, err_y=0)

    assert (
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
        == "vx=0,vy=0"
    )
