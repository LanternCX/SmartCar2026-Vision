"""main.py 纯视觉协议契约测试."""

import inspect

from tests.test_support import load_main_module


def test_main_formats_minimal_observation_frames() -> None:
    """main.py 必须按速度式协议输出最小文本帧."""
    module = load_main_module("vision_main_test_module_contract")

    assert module.format_vision_frame(vx=-1.2, vy=0.0) == "vx=-1.2,vy=0"
    assert module.format_vision_frame(vx=0, vy=0) == "vx=0,vy=0"


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
