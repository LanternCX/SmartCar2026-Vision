"""! @brief main.py 纯视觉协议契约测试"""

import inspect

from tests.test_support import load_main_module, load_role_main_module


def test_main_formats_minimal_observation_frames() -> None:
    """! @brief main.py 必须按 v 短包协议输出最小文本帧"""
    module = load_main_module("vision_main_test_module_contract")

    assert module.format_vision_frame(vx=-1.2, vy=0.0) == "v,-1.2,0"
    assert module.format_vision_frame(vx=0, vy=0) == "v,0,0"


def test_main_formats_zeroish_values_as_formal_zero_frame() -> None:
    """! @brief 协议输出必须把接近零的抖动收敛成正式零值文本"""
    module = load_main_module("vision_main_test_module_contract")

    assert module.format_vision_frame(vx=-0.0004, vy=0.0004) == "v,0,0"


def test_main_serialized_frame_keeps_only_current_velocity_packet_fields() -> None:
    """! @brief 正式主线文本只包含 v 包类型和两个速度字段"""
    module = load_main_module("vision_main_test_module_contract")
    frame = module.format_vision_frame(vx=1.25, vy=-0.5)
    parts = frame.split(",")

    assert parts == ["v", "1.25", "-0.5"]
    assert "=" not in frame
    assert "dx=" not in frame
    assert "dy=" not in frame
    assert "lock=" not in frame


def test_main_serialized_frame_does_not_require_follow_metadata() -> None:
    """! @brief 正式主线文本只输出 v 速度短包, 不附带 follow 协议元信息"""
    module = load_main_module("vision_main_test_module_contract")
    frame = module.format_vision_frame(vx=0, vy=0)

    assert frame == "v,0,0"
    assert "follow=" not in frame
    assert "seq=" not in frame
    assert "valid=" not in frame


def test_main_format_vision_frame_signature_keeps_only_minimal_inputs() -> None:
    """! @brief format_vision_frame 只应包含最小输入参数"""
    module = load_main_module("vision_main_test_module_contract")
    signature = inspect.signature(module.format_vision_frame)
    parameter_names = list(signature.parameters)

    assert "vx" in parameter_names
    assert "vy" in parameter_names
    assert len(parameter_names) == 2


def test_main_build_follow_command_returns_velocity_deltas() -> None:
    """! @brief build_follow_command 必须统一返回速度量字段"""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=0, err_x=0, err_y=0)

    assert "command_vx" in result
    assert "command_vy" in result
    assert "command_dx" not in result
    assert "command_dy" not in result


def test_main_missing_target_formats_formal_zero_velocity_frame() -> None:
    """! @brief 无目标时主线输出必须落到正式零速度文本口径"""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=0, err_x=30, err_y=-40)

    assert (
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
        == "v,0,0"
    )


def test_main_deadzone_hold_formats_formal_zero_velocity_frame() -> None:
    """! @brief 保持区输出必须继续使用正式零速度文本口径"""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=1, err_x=0, err_y=0)

    assert (
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
        == "v,0,0"
    )



def test_master_formats_observation_and_reliable_event_frames() -> None:
    """! @brief OpenART Vision master 使用观测包和可靠事件包"""

    module = load_role_main_module("master", "vision_master_contract_module")

    assert module.format_observation_frame(7, 1.0, -0.5, 300) == "o,7,1,-0.5,300"
    assert module.format_ack_frame(12) == "a,12"
    assert module.format_event_frame(30, 7, module.EVENT_TARGET_FOUND, 300) == "r,30,7,6,300"
