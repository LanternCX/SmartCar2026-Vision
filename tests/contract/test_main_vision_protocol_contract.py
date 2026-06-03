"""! @brief main.py 纯视觉协议契约测试"""

import inspect

from tests.test_support import (
    MODE_ACK,
    MODE_TCP,
    MODE_UDP,
    TOPIC_ASSISTANT_VISION_EVENT_REPORT,
    TOPIC_ASSISTANT_VISION_TASK_SYNC,
    TOPIC_LOCAL_VISION_VELOCITY,
    TOPIC_MASTER_VISION_EVENT_REPORT,
    TOPIC_MASTER_VISION_HOOK_SYNC,
    decode_assistant_vision_event_report_body,
    decode_frame,
    decode_master_vision_event_report_body,
    decode_velocity_body,
    encode_assistant_vision_task_sync_body,
    encode_frame,
    encode_master_vision_hook_sync_body,
    load_main_module,
    load_role_main_module,
)


def test_main_formats_minimal_observation_frames() -> None:
    """! @brief main.py 必须按固定短帧输出本地视觉速度"""
    module = load_main_module("vision_main_test_module_contract")
    frame = decode_frame(module.format_vision_frame(vx=-1.2, vy=0.0))

    assert frame is not None
    assert frame["mode"] == MODE_UDP
    assert frame["topic"] == TOPIC_LOCAL_VISION_VELOCITY
    assert frame["seq"] == 0
    assert decode_velocity_body(frame["body"]) == {
        "vx": -1.2,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }


def test_main_formats_zeroish_values_as_formal_zero_frame() -> None:
    """! @brief 协议输出必须把接近零的抖动收敛成正式零值字节"""
    module = load_main_module("vision_main_test_module_contract")
    frame = decode_frame(module.format_vision_frame(vx=-0.0004, vy=0.0004))

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }


def test_main_serialized_frame_keeps_only_current_velocity_packet_fields() -> None:
    """! @brief 正式主线速度帧只承载当前速度与 has_omega 标记"""
    module = load_main_module("vision_main_test_module_contract")
    frame = decode_frame(module.format_vision_frame(vx=1.25, vy=-0.5))

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 1.25,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }


def test_main_serialized_frame_does_not_require_follow_metadata() -> None:
    """! @brief 正式主线速度帧不附带额外跟随元信息"""
    module = load_main_module("vision_main_test_module_contract")
    frame = decode_frame(module.format_vision_frame(vx=0, vy=0))

    assert frame is not None
    assert frame["mode"] == MODE_UDP
    assert frame["topic"] == TOPIC_LOCAL_VISION_VELOCITY
    assert frame["seq"] == 0


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
    """! @brief 无目标时主线输出必须落到正式零速度短帧口径"""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=0, err_x=30, err_y=-40)
    frame = decode_frame(
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
    )

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }


def test_main_deadzone_hold_formats_formal_zero_velocity_frame() -> None:
    """! @brief 保持区输出必须继续使用正式零速度短帧口径"""
    module = load_main_module("vision_main_test_module_contract")
    result = module.build_follow_command(valid=1, err_x=0, err_y=0)
    frame = decode_frame(
        module.format_vision_frame(
            vx=result["command_vx"], vy=result["command_vy"]
        )
    )

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "has_omega": False,
    }


def test_assistant_formats_ack_and_reliable_event_frames() -> None:
    """! @brief 辅车视觉可靠协议使用固定 ACK 帧和事件帧"""
    module = load_main_module("vision_main_test_module_contract")
    ack_frame = decode_frame(module.format_ack_frame(12))
    event_frame = decode_frame(
        module.format_event_frame(12, module.EVENT_TARGET_FOUND, 180)
    )

    assert ack_frame == {
        "mode": MODE_ACK,
        "topic": TOPIC_ASSISTANT_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 8,
    }
    assert event_frame is not None
    assert event_frame["mode"] == MODE_TCP
    assert event_frame["topic"] == TOPIC_ASSISTANT_VISION_EVENT_REPORT
    assert event_frame["seq"] == 12
    assert decode_assistant_vision_event_report_body(event_frame["body"]) == {
        "event": module.EVENT_TARGET_FOUND,
        "value": 180,
    }


def test_assistant_sync_packet_uses_local_short_format() -> None:
    """! @brief 辅车视觉同步包使用本地任务同步短帧 body"""
    module = load_main_module("vision_main_test_module_contract")

    assert module.parse_sync_packet(
        encode_frame(
            MODE_TCP,
            TOPIC_ASSISTANT_VISION_TASK_SYNC,
            12,
            encode_assistant_vision_task_sync_body(2, 1, 1),
        )
    ) == {
        "reliable_seq": 12,
        "state": 2,
        "target": 1,
        "arg": 1,
    }
    assert module.parse_sync_packet(
        encode_frame(
            MODE_TCP,
            TOPIC_MASTER_VISION_HOOK_SYNC,
            12,
            encode_master_vision_hook_sync_body(7, 2, 1, 1),
        )
    ) is None


def test_master_formats_velocity_and_reliable_event_frames() -> None:
    """! @brief OpenART Vision master 使用速度流帧与可靠事件帧"""
    module = load_role_main_module("master", "vision_master_contract_module")
    velocity_frame = decode_frame(module.format_search_velocity_frame(1.0, -0.5))
    ack_frame = decode_frame(module.format_ack_frame(12))
    event_frame = decode_frame(
        module.format_event_frame(30, 7, module.EVENT_TARGET_FOUND, 300)
    )

    assert velocity_frame is not None
    assert velocity_frame["mode"] == MODE_UDP
    assert velocity_frame["topic"] == TOPIC_LOCAL_VISION_VELOCITY
    assert velocity_frame["seq"] == 0
    assert decode_velocity_body(velocity_frame["body"]) == {
        "vx": 1.0,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }
    assert ack_frame == {
        "mode": MODE_ACK,
        "topic": TOPIC_MASTER_VISION_HOOK_SYNC,
        "seq": 12,
        "body": b"\x00" * 8,
    }
    assert event_frame is not None
    assert event_frame["mode"] == MODE_TCP
    assert event_frame["topic"] == TOPIC_MASTER_VISION_EVENT_REPORT
    assert event_frame["seq"] == 30
    assert decode_master_vision_event_report_body(event_frame["body"]) == {
        "context_id": 7,
        "event": module.EVENT_TARGET_FOUND,
        "value": 300,
    }


def test_master_velocity_frame_keeps_only_vx_and_vy_fields() -> None:
    """! @brief 主车搜索速度流只携带 vx/vy, 不携带 omega"""
    module = load_role_main_module("master", "vision_master_contract_module")
    frame = decode_frame(module.format_search_velocity_frame(1.25, -0.5))

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 1.25,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }


def test_master_missing_target_velocity_frame_uses_configured_search_speed() -> None:
    """! @brief 主车无目标时经观测路径输出配置搜索速度短帧"""
    module = load_role_main_module("master", "vision_master_contract_module")

    class EmptyImage:
        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge):
            return []

    hook = module.MasterVisionHook()
    hook.handle_control_line(
        encode_frame(
            MODE_TCP,
            TOPIC_MASTER_VISION_HOOK_SYNC,
            12,
            encode_master_vision_hook_sync_body(7, 1, 1, 1),
        )
    )

    observation, best_blob = module.build_observation_from_image(
        hook, EmptyImage(), 320, 240
    )
    vx, vy = module.build_search_velocity_from_observation(observation, 240)
    frame = decode_frame(module.format_search_velocity_frame(vx, vy))
    expected = decode_frame(
        module.format_search_velocity_frame(
            module.MASTER_MISSING_SEARCH_VX,
            module.MASTER_MISSING_SEARCH_VY,
        )
    )

    assert best_blob is None
    assert frame is not None
    assert expected is not None
    assert frame["mode"] == MODE_UDP
    assert frame["topic"] == TOPIC_LOCAL_VISION_VELOCITY
    assert decode_velocity_body(frame["body"]) == decode_velocity_body(expected["body"])


def test_master_return_garage_events_use_reliable_event_topic() -> None:
    """! @brief 主车回库完成事件仍使用主车可靠事件短帧"""
    module = load_role_main_module("master", "vision_master_return_contract_module")

    finished_frame = decode_frame(
        module.format_event_frame(32, 7, module.EVENT_RETURN_GARAGE_FINISHED, 0)
    )

    assert int(module.MASTER_RETURN_GARAGE_LINE_HOOK_CONFIG_ID) == 5
    assert finished_frame is not None
    assert finished_frame["mode"] == MODE_TCP
    assert finished_frame["topic"] == TOPIC_MASTER_VISION_EVENT_REPORT
    assert decode_master_vision_event_report_body(finished_frame["body"]) == {
        "context_id": 7,
        "event": module.EVENT_RETURN_GARAGE_FINISHED,
        "value": 0,
    }
