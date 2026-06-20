"""主车 main_v2 协议契约测试."""

from tests.test_support import (
    MODE_ACK,
    MODE_TCP,
    MODE_UDP,
    decode_assistant_vision_event_report_body,
    decode_assistant_vision_task_sync_body,
    decode_frame,
    decode_master_vision_event_report_body,
    decode_velocity_body,
    encode_frame,
    load_role_entry_module,
)


def load_master_v2():
    module = load_role_entry_module("master", "main_v2.py", "vision_master_v2_contract_module")
    module.reset_runtime_state()
    return module


def load_assistant_v2():
    module = load_role_entry_module(
        "assistant",
        "main_v2.py",
        "vision_assistant_v2_contract_module",
    )
    module.reset_runtime_state()
    return module


def test_master_main_v2_formats_velocity_and_reliable_event_frames() -> None:
    module = load_master_v2()
    velocity_frame = decode_frame(module.format_search_velocity_frame(1.0, -0.5))
    ack_frame = decode_frame(module.format_ack_frame(12))
    event_frame = decode_frame(
        module.format_event_frame(30, 7, module.Event.TARGET_FOUND, 300)
    )

    assert velocity_frame is not None
    assert velocity_frame["mode"] == MODE_UDP
    assert velocity_frame["topic"] == module.Topic.LOCAL_VISION_VELOCITY
    assert velocity_frame["seq"] == 0
    assert decode_velocity_body(velocity_frame["body"]) == {
        "vx": 1.0,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }
    assert ack_frame == {
        "mode": MODE_ACK,
        "topic": module.Topic.MASTER_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 10,
    }
    assert event_frame is not None
    assert event_frame["mode"] == MODE_TCP
    assert event_frame["topic"] == module.Topic.MASTER_VISION_EVENT_REPORT
    assert event_frame["seq"] == 30
    assert decode_master_vision_event_report_body(event_frame["body"]) == {
        "context_id": 7,
        "event": module.Event.TARGET_FOUND,
        "value": 300,
    }


def test_master_main_v2_velocity_frame_keeps_only_vx_and_vy_fields() -> None:
    module = load_master_v2()
    frame = decode_frame(module.format_search_velocity_frame(1.25, -0.5))

    assert frame is not None
    assert decode_velocity_body(frame["body"]) == {
        "vx": 1.25,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }


def test_master_main_v2_missing_target_velocity_frame_uses_configured_search_speed() -> None:
    module = load_master_v2()
    frame = decode_frame(
        module.format_search_velocity_frame(
            module.MASTER_MISSING_SEARCH_VX,
            module.MASTER_MISSING_SEARCH_VY,
        )
    )

    assert frame is not None
    assert frame["mode"] == MODE_UDP
    assert frame["topic"] == module.Topic.LOCAL_VISION_VELOCITY
    assert decode_velocity_body(frame["body"]) == {
        "vx": module.MASTER_MISSING_SEARCH_VX,
        "vy": module.MASTER_MISSING_SEARCH_VY,
        "omega": 0.0,
        "has_omega": False,
    }


def test_master_main_v2_return_line_aligned_event_uses_reliable_event_topic() -> None:
    module = load_master_v2()
    assert int(module.Event.RETURN_LINE_ALIGNED) == 10
    aligned_frame = decode_frame(
        module.format_event_frame(32, 7, module.Event.RETURN_LINE_ALIGNED, 160)
    )

    assert aligned_frame is not None
    assert aligned_frame["mode"] == MODE_TCP
    assert aligned_frame["topic"] == module.Topic.MASTER_VISION_EVENT_REPORT
    assert decode_master_vision_event_report_body(aligned_frame["body"]) == {
        "context_id": 7,
        "event": module.Event.RETURN_LINE_ALIGNED,
        "value": 160,
    }


def test_assistant_main_v2_formats_velocity_and_reliable_event_frames_with_master_style_api() -> None:
    module = load_assistant_v2()
    velocity_frame = decode_frame(module.format_search_velocity_frame(1.0, -0.5))
    ack_frame = decode_frame(module.format_ack_frame(12))
    event_frame = decode_frame(
        module.format_event_frame(30, module.Event.TARGET_FOUND, 300)
    )

    assert velocity_frame is not None
    assert velocity_frame["mode"] == MODE_UDP
    assert velocity_frame["topic"] == module.Topic.LOCAL_VISION_VELOCITY
    assert velocity_frame["seq"] == 0
    assert decode_velocity_body(velocity_frame["body"]) == {
        "vx": 1.0,
        "vy": -0.5,
        "omega": 0.0,
        "has_omega": False,
    }
    assert ack_frame == {
        "mode": MODE_ACK,
        "topic": module.Topic.ASSISTANT_VISION_TASK_SYNC,
        "seq": 12,
        "body": b"\x00" * 10,
    }
    assert event_frame is not None
    assert event_frame["mode"] == MODE_TCP
    assert event_frame["topic"] == module.Topic.ASSISTANT_VISION_EVENT_REPORT
    assert event_frame["seq"] == 30
    assert decode_assistant_vision_event_report_body(event_frame["body"]) == {
        "event": module.Event.TARGET_FOUND,
        "value": 300,
    }


def test_assistant_main_v2_return_line_aligned_event_uses_reliable_event_topic() -> None:
    module = load_assistant_v2()
    assert int(module.Event.RETURN_LINE_ALIGNED) == 10
    aligned_frame = decode_frame(
        module.format_event_frame(30, module.Event.RETURN_LINE_ALIGNED, 160)
    )

    assert aligned_frame is not None
    assert aligned_frame["mode"] == MODE_TCP
    assert aligned_frame["topic"] == module.Topic.ASSISTANT_VISION_EVENT_REPORT
    assert decode_assistant_vision_event_report_body(aligned_frame["body"]) == {
        "event": module.Event.RETURN_LINE_ALIGNED,
        "value": 160,
    }


def test_assistant_main_v2_exposes_master_style_task_sync_helpers() -> None:
    module = load_assistant_v2()
    frame = encode_frame(
        MODE_TCP,
        module.Topic.ASSISTANT_VISION_TASK_SYNC,
        18,
        module.encode_assistant_vision_task_sync_body(
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            module.pack_task_arg(module.Task.SEARCH, 2),
        ),
    )

    assert decode_assistant_vision_task_sync_body(
        module.encode_assistant_vision_task_sync_body(
            module.State.APPROACH_OBJECT,
            module.Target.OBJECT,
            module.pack_task_arg(module.Task.SEARCH, 2),
        )
    ) == {
        "state": module.State.APPROACH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.pack_task_arg(module.Task.SEARCH, 2),
    }
    assert module.parse_task_sync_packet(frame) == {
        "reliable_seq": 18,
        "state": module.State.APPROACH_OBJECT,
        "target": module.Target.OBJECT,
        "arg": module.pack_task_arg(module.Task.SEARCH, 2),
    }
    assert callable(module.build_search_target_point)
    assert callable(module.build_observation)
    assert callable(module.build_observation_and_candidates)
    assert callable(module.build_search_velocity_from_observation)
    assert callable(module.build_orbit_correction_velocity_from_observation)
