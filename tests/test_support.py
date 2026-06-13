"""测试辅助函数."""

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_MAIN_PATH = ROOT / "assistant" / "main.py"

FRAME_BODY_SIZE = 8
FRAME_HEAD = 0xA5
FRAME_SIZE = 13
MODE_UDP = 0x01
MODE_TCP = 0x02
MODE_ACK = 0x03

TOPIC_LOCAL_VISION_VELOCITY = 0x01
TOPIC_MASTER_VISION_HOOK_SYNC = 0x10
TOPIC_ASSISTANT_VISION_TASK_SYNC = 0x11
TOPIC_MASTER_VISION_EVENT_REPORT = 0x12
TOPIC_ASSISTANT_VISION_EVENT_REPORT = 0x13

_SCALE = 1000.0


def role_entry_path(role: str, entry_name: str = "main.py") -> Path:
    """返回指定角色的视觉入口路径."""

    return ROOT / role / entry_name


def role_main_path(role: str) -> Path:
    """返回指定角色的默认视觉入口路径."""

    return role_entry_path(role, "main.py")


def load_main_module(module_name: str):
    """按真实模块导入方式加载辅车 main.py, 但不触发运行入口."""

    return load_role_main_module("assistant", module_name)


def load_role_main_module(role: str, module_name: str):
    """按真实模块导入方式加载指定角色 main.py, 但不触发运行入口."""

    return load_role_entry_module(role, "main.py", module_name)


def load_role_entry_module(role: str, entry_name: str, module_name: str):
    """按真实模块导入方式加载指定角色入口, 但不触发运行入口."""

    _install_board_runtime_stubs()
    spec = importlib.util.spec_from_file_location(
        module_name,
        role_entry_path(role, entry_name),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_board_runtime_stubs() -> None:
    """为主机测试补齐 OpenART 入口需要的最小模块桩."""

    if "sensor" not in sys.modules:
        sensor_module = types.ModuleType("sensor")
        sensor_module.RGB565 = "RGB565"
        sensor_module.QVGA = "QVGA"
        sensor_module.reset = lambda: None
        sensor_module.set_pixformat = lambda *_args, **_kwargs: None
        sensor_module.set_framesize = lambda *_args, **_kwargs: None
        sensor_module.set_vflip = lambda *_args, **_kwargs: None
        sensor_module.set_hmirror = lambda *_args, **_kwargs: None
        sensor_module.skip_frames = lambda *_args, **_kwargs: None
        sensor_module.set_auto_gain = lambda *_args, **_kwargs: None
        sensor_module.set_auto_whitebal = lambda *_args, **_kwargs: None
        sensor_module.set_auto_exposure = lambda *_args, **_kwargs: None
        sensor_module.width = lambda: 320
        sensor_module.height = lambda: 240
        sensor_module.snapshot = lambda: None
        sys.modules["sensor"] = sensor_module

    if "tf" not in sys.modules:
        tf_module = types.ModuleType("tf")
        tf_module.load = lambda path: path
        tf_module.detect = lambda _net, _img: ()
        sys.modules["tf"] = tf_module

    if "image" not in sys.modules:
        image_module = types.ModuleType("image")
        image_module.rgb_to_lab = lambda pixel: pixel
        image_module.lab_to_rgb = lambda pixel: pixel
        sys.modules["image"] = image_module

    if "machine" not in sys.modules:
        machine_module = types.ModuleType("machine")

        class FakeUART:
            def __init__(self, *_args, **_kwargs):
                self.buffer = []

            def any(self):
                return 0

            def read(self, _size):
                return None

            def write(self, data):
                self.buffer.append(data)
                return len(data)

        machine_module.UART = FakeUART
        sys.modules["machine"] = machine_module


def encode_frame(mode: int, topic: int, seq: int, body: bytes) -> bytes:
    """编码固定长度短帧."""

    if len(body) > FRAME_BODY_SIZE:
        raise ValueError("body too large")
    payload = bytes([int(mode), int(topic), int(seq)]) + body + (
        b"\x00" * (FRAME_BODY_SIZE - len(body))
    )
    return bytes([FRAME_HEAD]) + payload + bytes([crc8(payload)])


def decode_frame(frame: bytes):
    """解码固定长度短帧."""

    if not isinstance(frame, bytes) or len(frame) != FRAME_SIZE:
        return None
    if frame[0] != FRAME_HEAD:
        return None
    payload = frame[1:-1]
    if crc8(payload) != frame[-1]:
        return None
    return {
        "mode": payload[0],
        "topic": payload[1],
        "seq": payload[2],
        "body": payload[3:],
    }


def crc8(data: bytes) -> int:
    """计算协议帧 CRC8."""

    crc = 0
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _pack_i16(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True)


def _unpack_i16(body: bytes, offset: int) -> int:
    return int.from_bytes(body[offset : offset + 2], "little", signed=True)


def _pack_scaled(value: float) -> bytes:
    return _pack_i16(round(float(value) * _SCALE))


def _unpack_scaled(body: bytes, offset: int) -> float:
    return _unpack_i16(body, offset) / _SCALE


def encode_velocity_body(
    vx: float, vy: float, omega: float = 0.0, has_omega: bool = False
) -> bytes:
    """编码视觉速度 body."""

    return (
        _pack_scaled(vx)
        + _pack_scaled(vy)
        + _pack_scaled(omega)
        + bytes([1 if has_omega else 0])
    )


def decode_velocity_body(body: bytes):
    """解码视觉速度 body."""

    return {
        "vx": _unpack_scaled(body, 0),
        "vy": _unpack_scaled(body, 2),
        "omega": _unpack_scaled(body, 4),
        "has_omega": bool(body[6]),
    }


def encode_master_vision_hook_sync_body(
    context_id: int, state: int, target: int, arg: int
) -> bytes:
    """编码主车视觉同步 body."""

    return bytes([int(context_id), int(state), int(target)]) + _pack_i16(arg)


def decode_master_vision_hook_sync_body(body: bytes):
    """解码主车视觉同步 body."""

    return {
        "context_id": int(body[0]),
        "state": int(body[1]),
        "target": int(body[2]),
        "arg": _unpack_i16(body, 3),
    }


def encode_assistant_vision_task_sync_body(state: int, target: int, arg: int) -> bytes:
    """编码辅车视觉同步 body."""

    return bytes([int(state), int(target)]) + _pack_i16(arg)


def decode_assistant_vision_task_sync_body(body: bytes):
    """解码辅车视觉同步 body."""

    return {
        "state": int(body[0]),
        "target": int(body[1]),
        "arg": _unpack_i16(body, 2),
    }


def decode_master_vision_event_report_body(body: bytes):
    """解码主车视觉事件 body."""

    return {
        "context_id": int(body[0]),
        "event": int(body[1]),
        "value": _unpack_i16(body, 2),
    }


def decode_assistant_vision_event_report_body(body: bytes):
    """解码辅车视觉事件 body."""

    return {
        "event": int(body[0]),
        "value": _unpack_i16(body, 1),
    }


def assistant_sync_frame(seq: int, state: int, target: int, arg: int) -> bytes:
    """构造辅车视觉同步帧."""

    return encode_frame(
        MODE_TCP,
        TOPIC_ASSISTANT_VISION_TASK_SYNC,
        seq,
        encode_assistant_vision_task_sync_body(state, target, arg),
    )


def assistant_event_ack_frame(seq: int) -> bytes:
    """构造辅车视觉事件 ACK 帧."""

    return encode_frame(MODE_ACK, TOPIC_ASSISTANT_VISION_EVENT_REPORT, seq, b"")


def master_sync_frame(
    seq: int, context_id: int, state: int, target: int, arg: int
) -> bytes:
    """构造主车视觉同步帧."""

    return encode_frame(
        MODE_TCP,
        TOPIC_MASTER_VISION_HOOK_SYNC,
        seq,
        encode_master_vision_hook_sync_body(context_id, state, target, arg),
    )


def master_event_ack_frame(seq: int) -> bytes:
    """构造主车视觉事件 ACK 帧."""

    return encode_frame(MODE_ACK, TOPIC_MASTER_VISION_EVENT_REPORT, seq, b"")
