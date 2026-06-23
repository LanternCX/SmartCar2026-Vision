"""辅车 OpenART 视觉入口 v2."""

# pyright: reportAttributeAccessIssue=false

import gc
import image
import sensor
import tf
import time
from machine import UART

# 是否启用辅车调试画面显示
ASSISTANT_DEBUG_DISPLAY_ENABLED = False

# 板载串口编号, 用于和主控通信
UART_ID = 2
# 串口波特率, 单位为 bit/s
UART_BAUDRATE = 115200
# 图像曝光时间, 单位为 us
EXP_TIME_US = 500
# 可靠消息重发间隔, 单位为 ms
RELIABLE_RESEND_INTERVAL_MS = 100


class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    LOCAL_VISION_CONTROL = 0x04
    ASSISTANT_VISION_TASK_SYNC = 0x11
    ASSISTANT_VISION_EVENT_REPORT = 0x13


class LocalVisionControl:
    PAUSE = 1
    RESUME = 2
    RETURN_LINE_GATE_ON = 3
    RETURN_LINE_GATE_OFF = 4


class RunMode:
    FOLLOW = "follow"
    APPROACH_OBJECT = "approach_object"
    ORBIT_OBJECT = "orbit_object"
    RETURN_LINE = "return_line"


class State:
    APPROACH_OBJECT = 2
    ORBIT = 3
    TRANSPORT_OBJECT = 4
    RETURN_FOLLOW = 6


class Target:
    NONE = 0
    OBJECT = 1


class Task:
    SEARCH = 1
    TRANSPORT = 2
    ORBIT = 3
    RETURN_GARAGE_LINE = 5


class Event:
    TARGET_FOUND = 6
    ALIGNED = 7
    RETURN_LINE_ALIGNED = 10


# ROI 跟踪失败原因编号分组.
class TrackFailureReason:
    NONE = 0
    NO_CANDIDATE = 1
    OUT_OF_WINDOW = 2
    AREA_JUMP = 3
    EDGE_TOUCH = 4
    POOR_SEPARATION = 5


# 帧序号环形空间大小, 取满 1 字节范围
SEQ_RING_SIZE = 256
# 半环阈值, 用于比较环形序号前后关系
SEQ_HALF_RING = 128
# 协议消息体长度, 单位为 byte
FRAME_BODY_SIZE = 10
# 协议帧头标记
FRAME_HEAD = 0xA5
# 协议整帧长度, 单位为 byte
FRAME_SIZE = 15
# 是否启用 YOLO。False 时目标相关流程统一使用色块识别。
OBJECT_DETECTION_USE_YOLO = False
# YOLO 模型文件路径
YOLO_MODEL_PATH = "/sd/yolo.tflite"
# YOLO 输入图像复制缩放比例
YOLO_IMAGE_COPY_SCALE = 0.75
# YOLO 检测最小置信度阈值
YOLO_MIN_SCORE = 0.50
# 纯 YOLO 阶段两次检测之间跳过的帧数间隔
ASSISTANT_YOLO_ONLY_INTERVAL_FRAMES = 5
# YOLO 输出标签顺序, 需与模型保持一致
YOLO_LABELS = ("tennis", "red", "blue", "brown", "white")
# 视觉控制的参考帧率, 用于按时间尺度理解速度响应
VISION_REFERENCE_FPS = 25.0

# 色块候选合并边距, 单位为 px
OBJECT_BLOB_MERGE_MARGIN = 0
# 色块最小像素数阈值
OBJECT_BLOB_PIXELS_THRESHOLD = 200
# 色块最小面积阈值
OBJECT_BLOB_AREA_THRESHOLD = 200
# 跟随任务的颜色阈值配置
FOLLOW_TASKS = (("marker", (50, 100, 41, 127, -60, 127)),)
# 目标相关任务的筛选参数配置
OBJECT_TASKS = (
    ('red', ((14, 57, 24, 84, -4, 48),), 3, 30, 70, 220, True),
)
# 回库黄线识别阈值
RETURN_LINE_YELLOW_THRESHOLD = (58, 87, -24, -1, 21, 84)
# 回库 touch 固定物体区域左边界比例
RETURN_LINE_TOUCH_ROI_LEFT_RATIO = 0.25
# 回库 touch 固定物体区域右边界比例
RETURN_LINE_TOUCH_ROI_RIGHT_RATIO = 0.75
# 回库 touch 固定物体区域顶部比例
RETURN_LINE_TOUCH_ROI_TOP_RATIO = 2.0 / 3.0
# 回库 touch 黄色接触占比阈值
RETURN_LINE_TOUCH_RATIO_THRESHOLD = 0.1

# 跟随阶段的横向死区, 单位为 px
FOLLOW_X_DEADZONE_PX = 5.0
# 跟随阶段的目标纵向位置, 单位为 px
FOLLOW_TARGET_Y = 45.0
# 跟随阶段的纵向死区, 单位为 px
FOLLOW_Y_DEADZONE_PX = 8.0
# 跟随阶段横向控制比例系数
FOLLOW_CONTROL_KP_X = 0.04
# 跟随阶段纵向控制比例系数
FOLLOW_CONTROL_KP_Y = -0.10
# 跟随阶段最小输出速度
FOLLOW_CONTROL_MIN_SPEED = 0
# 跟随阶段纵向速度上限
FOLLOW_CONTROL_MAX_Y = 5

# 目标丢失时的默认搜索横向速度
OBJECT_MISSING_SEARCH_VX = 0.0
# 目标丢失时的默认搜索纵向速度
OBJECT_MISSING_SEARCH_VY = 0.0
# 接近目标阶段横向控制比例系数
OBJECT_APPROACH_KP_X = 0.05
# 接近目标阶段纵向控制比例系数
OBJECT_APPROACH_KP_Y = -0.2
# 接近目标阶段最小输出速度
OBJECT_APPROACH_MIN_SPEED = 2
# 接近目标阶段横向死区, 单位为 px
OBJECT_APPROACH_DEADZONE_X_PX = 15.0
# 接近目标阶段纵向死区, 单位为 px
OBJECT_APPROACH_DEADZONE_Y_PX = 8.0
# 接近目标阶段横向速度上限
OBJECT_APPROACH_MAX_VX = 5.0
# 接近目标阶段纵向速度上限
OBJECT_APPROACH_MAX_VY = 5.0
# 绕目标阶段横向速度修正比例系数
OBJECT_ORBIT_KP_X = 0.05
# 绕目标阶段纵向速度修正比例系数
OBJECT_ORBIT_KP_Y = -0.15
# 绕目标阶段最小输出速度
OBJECT_ORBIT_MIN_SPEED = 0
# 绕目标阶段横向死区, 单位为 px
OBJECT_ORBIT_DEADZONE_X_PX = 15.0
# 绕目标阶段纵向死区, 单位为 px
OBJECT_ORBIT_DEADZONE_Y_PX = 8.0
# 绕目标阶段横向速度上限
OBJECT_ORBIT_MAX_VX = 5.0
# 绕目标阶段纵向速度上限
OBJECT_ORBIT_MAX_VY = 5.0
# 接近目标阶段期望的图像横向位置, 单位为 px
OBJECT_APPROACH_TARGET_X_PX = 160.0
# 接近目标阶段期望的图像纵向位置, 单位为 px
OBJECT_APPROACH_TARGET_Y_PX = 200.0
# 绕目标阶段期望的图像横向位置, 单位为 px
OBJECT_ORBIT_TARGET_X_PX = 160.0
# 绕目标阶段期望的图像纵向位置, 单位为 px
OBJECT_ORBIT_TARGET_Y_PX = 200.0
# 辅车运输阶段期望的图像纵向位置, 单位为 px
ASSISTANT_TRANSPORT_TARGET_Y_PX = 240.0
# 目标最小有效面积阈值
OBJECT_MIN_AREA = 50.0
# 目标横向对齐容差, 单位为 px
OBJECT_X_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_X_PX
# 目标纵向对齐容差, 单位为 px
OBJECT_Y_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_Y_PX
# 判定目标稳定所需连续帧数
OBJECT_STABLE_FRAMES = 3
# 允许在两次 YOLO 之间连续使用 ROI 的最大帧数
ROI_TRACKING_MAX_FRAMES = 60
# ROI 连续失手达到该值后立即回退到 YOLO
ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 3
# 动态阈值健康检查连续失败达到该值后才触发重标定
DYNAMIC_THRESHOLD_REFRESH_FAILURE_FRAMES = 2
# 新阈值连续通过确认达到该值后才正式替换
DYNAMIC_THRESHOLD_REFRESH_CONFIRM_FRAMES = 2

# 协议约定的图像宽度, 单位为 px
PROTOCOL_IMAGE_WIDTH = 320
# 协议约定的图像高度, 单位为 px
PROTOCOL_IMAGE_HEIGHT = 240

# 有符号 16 位整数下界
_I16_MIN = -32768
# 有符号 16 位整数上界
_I16_MAX = 32767
# 浮点数打包为定点数时使用的缩放倍数
_SCALE = 1000


def _require_u8(value):
    value = int(value)
    if value < 0 or value > 0xFF:
        raise ValueError("u8 out of range")
    return value


def _saturate_i16(value):
    if value < _I16_MIN:
        return _I16_MIN
    if value > _I16_MAX:
        return _I16_MAX
    return value


def _pack_i16(value):
    value = _saturate_i16(int(value))
    if value < 0:
        value += 0x10000
    return bytes((value & 0xFF, (value >> 8) & 0xFF))


def _unpack_i16(body, offset):
    value = int(body[offset]) | (int(body[offset + 1]) << 8)
    if value >= 0x8000:
        value -= 0x10000
    return value


def _pack_i8(value):
    value = int(value)
    if value < -128 or value > 127:
        raise ValueError("i8 out of range")
    if value < 0:
        value += 0x100
    return value


def _unpack_i8(body, offset):
    value = int(body[offset])
    if value >= 0x80:
        value -= 0x100
    return value


def _pack_threshold(threshold):
    if threshold is None:
        threshold = (0, 0, 0, 0, 0, 0)
    if len(threshold) != 6:
        raise ValueError("threshold must have 6 values")
    return bytes(_pack_i8(value) for value in threshold)


def _unpack_threshold(body, offset):
    return tuple(_unpack_i8(body, offset + index) for index in range(6))


def _pack_scaled(value):
    return _pack_i16(round(float(value) * _SCALE))


def _unpack_scaled(body, offset):
    return _unpack_i16(body, offset) / float(_SCALE)


def encode_frame(mode, topic, seq, body):
    mode = _require_u8(mode)
    topic = _require_u8(topic)
    seq = _require_u8(seq)
    if not isinstance(body, bytes):
        body = bytes(body)
    if len(body) > FRAME_BODY_SIZE:
        raise ValueError("body too large")
    payload = bytes((mode, topic, seq)) + body + (b"\x00" * (FRAME_BODY_SIZE - len(body)))
    return bytes((FRAME_HEAD,)) + payload + bytes((_crc8(payload),))


def decode_frame(frame_bytes):
    if isinstance(frame_bytes, memoryview):
        frame_bytes = frame_bytes.tobytes()
    elif isinstance(frame_bytes, bytearray):
        frame_bytes = bytes(frame_bytes)
    if not isinstance(frame_bytes, bytes):
        return None
    if len(frame_bytes) != FRAME_SIZE:
        return None
    if frame_bytes[0] != FRAME_HEAD:
        return None
    payload = frame_bytes[1:-1]
    if _crc8(payload) != frame_bytes[-1]:
        return None
    return {
        "mode": payload[0],
        "topic": payload[1],
        "seq": payload[2],
        "body": payload[3:],
    }


def _crc8(data):
    crc = 0
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_velocity_body(vx, vy, omega=0.0, has_omega=False):
    return (
        _pack_scaled(vx)
        + _pack_scaled(vy)
        + _pack_scaled(omega)
        + bytes((1 if has_omega else 0,))
    )


def decode_velocity_body(body):
    return {
        "vx": _unpack_scaled(body, 0),
        "vy": _unpack_scaled(body, 2),
        "omega": _unpack_scaled(body, 4),
        "has_omega": bool(body[6]),
    }


def encode_local_vision_control_body(action):
    return bytes((_require_u8(action),))


def decode_local_vision_control_body(body):
    return {"action": int(body[0])}


def format_local_vision_control_frame(reliable_seq, action):
    return encode_frame(
        Mode.TCP,
        Topic.LOCAL_VISION_CONTROL,
        reliable_seq,
        encode_local_vision_control_body(action),
    )


def encode_assistant_vision_task_sync_body(state_value, target, arg, threshold=None):
    return bytes((_require_u8(state_value), _require_u8(target))) + _pack_i16(arg) + _pack_threshold(threshold)


def decode_assistant_vision_task_sync_body(body):
    return {
        "state": int(body[0]),
        "target": int(body[1]),
        "arg": _unpack_i16(body, 2),
        "threshold": _unpack_threshold(body, 4),
    }


def pack_task_arg(config_id, object_id):
    packed = (int(config_id) & 0xFF) | ((int(object_id) & 0xFF) << 8)
    if packed >= 0x8000:
        packed -= 0x10000
    return packed


def unpack_task_arg_config(arg):
    return int(arg) & 0xFF


def unpack_task_arg_object_id(arg):
    return (int(arg) >> 8) & 0xFF


def encode_assistant_vision_event_report_body(event, value):
    return bytes((_require_u8(event),)) + _pack_i16(value)


def decode_assistant_vision_event_report_body(body):
    return {
        "event": int(body[0]),
        "value": _unpack_i16(body, 1),
    }


def format_search_velocity_frame(vx, vy):
    return encode_frame(
        Mode.UDP,
        Topic.LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.ASSISTANT_VISION_TASK_SYNC, reliable_seq, b"")


def format_local_vision_control_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.LOCAL_VISION_CONTROL, reliable_seq, b"")


def format_event_frame(reliable_seq, event, value):
    return encode_frame(
        Mode.TCP,
        Topic.ASSISTANT_VISION_EVENT_REPORT,
        reliable_seq,
        encode_assistant_vision_event_report_body(event, value),
    )


def parse_task_sync_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.TCP or frame["topic"] != Topic.ASSISTANT_VISION_TASK_SYNC:
        return None
    packet = decode_assistant_vision_task_sync_body(frame["body"])
    result = {
        "reliable_seq": int(frame["seq"]),
        "state": int(packet["state"]),
        "target": int(packet["target"]),
        "arg": int(packet["arg"]),
    }
    threshold = tuple(packet["threshold"])
    if threshold_has_value(threshold):
        result["threshold"] = threshold  # pyright: ignore[reportArgumentType]
    return result


def parse_event_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.ASSISTANT_VISION_EVENT_REPORT:
        return None
    return {"reliable_seq": int(frame["seq"])}


def parse_local_vision_control_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.LOCAL_VISION_CONTROL:
        return None
    return {"reliable_seq": int(frame["seq"])}


def parse_local_vision_control_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.TCP or frame["topic"] != Topic.LOCAL_VISION_CONTROL:
        return None
    body = frame["body"]
    if len(body) < 1:
        return None
    packet = decode_local_vision_control_body(body)
    packet["reliable_seq"] = int(frame["seq"])
    return packet


def is_newer_seq(seq, last_seq):
    if last_seq is None:
        return True
    diff = (int(seq) - int(last_seq)) % SEQ_RING_SIZE
    return diff != 0 and diff < SEQ_HALF_RING


def default_now_ms():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


def reference_frame_interval_ms():
    return 1000.0 / VISION_REFERENCE_FPS


def should_resend(now_ms, last_sent_ms, interval_ms):
    if last_sent_ms is None:
        return True
    interval_ms = int(interval_ms)
    if interval_ms <= 0:
        return True
    if int(now_ms) < int(last_sent_ms):
        return True
    return int(now_ms) - int(last_sent_ms) >= interval_ms


def _apply_min_speed(value, min_speed, limit=None):
    value = float(value)
    if limit is not None:
        limit = abs(float(limit))
        if value > limit:
            value = limit
        if value < -limit:
            value = -limit
    min_speed = abs(float(min_speed))
    if value == 0.0:
        return 0.0
    if limit is not None and min_speed > limit:
        min_speed = limit
    if 0.0 < value < min_speed:
        return min_speed
    if -min_speed < value < 0.0:
        return -min_speed
    return value


def _clamp(value, limit):
    value = float(value)
    limit = abs(float(limit))
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def current_frame_time_scale():
    frame_interval_ms = float(state.current_frame_interval_ms)
    if frame_interval_ms <= 0.0:
        return 1.0
    reference_interval_ms = reference_frame_interval_ms()
    return reference_interval_ms / frame_interval_ms


def _build_axis_velocity(error, kp, limit=None):
    value = float(error) * float(kp) * current_frame_time_scale()
    if limit is None:
        return value
    return _clamp(value, float(limit) * current_frame_time_scale())


def _axis_p_velocity(error, deadzone, kp, limit, min_speed):
    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    if float(kp) == 0.0:
        return 0.0
    return _apply_min_speed(
        _build_axis_velocity(error, kp, limit),
        float(min_speed) * current_frame_time_scale(),
        float(limit) * current_frame_time_scale(),
    )


def build_follow_command(valid, err_x, err_y):
    if int(valid) != 1:
        return {
            "valid": 0,
            "err_x": 0.0,
            "err_y": 0.0,
            "command_vx": 0.0,
            "command_vy": 0.0,
        }
    command_vx = _axis_p_velocity(
        err_x,
        FOLLOW_X_DEADZONE_PX,
        FOLLOW_CONTROL_KP_X,
        FOLLOW_CONTROL_MAX_Y,
        FOLLOW_CONTROL_MIN_SPEED,
    )
    command_vy = _axis_p_velocity(
        err_y,
        FOLLOW_Y_DEADZONE_PX,
        FOLLOW_CONTROL_KP_Y,
        FOLLOW_CONTROL_MAX_Y,
        FOLLOW_CONTROL_MIN_SPEED,
    )
    return {
        "valid": 1,
        "err_x": float(err_x),
        "err_y": float(err_y),
        "command_vx": command_vx,
        "command_vy": command_vy,
    }


def blob_rect_to_bbox(rect):
    left, top, width, height = rect
    return float(left), float(top), float(left + width), float(top + height)


def normalize_bbox_for_protocol(left, top, right, bottom):
    img_height = float(state.current_image_height)
    normalized_top = img_height - float(bottom)
    normalized_bottom = img_height - float(top)
    return float(left), normalized_top, float(right), normalized_bottom


def compute_lateral_error(blob_cx, cx_screen):
    return float(blob_cx) - float(cx_screen)


def edge_length(p0, p1):
    dx = float(p0[0]) - float(p1[0])
    dy = float(p0[1]) - float(p1[1])
    return (dx * dx + dy * dy) ** 0.5


def get_marker_corners(blob):
    corners_fn = getattr(blob, "min_corners", None)
    if corners_fn is not None:
        return corners_fn()
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def compute_marker_span(corners):
    if len(corners) < 4:
        return 0.0
    return max(edge_length(corners[0], corners[1]), edge_length(corners[1], corners[2]))


def compute_vertical_error(marker_span, target_span):
    return float(marker_span) - float(target_span)


def blob_area(blob):
    area_fn = getattr(blob, "area", None)
    if area_fn is not None:
        return float(area_fn())
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(0.0, float(right) - float(left)) * max(0.0, float(bottom) - float(top))


class YoloDetectionBlob:
    def __init__(self, left, top, right, bottom, label, score):
        self._left = float(left)
        self._top = float(top)
        self._right = float(right)
        self._bottom = float(bottom)
        self.label = int(label)
        self.score = float(score)

    def rect(self):
        left = int(round(self._left))
        top = int(round(self._top))
        right = int(round(self._right))
        bottom = int(round(self._bottom))
        return left, top, right - left, bottom - top

    def cx(self):
        return (self._left + self._right) / 2.0

    def cy(self):
        return (self._top + self._bottom) / 2.0

    def area(self):
        return max(0.0, self._right - self._left) * max(0.0, self._bottom - self._top)

    def min_corners(self):
        return (
            (self._left, self._top),
            (self._right, self._top),
            (self._right, self._bottom),
            (self._left, self._bottom),
        )


class PredictedBlob:
    def __init__(self, left, top, right, bottom):
        self._left = float(left)
        self._top = float(top)
        self._right = float(right)
        self._bottom = float(bottom)

    def rect(self):
        left = int(round(self._left))
        top = int(round(self._top))
        right = int(round(self._right))
        bottom = int(round(self._bottom))
        return left, top, right - left, bottom - top

    def cx(self):
        return (self._left + self._right) / 2.0

    def cy(self):
        return (self._top + self._bottom) / 2.0

    def area(self):
        return max(0.0, self._right - self._left) * max(0.0, self._bottom - self._top)


def blob_max_side_length(blob):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(float(right - left), float(bottom - top))


def object_task_parts(task):
    if len(task) >= 7 and isinstance(task[1], (tuple, list)):
        return (
            task[0],
            int(task[2]),
            int(task[3]),
            int(task[4]),
            max(0, int(task[5])),
            bool(task[6]),
        )
    if len(task) >= 6:
        return (
            task[0],
            int(task[1]),
            int(task[2]),
            int(task[3]),
            max(0, int(task[4])),
            bool(task[5]),
        )
    if len(task) >= 5:
        return (
            task[0],
            int(task[1]),
            int(task[2]),
            int(task[3]),
            0,
            bool(task[4]),
        )
    return (
        task[0],
        OBJECT_BLOB_MERGE_MARGIN,
        OBJECT_BLOB_PIXELS_THRESHOLD,
        OBJECT_BLOB_AREA_THRESHOLD,
        0,
        True,
    )


def object_task_thresholds(task):
    if len(task) < 7 or not isinstance(task[1], (tuple, list)):
        return ()
    thresholds = task[1]
    if len(thresholds) == 6 and not isinstance(thresholds[0], (tuple, list)):
        return (tuple(thresholds),)
    return tuple(tuple(threshold) for threshold in thresholds)


def object_task_name_from_id(object_id):
    object_id = int(object_id)
    if object_id <= 0:
        return None
    index = object_id - 1
    if index >= len(OBJECT_TASKS):
        return None
    return OBJECT_TASKS[index][0]


def threshold_has_value(threshold):
    if threshold is None:
        return False
    for value in threshold:
        if int(value) != 0:
            return True
    return False


def object_task_id(task_name):
    for index, task in enumerate(OBJECT_TASKS, 1):
        if task[0] == task_name:
            return index
    return 0


def current_blob_task_name():
    if state.track_task_name is not None:
        return state.track_task_name
    task_name = object_task_name_from_id(state.current_object_id())
    if task_name is not None:
        return task_name
    if OBJECT_TASKS:
        return OBJECT_TASKS[0][0]
    return None


def _object_task_thresholds(task_name):
    for task in OBJECT_TASKS:
        if task[0] == task_name:
            return object_task_thresholds(task)
    return ()


def _object_task_config(task_name):
    for task in OBJECT_TASKS:
        if task[0] == task_name:
            return object_task_parts(task)
    return None


def load_yolo_model():
    if not OBJECT_DETECTION_USE_YOLO:
        return None
    if state.yolo_net is None:
        state.yolo_net = tf.load(YOLO_MODEL_PATH)
    return state.yolo_net


def _copy_image_for_yolo(img):
    return img.copy(YOLO_IMAGE_COPY_SCALE, 1)


def label_name(label):
    label = int(label)
    if 0 <= label < len(YOLO_LABELS):
        return YOLO_LABELS[label]
    return "unknown"


def yolo_detect(img):
    net = state.yolo_net
    if net is None:
        raise RuntimeError("yolo_net not loaded")
    detect_img = _copy_image_for_yolo(img)
    image_width = float(img.width())
    image_height = float(img.height())
    allowed_task_names = {task[0] for task in OBJECT_TASKS}
    candidates = []
    for detected in tf.detect(net, detect_img):  # pyright: ignore[reportCallIssue]
        x1, y1, x2, y2, label, score = detected
        if float(score) <= float(YOLO_MIN_SCORE):
            continue
        task_name = label_name(label)
        if task_name not in allowed_task_names:
            continue
        left = float(x1) * image_width
        top = float(y1) * image_height
        right = float(x2) * image_width
        bottom = float(y2) * image_height
        if right <= left or bottom <= top:
            continue
        blob = YoloDetectionBlob(left, top, right, bottom, label, score)
        if blob.area() < float(OBJECT_MIN_AREA):
            continue
        _, _, _, protocol_bottom = normalize_bbox_for_protocol(
            left,
            top,
            right,
            bottom,
        )
        candidates.append((task_name, blob.cx(), blob.cy(), protocol_bottom, blob.area(), blob))
    return candidates


def _find_blobs_with_task_config(img, thresholds, pixels_threshold, area_threshold, merge_margin, roi=None):
    try:
        return img.find_blobs(
            list(thresholds),
            pixels_threshold=pixels_threshold,
            area_threshold=1,
            merge=True,
            roi=roi,
            margin=max(0, int(merge_margin)),
        )
    except TypeError:
        try:
            return img.find_blobs(
                list(thresholds),
                pixels_threshold=pixels_threshold,
                area_threshold=1,
                merge=True,
                roi=roi,
            )
        except TypeError:
            return img.find_blobs(
                list(thresholds),
                pixels_threshold=pixels_threshold,
                area_threshold=1,
                merge=True,
            )


def build_blob_candidates(img):
    candidates = []
    img_height = img.height()
    for task_name, threshold in FOLLOW_TASKS:
        blobs = img.find_blobs(
            [threshold],
            pixels_threshold=OBJECT_BLOB_PIXELS_THRESHOLD,
            area_threshold=OBJECT_BLOB_AREA_THRESHOLD,
            merge=True,
        )
        for blob in blobs:
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, bottom = normalize_bbox_for_protocol(left, top, right, bottom)
            marker_span = compute_marker_span(blob.min_corners())
            candidates.append((task_name, blob.cx(), blob.cy(), bottom, marker_span, blob))
    return candidates
def _tracked_search_roi():
    window = _tracked_target_window()
    if window is None:
        return None
    roi_left = max(0, int(window[4]))
    roi_top = max(0, int(window[5]))
    roi_right = min(int(state.current_image_width), int(window[6]))
    roi_bottom = min(int(state.current_image_height), int(window[7]))
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return None
    return (roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top)


def _build_dynamic_blob_object_candidates(img, use_tracking_roi=True):
    task_name = state.track_task_name
    if task_name is None and not use_tracking_roi:
        task_name = current_blob_task_name()
    if task_name is None:
        return ()
    config = _object_task_config(task_name)
    if config is None:
        return ()
    (
        _task_name,
        merge_margin,
        pixels_threshold,
        area_threshold,
        max_side_length,
        _require_all_thresholds,
    ) = config
    thresholds = _object_task_thresholds(task_name)
    if not thresholds and state.track_dynamic_threshold is not None:
        thresholds = (state.track_dynamic_threshold,)
    if not thresholds:
        return ()
    blobs = _find_blobs_with_task_config(
        img,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge_margin,
        _tracked_search_roi() if use_tracking_roi else None,
    )
    candidates = []
    for blob in blobs:
        if blob_area(blob) < float(area_threshold):
            continue
        if int(max_side_length) > 0 and blob_max_side_length(blob) > float(max_side_length):
            continue
        left, top, right, bottom = blob_rect_to_bbox(blob.rect())
        _, _, _, bottom = normalize_bbox_for_protocol(left, top, right, bottom)
        candidates.append((task_name, blob.cx(), blob.cy(), bottom, blob_area(blob), blob))
    return tuple(candidates)


def build_object_candidates(img, yolo_candidates):
    state.current_image = img
    state.current_image_width = int(img.width())
    state.current_image_height = int(img.height())
    if not bool(OBJECT_DETECTION_USE_YOLO):
        candidates = _build_dynamic_blob_object_candidates(img, use_tracking_roi=False)
        if candidates:
            state.current_detection_source = "roi"
            state.track_failure_reason = TrackFailureReason.NONE
            return tuple(candidates)
        state.current_detection_source = "miss"
        state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
        return ()
    if current_sync_is_blob_only(state.current_sync):
        if state.track_task_name is None:
            state.current_detection_source = "miss"
            state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
            return ()
        candidates = _build_dynamic_blob_object_candidates(img, use_tracking_roi=False)
        if state.track_task_name is not None:
            candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
        if candidates:
            state.current_detection_source = "roi"
            state.track_failure_reason = TrackFailureReason.NONE
            best = min(candidates, key=_tracked_candidate_sort_key)
            return (_filter_candidate_by_track(best),)
        state.current_detection_source = "miss"
        state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
        return ()
    if should_use_blob_tracking():
        candidates = _build_tracked_blob_object_candidates(img)
        if candidates:
            state.record_roi_attempt(True)
            state.current_detection_source = "roi"
            return candidates
        state.record_roi_attempt(False)
        state.track_roi_failure_frames += 1
        if state.track_roi_failure_frames < int(ROI_TRACKING_FAILURE_TO_YOLO_FRAMES):
            state.current_detection_source = "predict"
            state.track_confidence = max(0, int(state.track_confidence) - 20)
            return _build_predicted_object_candidates()
        state.record_roi_fallback()
    if current_sync_is_yolo_only(state.current_sync):
        state.current_detection_source = "yolo"
        state.track_failure_reason = TrackFailureReason.NONE
        return tuple(yolo_candidates)
    if state.track_rect is None and state.track_dynamic_threshold is not None:
        candidates = _build_dynamic_blob_object_candidates(img)
        if candidates:
            state.current_detection_source = "sync_threshold"
            return candidates
    if state.current_sync is not None and not current_sync_allows_yolo():
        state.current_detection_source = "miss"
        state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
        return ()
    state.current_detection_source = "yolo"
    candidates = tuple(yolo_candidates)
    if state.track_rect is None:
        state.track_failure_reason = TrackFailureReason.NONE
        return candidates
    tracked_candidates = _prefer_tracked_yolo_candidates(candidates)
    if tracked_candidates is not None:
        state.track_failure_reason = TrackFailureReason.NONE
        return tracked_candidates
    if state.current_sync is not None:
        state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
        return ()
    if state.track_task_name is not None:
        same_task_candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
        if same_task_candidates:
            best = min(same_task_candidates, key=_tracked_candidate_sort_key)
            state.track_failure_reason = TrackFailureReason.OUT_OF_WINDOW
            return (_filter_candidate_by_track(best),)
    state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
    return candidates


def _pixel_to_lab(pixel):
    if pixel is None:
        return None
    lab = image.rgb_to_lab(pixel)
    try:
        if len(lab) < 3:
            return None
    except TypeError:
        return None
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def _pixel_matches_threshold(pixel, threshold):
    lab = _pixel_to_lab(pixel)
    if lab is None:
        return False
    return (
        float(threshold[0]) <= float(lab[0]) <= float(threshold[1])
        and float(threshold[2]) <= float(lab[1]) <= float(threshold[3])
        and float(threshold[4]) <= float(lab[2]) <= float(threshold[5])
    )


def _clamp_range_value(value, lower, upper):
    value = int(round(float(value)))
    if value < int(lower):
        return int(lower)
    if value > int(upper):
        return int(upper)
    return value


def _sample_grid_count(span):
    return min(9, max(3, int((int(span) + 3) // 4)))


def _sample_grid_axis(start, end):
    return tuple(_sample_roi_positions(start, end, _sample_grid_count(int(end) - int(start))))


def _median_channel(samples, channel_index):
    values = sorted(sample[channel_index] for sample in samples)
    return values[len(values) // 2]


def _quantile_value(sorted_values, numerator, denominator):
    if not sorted_values:
        return None
    index = ((len(sorted_values) - 1) * int(numerator)) // int(denominator)
    return sorted_values[index]


def _lab_distance_value(lab, center_lab):
    delta_l = float(lab[0]) - float(center_lab[0])
    delta_a = float(lab[1]) - float(center_lab[1])
    delta_b = float(lab[2]) - float(center_lab[2])
    return int(delta_l * delta_l + 4.0 * delta_a * delta_a + 4.0 * delta_b * delta_b)


def _otsu_threshold(values):
    if not values:
        return None
    min_value = int(min(values))
    max_value = int(max(values))
    if max_value <= min_value:
        return None
    histogram = [0] * (max_value - min_value + 1)
    total_sum = 0
    for value in values:
        index = int(value) - min_value
        histogram[index] += 1
        total_sum += int(value)
    total_count = len(values)
    foreground_sum = 0
    foreground_count = 0
    best_threshold = None
    best_score = -1.0
    for index in range(len(histogram) - 1):
        count = histogram[index]
        foreground_count += count
        if foreground_count <= 0:
            continue
        background_count = total_count - foreground_count
        if background_count <= 0:
            break
        value = min_value + index
        foreground_sum += value * count
        foreground_mean = float(foreground_sum) / float(foreground_count)
        background_mean = float(total_sum - foreground_sum) / float(background_count)
        score = (
            float(foreground_count)
            * float(background_count)
            * (foreground_mean - background_mean)
            * (foreground_mean - background_mean)
        )
        if score > best_score:
            best_score = score
            best_threshold = value
    if best_score <= 0.0:
        return None
    return best_threshold


def _center_connected_sample_indices(mask_rows, center_mask_rows):
    height = len(mask_rows)
    if height <= 0:
        return ()
    width = len(mask_rows[0])
    queue = []
    visited = {}
    for row_index in range(height):
        for col_index in range(width):
            if not mask_rows[row_index][col_index]:
                continue
            if not center_mask_rows[row_index][col_index]:
                continue
            key = row_index * width + col_index
            visited[key] = True
            queue.append((row_index, col_index))
    if not queue:
        return ()
    connected = []
    while queue:
        row_index, col_index = queue.pop()
        connected.append((row_index, col_index))
        for next_row, next_col in (
            (row_index - 1, col_index),
            (row_index + 1, col_index),
            (row_index, col_index - 1),
            (row_index, col_index + 1),
        ):
            if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                continue
            if not mask_rows[next_row][next_col]:
                continue
            key = next_row * width + next_col
            if key in visited:
                continue
            visited[key] = True
            queue.append((next_row, next_col))
    return tuple(connected)


def _build_dynamic_threshold_from_labs(samples):
    threshold = []
    for channel_index, lower_bound, upper_bound in (
        (0, 0, 100),
        (1, -128, 127),
        (2, -128, 127),
    ):
        channel_values = sorted(sample[channel_index] for sample in samples)
        low = _quantile_value(channel_values, 1, 8)
        high = _quantile_value(channel_values, 7, 8)
        if low is None or high is None:
            return None
        margin = max(1, int((float(high) - float(low) + 3.0) // 4.0))
        threshold.append(_clamp_range_value(float(low) - float(margin), lower_bound, upper_bound))
        threshold.append(_clamp_range_value(float(high) + float(margin), lower_bound, upper_bound))
    return tuple(threshold)


def _build_dynamic_threshold_for_blob(img, blob):
    calibrated_thresholds = _object_task_thresholds(state.track_task_name)
    if calibrated_thresholds:
        return calibrated_thresholds[0]
    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None:
        return None
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(int(state.current_image_width), int(right))
    bottom = min(int(state.current_image_height), int(bottom))
    if right - left <= 4 or bottom - top <= 4:
        return None
    if (right - left) * (bottom - top) < int(OBJECT_MIN_AREA):
        return None
    sample_xs = _sample_grid_axis(left, right)
    sample_ys = _sample_grid_axis(top, bottom)
    if len(sample_xs) * len(sample_ys) < 9:
        return None
    inset_x = max(1, int((right - left) // 4))
    inset_y = max(1, int((bottom - top) // 4))
    center_left = left + inset_x
    center_top = top + inset_y
    center_right = right - inset_x
    center_bottom = bottom - inset_y
    if center_right <= center_left or center_bottom <= center_top:
        return None
    lab_rows = []
    center_samples = []
    total_valid = 0
    center_valid = 0
    for sample_y in sample_ys:
        row = []
        for sample_x in sample_xs:
            lab = _pixel_to_lab(get_pixel(int(sample_x), int(sample_y)))
            row.append(lab)
            if lab is None:
                continue
            total_valid += 1
            if (
                int(center_left) <= int(sample_x) < int(center_right)
                and int(center_top) <= int(sample_y) < int(center_bottom)
            ):
                center_samples.append(lab)
                center_valid += 1
        lab_rows.append(row)
    if total_valid < 9 or center_valid < 3:
        return None
    center_lab = (
        _median_channel(center_samples, 0),
        _median_channel(center_samples, 1),
        _median_channel(center_samples, 2),
    )
    distance_rows = []
    distances = []
    center_mask_rows = []
    for row_index in range(len(sample_ys)):
        distance_row = []
        center_mask_row = []
        sample_y = sample_ys[row_index]
        for col_index in range(len(sample_xs)):
            sample_x = sample_xs[col_index]
            lab = lab_rows[row_index][col_index]
            if lab is None:
                distance_row.append(None)
                center_mask_row.append(False)
                continue
            distance = _lab_distance_value(lab, center_lab)
            distance_row.append(distance)
            distances.append(distance)
            center_mask_row.append(
                int(center_left) <= int(sample_x) < int(center_right)
                and int(center_top) <= int(sample_y) < int(center_bottom)
            )
        distance_rows.append(distance_row)
        center_mask_rows.append(center_mask_row)
    threshold_value = _otsu_threshold(distances)
    if threshold_value is None:
        return None
    max_distance = int(max(distances))
    if threshold_value >= max_distance:
        return None
    mask_rows = []
    for distance_row in distance_rows:
        mask_rows.append(
            [
                distance is not None and int(distance) <= int(threshold_value)
                for distance in distance_row
            ]
        )
    connected_indices = _center_connected_sample_indices(mask_rows, center_mask_rows)
    if not connected_indices:
        return None
    connected_samples = []
    center_connected = 0
    for row_index, col_index in connected_indices:
        lab = lab_rows[row_index][col_index]
        if lab is None:
            continue
        connected_samples.append(lab)
        if center_mask_rows[row_index][col_index]:
            center_connected += 1
    connected_count = len(connected_samples)
    if connected_count < max(3, total_valid // 10):
        return None
    if connected_count * 20 <= total_valid:
        return None
    if connected_count * 20 >= total_valid * 19:
        return None
    if center_connected * total_valid < connected_count * center_valid:
        return None
    return _build_dynamic_threshold_from_labs(connected_samples)


def _estimate_dynamic_foreground_area(img, blob, threshold):
    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None or threshold is None:
        return None
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    width = max(1.0, float(right) - float(left))
    height = max(1.0, float(bottom) - float(top))
    sample_xs = _sample_grid_axis(int(left), int(right))
    sample_ys = _sample_grid_axis(int(top), int(bottom))
    sample_total = len(sample_xs) * len(sample_ys)
    if sample_total <= 0:
        return None
    foreground_count = 0
    for sample_y in sample_ys:
        for sample_x in sample_xs:
            if _pixel_matches_threshold(get_pixel(int(sample_x), int(sample_y)), threshold):
                foreground_count += 1
    if foreground_count <= 0:
        return None
    estimated_area = width * height * float(foreground_count) / float(sample_total)
    return max(1.0, float(estimated_area))


def _blob_bbox_area_from_rect(rect):
    left, top, right, bottom = rect
    return max(0.0, float(right) - float(left)) * max(0.0, float(bottom) - float(top))


def _dynamic_threshold_center_is_healthy(img, blob, threshold):
    get_pixel = getattr(img, "get_pixel", None)
    if threshold is None or get_pixel is None:
        return False
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    sample_xs = _sample_grid_axis(int(left), int(right))
    sample_ys = _sample_grid_axis(int(top), int(bottom))
    inset_x = max(1, int((int(right) - int(left)) // 4))
    inset_y = max(1, int((int(bottom) - int(top)) // 4))
    center_left = int(left) + inset_x
    center_top = int(top) + inset_y
    center_right = int(right) - inset_x
    center_bottom = int(bottom) - inset_y
    if center_right <= center_left or center_bottom <= center_top:
        return False
    center_hits = 0
    for sample_y in sample_ys:
        for sample_x in sample_xs:
            if not (
                center_left <= int(sample_x) < center_right
                and center_top <= int(sample_y) < center_bottom
            ):
                continue
            center_hits += 1
            if not _pixel_matches_threshold(get_pixel(int(sample_x), int(sample_y)), threshold):
                return False
    return center_hits >= 3


def _sample_roi_positions(start, end, sample_count):
    start = int(start)
    end = int(end)
    count = max(1, int(sample_count))
    span = max(1, end - start)
    for index in range(count):
        yield start + (span * (index * 2 + 1)) // (count * 2)
def _build_return_line_touch_roi(img):
    image_width = int(img.width())
    image_height = int(img.height())
    display_left = int(float(image_width) * float(RETURN_LINE_TOUCH_ROI_LEFT_RATIO))
    display_right = int(float(image_width) * float(RETURN_LINE_TOUCH_ROI_RIGHT_RATIO))
    display_top = int(float(image_height) * float(RETURN_LINE_TOUCH_ROI_TOP_RATIO))
    display_left = max(0, min(int(image_width), int(display_left)))
    display_right = max(int(display_left), min(int(image_width), int(display_right)))
    display_top = max(0, min(int(image_height), int(display_top)))
    left = int(image_width) - int(display_right)
    right = int(image_width) - int(display_left)
    top = 0
    bottom = int(image_height) - int(display_top)
    return (int(left), int(top), int(right) - int(left), int(bottom) - int(top))


def _count_yellow_pixels_in_roi(img, roi):
    roi_area = int(roi[2]) * int(roi[3])
    if roi_area <= 0:
        return 0
    blobs = img.find_blobs(
        [RETURN_LINE_YELLOW_THRESHOLD],
        roi=roi,
        pixels_threshold=1,
        area_threshold=1,
        merge=True,
    )
    if not blobs:
        return 0
    yellow_pixels = 0.0
    for blob in blobs:
        yellow_pixels += blob_area(blob)
    if yellow_pixels >= float(roi_area):
        return roi_area
    return int(yellow_pixels)


def _build_return_line_touch_ratio_percent(img):
    roi = _build_return_line_touch_roi(img)
    roi_area = int(roi[2]) * int(roi[3])
    if roi_area <= 0:
        return 0.0
    yellow_pixels = _count_yellow_pixels_in_roi(img, roi)
    return float(yellow_pixels) * 100.0 / float(roi_area)


def choose_best_candidate(candidates, target_x, target_y):
    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[3]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    return max(candidates, key=lambda item: float(item[4]))


def filter_candidates_in_target_window(candidates, target_x, target_y, tolerance_x, tolerance_y):
    _ = (target_x, tolerance_x)
    return [
        candidate
        for candidate in candidates
        if abs(float(candidate[3]) - float(target_y)) <= float(tolerance_y)
    ]


def should_filter_candidates_by_target_window(vision_state):
    if vision_state.current_sync is None:
        return False
    return (
        int(vision_state.current_sync["state"]) == int(State.TRANSPORT_OBJECT)
        and int(vision_state.current_object_config_id()) == int(Task.TRANSPORT)
    )


def build_object_observation_and_candidates():
    image_width = PROTOCOL_IMAGE_WIDTH
    image_height = PROTOCOL_IMAGE_HEIGHT
    current_object_candidates = state.current_object_candidates
    if not current_object_candidates:
        return build_object_observation(0, 0, 0, 0), None, None, current_object_candidates
    candidates = list(current_object_candidates)
    object_id = state.current_object_id()
    if object_id > 0:
        selected_task_name = object_task_name_from_id(object_id)
        if selected_task_name is not None:
            candidates = [candidate for candidate in candidates if candidate[0] == selected_task_name]
    if not candidates:
        return build_object_observation(0, 0, 0, 0), None, None, candidates
    config_id = state.current_object_config_id()
    target_x, target_y = build_object_target_point(config_id)
    use_target_window_filter = should_filter_candidates_by_target_window(state)
    if use_target_window_filter:
        candidates = filter_candidates_in_target_window(
            candidates,
            target_x,
            target_y,
            OBJECT_X_TOLERANCE_PX,
            OBJECT_Y_TOLERANCE_PX,
        )
    if not candidates:
        return build_object_observation(0, 0, 0, 0), None, None, candidates
    if use_target_window_filter:
        task_name, center_x, _center_y, bottom_y, area, best_blob = choose_largest_area_candidate(candidates)
    else:
        task_name, center_x, _center_y, bottom_y, area, best_blob = choose_best_candidate(
            candidates,
            target_x,
            target_y,
        )
    return (
        build_object_observation(1, center_x, bottom_y, area),
        best_blob,
        task_name,
        candidates,
    )


def _draw_debug_line(img, x0, y0, x1, y1):
    draw_line = getattr(img, "draw_line", None)
    if draw_line is None:
        return
    try:
        draw_line(int(x0), int(y0), int(x1), int(y1), color=(255, 255, 0))
    except TypeError:
        draw_line(int(x0), int(y0), int(x1), int(y1))


def _draw_debug_cross(img, x, y):
    draw_cross = getattr(img, "draw_cross", None)
    if draw_cross is None:
        return
    try:
        draw_cross(int(x), int(y), color=(255, 0, 0))
    except TypeError:
        draw_cross(int(x), int(y))


def _draw_debug_protocol_point(img, x, y):
    image_width = state.current_image_width
    image_height = state.current_image_height
    draw_x = int(image_width) - 1 - int(x)
    draw_y = int(image_height) - 1 - int(y)
    _draw_debug_cross(img, draw_x, draw_y)


def _draw_debug_text(img, x, y, text):
    draw_string = getattr(img, "draw_string", None)
    if draw_string is None:
        return
    try:
        draw_string(int(x), int(y), str(text), color=(255, 255, 255))
    except TypeError:
        draw_string(int(x), int(y), str(text))


def _draw_debug_rect(img, left, top, right, bottom, color=(0, 255, 255), thickness=1):
    draw_rectangle = getattr(img, "draw_rectangle", None)
    if draw_rectangle is None:
        return
    width = int(right) - int(left)
    height = int(bottom) - int(top)
    if width <= 0 or height <= 0:
        return
    try:
        draw_rectangle((int(left), int(top), width, height), color=color, thickness=thickness)
    except TypeError:
        draw_rectangle((int(left), int(top), width, height))


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    width = int(right) - int(left)
    height = int(bottom) - int(top)
    draw_rectangle = getattr(img, "draw_rectangle", None)
    if draw_rectangle is None:
        return
    try:
        draw_rectangle(int(left), int(top), int(width), int(height))
    except TypeError:
        draw_rectangle((int(left), int(top), int(width), int(height)))
    _draw_debug_text(img, int(pixel_x) + 4, int(pixel_y) - 10, "SELECT")


def tracking_source_debug_name(source):
    if source == "yolo":
        return "YOLO"
    if source == "roi":
        return "ROI"
    if source == "predict":
        return "PRED"
    return "MISS"


def tracking_failure_debug_name(reason):
    if int(reason) == int(TrackFailureReason.NO_CANDIDATE):
        return "NO_CAND"
    if int(reason) == int(TrackFailureReason.OUT_OF_WINDOW):
        return "OUT_WIN"
    if int(reason) == int(TrackFailureReason.AREA_JUMP):
        return "AREA"
    if int(reason) == int(TrackFailureReason.EDGE_TOUCH):
        return "EDGE"
    if int(reason) == int(TrackFailureReason.POOR_SEPARATION):
        return "SEP"
    return "NONE"


def debug_log(tag, text):
    if not ASSISTANT_DEBUG_DISPLAY_ENABLED:
        return
    print("[assistant_v2][%s] %s" % (str(tag), str(text)))


def draw_object_tracking_debug(img):
    predicted_roi = state.track_predicted_roi
    if predicted_roi is not None:
        _draw_debug_rect(img, predicted_roi[0], predicted_roi[1], predicted_roi[2], predicted_roi[3])
    if state.track_predicted_center_x is not None and state.track_predicted_bottom_y is not None:
        _draw_debug_protocol_point(img, state.track_predicted_center_x, state.track_predicted_bottom_y)
    _draw_debug_text(
        img,
        2,
        2,
        "src=%s conf=%d gap=%d" % (
            tracking_source_debug_name(state.current_detection_source),
            int(state.track_confidence),
            int(state.track_frames_since_yolo),
        ),
    )
    _draw_debug_text(
        img,
        2,
        14,
        "fail=%s rf=%d id=%d" % (
            tracking_failure_debug_name(state.track_failure_reason),
            int(state.track_roi_failure_frames),
            int(state.track_object_id),
        ),
    )
    _draw_debug_text(
        img,
        2,
        26,
        "fps t=%d y=%d r=%d p=%d fb=%d" % (
            int(state.last_second_total_frames),
            int(state.last_second_yolo_frames),
            int(state.last_second_roi_frames),
            int(state.last_second_predict_frames),
            int(state.last_second_roi_fallbacks),
        ),
    )
    _draw_debug_text(
        img,
        2,
        38,
        "roi ok=%d/%d" % (
            int(state.last_second_roi_success_frames),
            int(state.last_second_roi_attempt_frames),
        ),
    )


def build_object_observation(valid, center_x, bottom_y, area):
    if int(valid) != 1:
        return 0.0, 0.0, 0.0
    target_x, target_y = build_object_target_point(current_task_config_id())
    return (
        float(center_x) - target_x,
        float(bottom_y) - target_y,
        float(area),
    )


def build_object_target_point(config_id=Task.SEARCH):
    target_x = float(OBJECT_APPROACH_TARGET_X_PX)
    if int(config_id) == int(Task.ORBIT):
        return float(OBJECT_ORBIT_TARGET_X_PX), float(OBJECT_ORBIT_TARGET_Y_PX)
    if int(config_id) == int(Task.TRANSPORT):
        return target_x, float(ASSISTANT_TRANSPORT_TARGET_Y_PX)
    return target_x, float(OBJECT_APPROACH_TARGET_Y_PX)


def build_search_target_point(config_id):
    return build_object_target_point(config_id)


def current_task_config_id():
    return state.current_object_config_id()


def build_observation(valid, center_x, bottom_y, area):
    return build_object_observation(valid, center_x, bottom_y, area)


def build_observation_and_candidates():
    return build_object_observation_and_candidates()


def _build_object_y_velocity(err_y):
    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_APPROACH_DEADZONE_Y_PX):
        return 0.0
    image_height = float(state.current_image_height)
    scaled_error = err_y * (
        float(OBJECT_APPROACH_MAX_VY)
        / abs(float(OBJECT_APPROACH_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_APPROACH_KP_Y) * current_frame_time_scale(),
        float(OBJECT_APPROACH_MIN_SPEED) * current_frame_time_scale(),
        float(OBJECT_APPROACH_MAX_VY) * current_frame_time_scale(),
    )


def build_object_approach_velocity_from_error(err_x, err_y):
    return (
        _axis_p_velocity(
            err_x,
            OBJECT_APPROACH_DEADZONE_X_PX,
            OBJECT_APPROACH_KP_X,
            OBJECT_APPROACH_MAX_VX,
            OBJECT_APPROACH_MIN_SPEED,
        ),
        _build_object_y_velocity(err_y),
    )


def build_object_approach_velocity_from_observation(observation):
    x, y, value = observation
    if float(value) <= 0.0:
        return float(OBJECT_MISSING_SEARCH_VX), float(OBJECT_MISSING_SEARCH_VY)
    return build_object_approach_velocity_from_error(x, y)


def build_search_velocity_from_observation(observation):
    return build_object_approach_velocity_from_observation(observation)


def _build_object_orbit_y_velocity(err_y):
    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_ORBIT_DEADZONE_Y_PX):
        return 0.0
    if float(OBJECT_ORBIT_KP_Y) == 0.0:
        return 0.0
    image_height = float(state.current_image_height)
    scaled_error = err_y * (
        float(OBJECT_ORBIT_MAX_VY)
        / abs(float(OBJECT_ORBIT_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_ORBIT_KP_Y) * current_frame_time_scale(),
        float(OBJECT_ORBIT_MIN_SPEED) * current_frame_time_scale(),
        float(OBJECT_ORBIT_MAX_VY) * current_frame_time_scale(),
    )


def build_object_orbit_velocity_from_observation(observation):
    x, y, value = observation
    if float(value) <= 0.0:
        return 0.0, 0.0
    return (
        _axis_p_velocity(
            x,
            OBJECT_ORBIT_DEADZONE_X_PX,
            OBJECT_ORBIT_KP_X,
            OBJECT_ORBIT_MAX_VX,
            OBJECT_ORBIT_MIN_SPEED,
        ),
        _build_object_orbit_y_velocity(y),
    )


def build_orbit_correction_velocity_from_observation(observation):
    return build_object_orbit_velocity_from_observation(observation)


# 当前辅车视觉运行态统一集中在单一状态对象里.
class RuntimeState:
    def __init__(
        self,
        min_area=OBJECT_MIN_AREA,
        tolerance_x=OBJECT_X_TOLERANCE_PX,
        tolerance_y=OBJECT_Y_TOLERANCE_PX,
        stable_frames=OBJECT_STABLE_FRAMES,
        now_ms=None,
        event_resend_interval_ms=RELIABLE_RESEND_INTERVAL_MS,
    ):
        self._now_ms = now_ms or default_now_ms
        self._event_resend_interval_ms = int(event_resend_interval_ms)
        self._default_min_area = float(min_area)
        self._default_tolerance_x = float(tolerance_x)
        self._default_tolerance_y = float(tolerance_y)
        self._default_stable_frames = int(stable_frames)
        self.reset()

    def reset(self):
        self.min_area = float(self._default_min_area)
        self.tolerance_x = float(self._default_tolerance_x)
        self.tolerance_y = float(self._default_tolerance_y)
        self.required_stable_frames = int(self._default_stable_frames)
        self.mode = RunMode.FOLLOW
        self.current_sync = None
        self._last_sync_seq = None
        self._stable_count = 0
        self._pending_event = None
        self._pending_event_last_sent_ms = None
        self.pending_local_vision_control = None
        self.pending_local_vision_control_last_sent_ms = None
        self.next_local_vision_control_seq = 1
        self.local_vision_control_paused = False
        self.return_line_gate_enabled = False
        self._completed_event_sync_seq = None
        self.entry_yolo_pending = False
        self.yolo_net = None
        self.uart_device = None
        self.rx_buffer = b""
        self.current_yolo_candidates = ()
        self.current_object_candidates = ()
        self.current_image = None
        self.current_image_width = PROTOCOL_IMAGE_WIDTH
        self.current_image_height = PROTOCOL_IMAGE_HEIGHT
        self.current_frame_interval_ms = reference_frame_interval_ms()
        self.current_detection_source = "miss"
        self.yolo_only_skip_frames_remaining = 0
        self.yolo_only_skip_active_for_frame = False
        self.clear_track()

    def clear_track(self):
        self.track_task_name = None
        self.track_object_id = 0
        self.track_center_x = None
        self.track_center_y = None
        self.track_bottom_y = None
        self.track_area = None
        self.track_rect = None
        self.track_velocity_x = 0.0
        self.track_velocity_bottom_y = 0.0
        self.track_source = None
        self.track_roi_success_frames = 0
        self.track_roi_failure_frames = 0
        self.track_frames_since_yolo = 0
        self.track_confidence = 0
        self.track_failure_reason = TrackFailureReason.NONE
        self.track_predicted_center_x = None
        self.track_predicted_bottom_y = None
        self.track_predicted_roi = None
        self.track_dynamic_threshold = None
        self.track_dynamic_threshold_rect = None
        self.track_dynamic_threshold_generation = 0
        self.track_dynamic_threshold_failure = None
        self.track_dynamic_threshold_health_failures = 0
        self.track_pending_dynamic_threshold = None
        self.track_pending_dynamic_threshold_ok_frames = 0
        self.current_second_total_frames = 0
        self.current_second_yolo_frames = 0
        self.current_second_roi_frames = 0
        self.current_second_predict_frames = 0
        self.current_second_miss_frames = 0
        self.current_second_roi_fallbacks = 0
        self.current_second_roi_attempt_frames = 0
        self.current_second_roi_success_frames = 0
        self.last_second_total_frames = 0
        self.last_second_yolo_frames = 0
        self.last_second_roi_frames = 0
        self.last_second_predict_frames = 0
        self.last_second_miss_frames = 0
        self.last_second_roi_fallbacks = 0
        self.last_second_roi_attempt_frames = 0
        self.last_second_roi_success_frames = 0
        self.frame_stats_window_start_ms = None

    def _roll_frame_stats(self, now_ms):
        if self.frame_stats_window_start_ms is None:
            self.frame_stats_window_start_ms = int(now_ms)
            return
        if int(now_ms) - int(self.frame_stats_window_start_ms) < 1000:
            return
        self.last_second_total_frames = int(self.current_second_total_frames)
        self.last_second_yolo_frames = int(self.current_second_yolo_frames)
        self.last_second_roi_frames = int(self.current_second_roi_frames)
        self.last_second_predict_frames = int(self.current_second_predict_frames)
        self.last_second_miss_frames = int(self.current_second_miss_frames)
        self.last_second_roi_fallbacks = int(self.current_second_roi_fallbacks)
        self.last_second_roi_attempt_frames = int(self.current_second_roi_attempt_frames)
        self.last_second_roi_success_frames = int(self.current_second_roi_success_frames)
        self.current_second_total_frames = 0
        self.current_second_yolo_frames = 0
        self.current_second_roi_frames = 0
        self.current_second_predict_frames = 0
        self.current_second_miss_frames = 0
        self.current_second_roi_fallbacks = 0
        self.current_second_roi_attempt_frames = 0
        self.current_second_roi_success_frames = 0
        self.frame_stats_window_start_ms = int(now_ms)

    def record_frame_source(self, source, now_ms=None):
        if now_ms is None:
            now_ms = default_now_ms()
        self._roll_frame_stats(now_ms)
        self.current_second_total_frames += 1
        if source == "yolo":
            self.current_second_yolo_frames += 1
        elif source == "roi":
            self.current_second_roi_frames += 1
        elif source == "predict":
            self.current_second_predict_frames += 1
        else:
            self.current_second_miss_frames += 1

    def record_roi_fallback(self, now_ms=None):
        if now_ms is None:
            now_ms = default_now_ms()
        self._roll_frame_stats(now_ms)
        self.current_second_roi_fallbacks += 1

    def record_roi_attempt(self, success, now_ms=None):
        if now_ms is None:
            now_ms = default_now_ms()
        self._roll_frame_stats(now_ms)
        self.current_second_roi_attempt_frames += 1
        if success:
            self.current_second_roi_success_frames += 1

    def handle_control_line(self, line):
        sync_packet = parse_task_sync_packet(line)
        if sync_packet is not None:
            return self._handle_sync_packet(sync_packet)
        control_packet = parse_local_vision_control_packet(line)
        if control_packet is not None:
            action = int(control_packet["action"])
            if action == int(LocalVisionControl.RETURN_LINE_GATE_ON):
                self.return_line_gate_enabled = True
            elif action == int(LocalVisionControl.RETURN_LINE_GATE_OFF):
                self.return_line_gate_enabled = False
            return format_local_vision_control_ack_frame(control_packet["reliable_seq"])
        ack_packet = parse_event_ack_packet(line)
        if ack_packet is not None:
            self._handle_ack_packet(ack_packet)
        return None

    def _handle_sync_packet(self, packet):
        reliable_seq = int(packet["reliable_seq"])
        if self._should_apply_sync(reliable_seq):
            sync_state = int(packet["state"])
            config_id = unpack_task_arg_config(packet["arg"])
            preserve_track = (
                self.track_task_name is not None
                and int(packet["target"]) == int(Target.OBJECT)
                and (
                    (
                        sync_state == int(State.TRANSPORT_OBJECT)
                        and config_id == int(Task.TRANSPORT)
                    )
                    or (
                        sync_state == int(State.APPROACH_OBJECT)
                        and config_id == int(Task.TRANSPORT)
                    )
                    or (
                        sync_state == int(State.ORBIT)
                        and config_id in (int(Task.ORBIT), int(Task.TRANSPORT))
                    )
                )
            )
            clear_local_vision_control_state()
            self.current_sync = {
                "reliable_seq": reliable_seq,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
                "threshold": tuple(packet.get("threshold", (0, 0, 0, 0, 0, 0))),
            }
            self._last_sync_seq = reliable_seq
            self.mode = self._mode_from_sync(self.current_sync)
            self._stable_count = 0
            self.return_line_gate_enabled = False
            clear_yolo_only_skip()
            if not preserve_track:
                self.clear_track()
            self.entry_yolo_pending = current_sync_is_entry_yolo_only(self.current_sync)
        return format_ack_frame(reliable_seq)

    def _should_apply_sync(self, reliable_seq):
        if self._last_sync_seq is None:
            return True
        if int(reliable_seq) == int(self._last_sync_seq):
            return False
        return is_newer_seq(reliable_seq, self._last_sync_seq)

    def _mode_from_sync(self, sync):
        config_id = unpack_task_arg_config(sync["arg"])
        if (
            int(sync["state"]) == State.APPROACH_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and config_id in (Task.SEARCH, Task.TRANSPORT)
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.TRANSPORT_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.ORBIT
        ):
            return RunMode.ORBIT_OBJECT
        if (
            int(sync["state"]) == State.RETURN_FOLLOW
            and int(sync["target"]) == Target.NONE
            and config_id == Task.RETURN_GARAGE_LINE
        ):
            return RunMode.RETURN_LINE
        return RunMode.FOLLOW

    def current_object_config_id(self):
        if self.current_sync is None:
            return Task.SEARCH
        return unpack_task_arg_config(self.current_sync["arg"])

    def current_object_id(self):
        if self.current_sync is None:
            return 0
        return unpack_task_arg_object_id(self.current_sync["arg"])

    def _handle_ack_packet(self, packet):
        if self._pending_event is None:
            return
        if int(packet["reliable_seq"]) == int(self._pending_event["reliable_seq"]):
            self._pending_event = None
            self._pending_event_last_sent_ms = None

    def has_pending_event(self):
        return self._pending_event is not None

    def accept_object_observation(self, observation):
        if self.mode != RunMode.APPROACH_OBJECT or self.current_sync is None:
            self._stable_count = 0
            return
        current_sync_seq = int(self.current_sync["reliable_seq"])
        if self._pending_event is not None:
            return
        if self._completed_event_sync_seq == current_sync_seq:
            return
        event_id = self._current_event_id()
        if event_id is None:
            self._stable_count = 0
            return
        x, y, value = observation
        if self._observation_matches_target(x, y, value):
            self._stable_count += 1
        else:
            self._stable_count = 0
            return
        if self._stable_count >= self.required_stable_frames:
            self._create_event(current_sync_seq, event_id, value)

    def accept_return_line_observation(self, img):
        if self.current_sync is None:
            self._stable_count = 0
            return
        current_sync_seq = int(self.current_sync["reliable_seq"])
        if self._pending_event is not None:
            return
        if self._completed_event_sync_seq == current_sync_seq:
            return
        if self._current_event_id() != Event.RETURN_LINE_ALIGNED:
            self._stable_count = 0
            return
        if not bool(self.return_line_gate_enabled):
            self._stable_count = 0
            return
        yellow_ratio = _build_return_line_touch_ratio_percent(img)
        if float(yellow_ratio) <= float(RETURN_LINE_TOUCH_RATIO_THRESHOLD) * 100.0:
            self._stable_count = 0
            return
        self._create_event(current_sync_seq, Event.RETURN_LINE_ALIGNED, int(yellow_ratio))
        self._stable_count = 0

    def _observation_matches_target(self, x, y, value):
        return (
            float(value) >= self.min_area
            and abs(float(x)) <= self.tolerance_x
            and abs(float(y)) <= self.tolerance_y
        )

    def _current_event_id(self):
        if self.current_sync is None:
            return None
        sync_state = int(self.current_sync["state"])
        target = int(self.current_sync["target"])
        config_id = unpack_task_arg_config(self.current_sync["arg"])
        if sync_state == State.APPROACH_OBJECT and target == Target.OBJECT and config_id == Task.SEARCH:
            return Event.TARGET_FOUND
        if sync_state == State.APPROACH_OBJECT and target == Target.OBJECT and config_id == Task.TRANSPORT:
            return Event.ALIGNED
        if sync_state == State.ORBIT and target == Target.OBJECT and config_id == Task.TRANSPORT:
            return Event.ALIGNED
        if sync_state == State.RETURN_FOLLOW and target == Target.NONE and config_id == Task.RETURN_GARAGE_LINE:
            return Event.RETURN_LINE_ALIGNED
        return None

    def _create_event(self, reliable_seq, event, value):
        self._pending_event = {
            "reliable_seq": int(reliable_seq),
            "event": int(event),
            "value": int(float(value)),
        }
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = int(reliable_seq)

    def next_event_frame(self):
        if self._pending_event is None:
            return None
        now_ms = self._now_ms()
        if not should_resend(now_ms, self._pending_event_last_sent_ms, self._event_resend_interval_ms):
            return None
        self._pending_event_last_sent_ms = now_ms
        return format_event_frame(
            self._pending_event["reliable_seq"],
            self._pending_event["event"],
            self._pending_event["value"],
        )


AssistantVisionState = RuntimeState


state = RuntimeState()


def reset_runtime_state():
    state.reset()


def current_event_type():
    return state._current_event_id()


def required_stable_frames():
    return int(state.required_stable_frames)


def resolve_event_value(observation_value, event_value=None):
    _ = event_value
    return int(float(observation_value))


def create_pending_event(reliable_seq, event, value):
    state._create_event(reliable_seq, event, resolve_event_value(value))


def next_event_frame():
    return state.next_event_frame()


def _allocate_local_vision_control_seq():
    reliable_seq = state.next_local_vision_control_seq
    state.next_local_vision_control_seq = (reliable_seq + 1) % SEQ_RING_SIZE
    return reliable_seq


def request_local_vision_control(action):
    pending = state.pending_local_vision_control
    if pending is not None and int(pending["action"]) == int(action):
        return
    state.pending_local_vision_control = {
        "reliable_seq": _allocate_local_vision_control_seq(),
        "action": int(action),
    }
    state.pending_local_vision_control_last_sent_ms = None


def next_local_vision_control_frame():
    pending = state.pending_local_vision_control
    if pending is None:
        return None
    now_ms = default_now_ms()
    if not should_resend(
        now_ms,
        state.pending_local_vision_control_last_sent_ms,
        RELIABLE_RESEND_INTERVAL_MS,
    ):
        return None
    state.pending_local_vision_control_last_sent_ms = now_ms
    return format_local_vision_control_frame(
        pending["reliable_seq"],
        pending["action"],
    )


def send_pending_local_vision_control():
    frame = next_local_vision_control_frame()
    if frame is None:
        return False
    return write_reliable_line(frame)


def ensure_yolo_control_paused():
    if state.local_vision_control_paused:
        return True
    request_local_vision_control(LocalVisionControl.PAUSE)
    send_pending_local_vision_control()
    return False


def request_yolo_control_resume():
    request_local_vision_control(LocalVisionControl.RESUME)
    send_pending_local_vision_control()


def clear_local_vision_control_state():
    state.pending_local_vision_control = None
    state.pending_local_vision_control_last_sent_ms = None
    state.local_vision_control_paused = False


def accept_observation(observation=None, line_y=None):
    if state.local_vision_control_paused:
        state._stable_count = 0
        return
    if state.mode == RunMode.RETURN_LINE:
        state.accept_return_line_observation(state.current_image)
        return
    if observation is None:
        observation = build_object_observation(0, 0, 0, 0)
    if state.current_detection_source == "predict":
        state.accept_object_observation(build_object_observation(0, 0, 0, 0))
        return
    state.accept_object_observation(observation)


def blob_center_y(blob):
    cy_fn = getattr(blob, "cy", None)
    if cy_fn is not None:
        return float(cy_fn())
    _left, top, _right, bottom = blob_rect_to_bbox(blob.rect())
    return (float(top) + float(bottom)) / 2.0


def _tracked_rect():
    rect = state.track_rect
    if rect is None:
        return None
    left, top, right, bottom = rect  # pyright: ignore[reportGeneralTypeIssues]
    return float(left), float(top), float(right), float(bottom)


def _tracked_target_window():
    rect = _tracked_rect()
    if rect is None:
        return None
    if state.track_center_x is None or state.track_bottom_y is None:
        return None
    left, top, right, bottom = rect
    width = max(1.0, float(right) - float(left))
    height = max(1.0, float(bottom) - float(top))
    predicted_center_x = float(state.track_center_x) + float(state.track_velocity_x)
    predicted_bottom_y = float(state.track_bottom_y) + float(state.track_velocity_bottom_y)
    tolerance_x = max(float(OBJECT_X_TOLERANCE_PX) * 2.0, width)
    tolerance_y = max(float(OBJECT_Y_TOLERANCE_PX) * 2.0, height)
    roi_left = predicted_center_x - width * 1.5
    roi_right = predicted_center_x + width * 1.5
    roi_top = float(state.current_image_height) - predicted_bottom_y - height * 1.5
    roi_bottom = roi_top + height * 3.0
    state.track_predicted_center_x = predicted_center_x
    state.track_predicted_bottom_y = predicted_bottom_y
    state.track_predicted_roi = (roi_left, roi_top, roi_right, roi_bottom)
    return (
        predicted_center_x,
        predicted_bottom_y,
        tolerance_x,
        tolerance_y,
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
    )


def _candidate_tracking_failure_reason(candidate):
    window = _tracked_target_window()
    if window is None:
        return TrackFailureReason.OUT_OF_WINDOW
    (
        predicted_center_x,
        predicted_bottom_y,
        tolerance_x,
        tolerance_y,
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
    ) = window
    _task_name, center_x, _center_y, bottom_y, area, blob = candidate
    if abs(float(center_x) - predicted_center_x) > tolerance_x:
        return TrackFailureReason.OUT_OF_WINDOW
    if abs(float(bottom_y) - predicted_bottom_y) > tolerance_y:
        return TrackFailureReason.OUT_OF_WINDOW
    track_area = state.track_area
    if track_area is not None and float(track_area) > 0.0:
        area_ratio = float(area) / float(track_area)
        if area_ratio < 0.5 or area_ratio > 2.0:
            return TrackFailureReason.AREA_JUMP
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    if (
        float(left) <= roi_left
        or float(right) >= roi_right
        or float(top) <= roi_top
        or float(bottom) >= roi_bottom
    ):
        return TrackFailureReason.EDGE_TOUCH
    return TrackFailureReason.NONE


def _candidate_hits_roi_window(candidate):
    return _candidate_tracking_failure_reason(candidate) == TrackFailureReason.NONE


def _tracked_candidate_sort_key(candidate):
    window = _tracked_target_window()
    if window is None:
        return 0.0, 0.0
    predicted_center_x, predicted_bottom_y, _, _, _, _, _, _ = window
    _task_name, center_x, _center_y, bottom_y, area, _blob = candidate
    track_area = state.track_area
    if track_area is None:
        track_area = 0.0
    return (
        abs(float(center_x) - predicted_center_x) + abs(float(bottom_y) - predicted_bottom_y),
        abs(float(area) - float(track_area)),
    )


def _clamp_tracking_value(value, previous_value, max_delta):
    value = float(value)
    previous_value = float(previous_value)
    max_delta = abs(float(max_delta))
    if value > previous_value + max_delta:
        return previous_value + max_delta
    if value < previous_value - max_delta:
        return previous_value - max_delta
    return value


def _filter_candidate_by_track(candidate):
    if state.track_rect is None:
        return candidate
    task_name, center_x, _center_y, bottom_y, area, blob = candidate
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    width = max(1.0, float(right) - float(left))
    height = max(1.0, float(bottom) - float(top))
    filtered_center_x = _clamp_tracking_value(
        center_x,
        state.track_center_x,
        max(float(OBJECT_X_TOLERANCE_PX) * 2.0, width),
    )
    filtered_bottom_y = _clamp_tracking_value(
        bottom_y,
        state.track_bottom_y,
        max(float(OBJECT_Y_TOLERANCE_PX) * 2.0, height),
    )
    filtered_area = float(area)
    if state.track_area is not None and float(state.track_area) > 0.0:
        min_area = float(state.track_area) * 0.5
        max_area = float(state.track_area) * 2.0
        if filtered_area < min_area:
            filtered_area = min_area
        if filtered_area > max_area:
            filtered_area = max_area
    filtered_top = float(state.current_image_height) - filtered_bottom_y
    filtered_left = filtered_center_x - width / 2.0
    filtered_blob = PredictedBlob(
        filtered_left,
        filtered_top,
        filtered_left + width,
        filtered_top + height,
    )
    return (
        task_name,
        filtered_center_x,
        filtered_top + height / 2.0,
        filtered_bottom_y,
        filtered_area,
        filtered_blob,
    )


def _build_tracked_blob_object_candidates(img):
    candidates = _build_dynamic_blob_object_candidates(img)
    if state.track_task_name is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
    if not candidates:
        state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
        return ()
    filtered_candidates = []
    failure_reason = TrackFailureReason.NO_CANDIDATE
    for candidate in candidates:
        reason = _candidate_tracking_failure_reason(candidate)
        if reason == TrackFailureReason.NONE:
            filtered_candidates.append(candidate)
            continue
        if failure_reason == TrackFailureReason.NO_CANDIDATE:
            failure_reason = reason
    candidates = filtered_candidates
    if not candidates:
        state.track_failure_reason = failure_reason
        return ()
    best = min(candidates, key=_tracked_candidate_sort_key)
    state.track_failure_reason = TrackFailureReason.NONE
    return (_filter_candidate_by_track(best),)


def _build_predicted_object_candidates():
    rect = _tracked_rect()
    if rect is None or state.track_task_name is None:
        return ()
    if (
        state.track_center_x is None
        or state.track_center_y is None
        or state.track_bottom_y is None
        or state.track_area is None
    ):
        return ()
    left, top, right, bottom = rect
    width = float(right) - float(left)
    height = float(bottom) - float(top)
    predicted_center_x = float(state.track_center_x) + float(state.track_velocity_x)
    predicted_center_y = float(state.track_center_y) + float(state.track_velocity_bottom_y)
    predicted_bottom_y = float(state.track_bottom_y) + float(state.track_velocity_bottom_y)
    predicted_left = predicted_center_x - width / 2.0
    predicted_top = predicted_center_y - height / 2.0
    state.track_frames_since_yolo += 1
    return (
        (
            state.track_task_name,
            predicted_center_x,
            predicted_center_y,
            predicted_bottom_y,
            float(state.track_area),
            PredictedBlob(
                predicted_left,
                predicted_top,
                predicted_left + width,
                predicted_top + height,
            ),
        ),
    )


def _prefer_tracked_yolo_candidates(candidates):
    if state.track_task_name is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
    tracked_candidates = [candidate for candidate in candidates if _candidate_hits_roi_window(candidate)]
    if not tracked_candidates:
        return None
    best = min(tracked_candidates, key=_tracked_candidate_sort_key)
    return (_filter_candidate_by_track(best),)


def should_use_blob_tracking():
    if (
        state.mode not in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT)
        and not (state.current_sync is None and ASSISTANT_DEBUG_DISPLAY_ENABLED)
    ):
        return False
    if bool(OBJECT_DETECTION_USE_YOLO) and current_sync_is_yolo_only(state.current_sync):
        return False
    if state.track_task_name is None or state.track_rect is None:
        return False
    if state.track_dynamic_threshold is None:
        return False
    if (
        bool(OBJECT_DETECTION_USE_YOLO)
        and state.track_frames_since_yolo >= int(ROI_TRACKING_MAX_FRAMES)
    ):
        return False
    if state.track_roi_failure_frames >= int(ROI_TRACKING_FAILURE_TO_YOLO_FRAMES):
        return False
    return True


def should_run_yolo_for_current_frame():
    if not bool(OBJECT_DETECTION_USE_YOLO):
        return False
    if state.has_pending_event():
        return False
    if (
        state.mode not in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT)
        and not (state.current_sync is None and ASSISTANT_DEBUG_DISPLAY_ENABLED)
    ):
        return False
    if current_sync_is_yolo_only(state.current_sync):
        return not should_skip_yolo_only_frame()
    if current_sync_is_blob_only(state.current_sync):
        return False
    if state.current_sync is not None and current_sync_is_entry_yolo_only(state.current_sync):
        return bool(state.entry_yolo_pending)
    if state.current_sync is not None and not current_sync_allows_yolo():
        return False
    if state.track_rect is None and state.track_dynamic_threshold is not None:
        return False
    return not should_use_blob_tracking()


def current_sync_allows_yolo():
    current_sync = state.current_sync
    if current_sync is None:
        return True
    if current_sync_is_yolo_only(current_sync):
        return True
    if current_sync_is_blob_only(current_sync):
        return False
    if current_sync_is_entry_yolo_only(current_sync):
        return bool(state.entry_yolo_pending)
    return (
        int(current_sync["target"]) == int(Target.OBJECT)
    )


def current_sync_is_entry_yolo_only(current_sync):
    return False


def clear_yolo_only_skip():
    state.yolo_only_skip_frames_remaining = 0
    state.yolo_only_skip_active_for_frame = False


def should_skip_yolo_only_frame():
    state.yolo_only_skip_active_for_frame = False
    if state.yolo_only_skip_frames_remaining <= 0:
        return False
    state.yolo_only_skip_frames_remaining -= 1
    state.yolo_only_skip_active_for_frame = True
    return True


def record_yolo_frame_run():
    clear_yolo_only_skip()
    current_sync = state.current_sync
    if current_sync is None or not current_sync_is_yolo_only(current_sync):
        return
    interval_frames = max(1, int(ASSISTANT_YOLO_ONLY_INTERVAL_FRAMES))
    state.yolo_only_skip_frames_remaining = interval_frames - 1


def current_sync_is_yolo_only(current_sync):
    if current_sync is None:
        return False
    sync_state = int(current_sync["state"])
    target = int(current_sync["target"])
    config_id = unpack_task_arg_config(current_sync["arg"])
    return (
        (
            sync_state == int(State.APPROACH_OBJECT)
            and target == int(Target.OBJECT)
            and config_id == int(Task.TRANSPORT)
        )
        or (
            sync_state == int(State.ORBIT)
            and target == int(Target.OBJECT)
            and config_id == int(Task.TRANSPORT)
        )
    )


def current_sync_is_blob_only(current_sync):
    if current_sync is None:
        return False
    sync_state = int(current_sync["state"])
    target = int(current_sync["target"])
    config_id = unpack_task_arg_config(current_sync["arg"])
    return (
        (
            sync_state == int(State.ORBIT)
            and target == int(Target.OBJECT)
            and config_id == int(Task.ORBIT)
        )
        or (
            sync_state == int(State.TRANSPORT_OBJECT)
            and target == int(Target.OBJECT)
            and config_id == int(Task.TRANSPORT)
        )
    )


def should_refresh_dynamic_threshold_from_yolo():
    current_sync = state.current_sync
    if current_sync is None:
        return False
    return (
        int(current_sync["state"]) == int(State.APPROACH_OBJECT)
        and int(current_sync["target"]) == int(Target.OBJECT)
        and unpack_task_arg_config(current_sync["arg"]) == int(Task.SEARCH)
    )


def remember_object_tracking(task_name, blob, center_x, center_y, bottom_y, area, source):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    previous_task_name = state.track_task_name
    previous_threshold = state.track_dynamic_threshold
    previous_pending_threshold = state.track_pending_dynamic_threshold
    previous_center_x = state.track_center_x
    previous_bottom_y = state.track_bottom_y
    state.track_task_name = task_name
    state.track_object_id = object_task_id(task_name)
    state.track_center_x = float(center_x)
    state.track_center_y = float(center_y)
    state.track_bottom_y = float(bottom_y)
    state.track_area = float(area)
    state.track_rect = (float(left), float(top), float(right), float(bottom))
    if previous_center_x is None:
        state.track_velocity_x = 0.0
    else:
        state.track_velocity_x = float(center_x) - float(previous_center_x)
    if previous_bottom_y is None:
        state.track_velocity_bottom_y = 0.0
    else:
        state.track_velocity_bottom_y = float(bottom_y) - float(previous_bottom_y)
    state.track_source = str(source)
    if source == "yolo":
        state.track_frames_since_yolo = 0
    else:
        state.track_frames_since_yolo += 1
    state.track_roi_failure_frames = 0
    if source == "roi":
        state.track_roi_success_frames += 1
    else:
        state.track_roi_success_frames = 0
    if source == "yolo":
        state.track_confidence = 80
        same_lock = previous_task_name == task_name and previous_threshold is not None
        if not same_lock:
            next_threshold = _build_dynamic_threshold_for_blob(state.current_image, blob)
            state.track_dynamic_threshold = next_threshold
            state.track_pending_dynamic_threshold = None
            state.track_pending_dynamic_threshold_ok_frames = 0
            state.track_dynamic_threshold_health_failures = 0
        elif not should_refresh_dynamic_threshold_from_yolo():
            next_threshold = previous_threshold
            state.track_dynamic_threshold = next_threshold
            state.track_pending_dynamic_threshold = None
            state.track_pending_dynamic_threshold_ok_frames = 0
            state.track_dynamic_threshold_health_failures = 0
            state.track_dynamic_threshold_failure = None
        elif _dynamic_threshold_center_is_healthy(state.current_image, blob, previous_threshold):
            next_threshold = previous_threshold
            state.track_dynamic_threshold = next_threshold
            state.track_pending_dynamic_threshold = None
            state.track_pending_dynamic_threshold_ok_frames = 0
            state.track_dynamic_threshold_health_failures = 0
            state.track_dynamic_threshold_failure = None
        else:
            next_threshold = previous_threshold
            state.track_dynamic_threshold = next_threshold
            state.track_dynamic_threshold_health_failures += 1
            if state.track_dynamic_threshold_health_failures >= int(DYNAMIC_THRESHOLD_REFRESH_FAILURE_FRAMES):
                if previous_pending_threshold is None:
                    pending_threshold = _build_dynamic_threshold_for_blob(state.current_image, blob)
                    pending_ok_frames = 0
                else:
                    pending_threshold = previous_pending_threshold
                    pending_ok_frames = int(state.track_pending_dynamic_threshold_ok_frames)
                if pending_threshold is not None and _dynamic_threshold_center_is_healthy(
                    state.current_image,
                    blob,
                    pending_threshold,
                ):
                    pending_ok_frames += 1
                    state.track_pending_dynamic_threshold = pending_threshold
                    state.track_pending_dynamic_threshold_ok_frames = pending_ok_frames
                    if pending_ok_frames >= int(DYNAMIC_THRESHOLD_REFRESH_CONFIRM_FRAMES):
                        next_threshold = pending_threshold
                        state.track_dynamic_threshold = next_threshold
                        state.track_dynamic_threshold_generation += 1
                        state.track_pending_dynamic_threshold = None
                        state.track_pending_dynamic_threshold_ok_frames = 0
                        state.track_dynamic_threshold_health_failures = 0
                        state.track_dynamic_threshold_failure = None
                else:
                    state.track_pending_dynamic_threshold = None
                    state.track_pending_dynamic_threshold_ok_frames = 0
                    state.track_dynamic_threshold_failure = "calibration_failed"
            else:
                state.track_dynamic_threshold_failure = "health_check_failed"
        state.track_dynamic_threshold_rect = (float(left), float(top), float(right), float(bottom))
        if state.track_dynamic_threshold is not None:
            estimated_area = _estimate_dynamic_foreground_area(
                state.current_image,
                blob,
                state.track_dynamic_threshold,
            )
            if estimated_area is not None:
                state.track_area = float(estimated_area)
            if not same_lock:
                state.track_dynamic_threshold_generation += 1
            if state.track_dynamic_threshold_failure != "health_check_failed":
                state.track_dynamic_threshold_failure = None
        else:
            state.track_dynamic_threshold_failure = "calibration_failed"
    elif source == "roi":
        state.track_confidence = min(100, max(int(state.track_confidence), 60) + 10)
    state.track_failure_reason = TrackFailureReason.NONE


def _candidate_values_for_blob(candidates, best_blob):
    for task_name, center_x, center_y, bottom_y, area, blob in candidates:
        if blob is best_blob:
            return task_name, center_x, center_y, bottom_y, area
    return None, best_blob.cx(), blob_center_y(best_blob), 0.0, blob_area(best_blob)


def _write_all(frame_bytes):
    if not isinstance(frame_bytes, bytes):
        frame_bytes = bytes(frame_bytes)
    remaining = frame_bytes
    uart_device = state.uart_device
    if uart_device is None:
        return False
    while remaining:
        written = uart_device.write(remaining)
        if written is None:
            written = len(remaining)
        written = int(written)
        if written <= 0:
            return False
        remaining = remaining[written:]
    return True


def write_data_line(frame_bytes):
    if state.local_vision_control_paused:
        return False
    _write_all(frame_bytes)


def write_reliable_line(frame_bytes):
    return _write_all(frame_bytes)


def init_uart():
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(0, time=2000)
    sensor.set_auto_gain(False)  # pyright: ignore[reportCallIssue]
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width(), sensor.height()


def _find_control_frame_start(rx_buffer):
    limit = len(rx_buffer) - FRAME_SIZE + 1
    for index in range(limit):
        frame = decode_frame(rx_buffer[index : index + FRAME_SIZE])
        if frame is None:
            continue
        mode = frame["mode"]
        topic = frame["topic"]
        if mode == Mode.TCP and topic == Topic.ASSISTANT_VISION_TASK_SYNC:
            return index
        if mode == Mode.TCP and topic == Topic.LOCAL_VISION_CONTROL:
            return index
        if mode == Mode.ACK and topic == Topic.ASSISTANT_VISION_EVENT_REPORT:
            return index
        if mode == Mode.ACK and topic == Topic.LOCAL_VISION_CONTROL:
            return index
    return -1


def handle_control_frame(frame_bytes):
    packet = parse_local_vision_control_ack_packet(frame_bytes)
    pending_control = state.pending_local_vision_control
    if packet is not None and pending_control is not None:
        if int(packet["reliable_seq"]) == int(pending_control["reliable_seq"]):
            action = int(pending_control["action"])
            state.pending_local_vision_control = None
            state.pending_local_vision_control_last_sent_ms = None
            state.local_vision_control_paused = action == LocalVisionControl.PAUSE
        return None

    return state.handle_control_line(frame_bytes)


def process_uart_input(rx_buffer):
    uart = state.uart_device
    if uart is None:
        return rx_buffer
    size = uart.any()
    if not size:
        return rx_buffer
    data = uart.read(size)
    if data is None:
        return rx_buffer
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif isinstance(data, bytearray):
        data = bytes(data)
    elif not isinstance(data, bytes):
        return rx_buffer
    rx_buffer += data
    while len(rx_buffer) >= FRAME_SIZE:
        frame_start = _find_control_frame_start(rx_buffer)
        if frame_start < 0:
            return rx_buffer[-(FRAME_SIZE - 1) :]
        if frame_start > 0:
            rx_buffer = rx_buffer[frame_start:]
        if len(rx_buffer) < FRAME_SIZE:
            return rx_buffer
        line = rx_buffer[:FRAME_SIZE]
        rx_buffer = rx_buffer[FRAME_SIZE:]
        reply = handle_control_frame(line)
        if reply is not None:
            write_reliable_line(reply)
    return rx_buffer


def _process_follow_frame(img):
    image_width = state.current_image_width
    image_height = state.current_image_height
    candidates = build_blob_candidates(img)
    if not candidates:
        follow_command = build_follow_command(valid=0, err_x=0, err_y=0)
        frame_bytes = format_search_velocity_frame(
            follow_command["command_vx"],
            follow_command["command_vy"],
        )
        write_data_line(frame_bytes)
        return
    cx_screen = float(image_width) / 2.0
    _, pixel_x, pixel_y, _, marker_span, best_blob = choose_best_candidate(
        candidates,
        cx_screen,
        image_height,
    )
    err_x = compute_lateral_error(blob_cx=pixel_x, cx_screen=cx_screen)
    err_y = compute_vertical_error(marker_span=marker_span, target_span=FOLLOW_TARGET_Y)
    follow_command = build_follow_command(valid=1, err_x=err_x, err_y=err_y)
    draw_selected_marker(img, best_blob, pixel_x, pixel_y)
    frame_bytes = format_search_velocity_frame(
        follow_command["command_vx"],
        follow_command["command_vy"],
    )
    write_data_line(frame_bytes)


def _process_object_frame(img):
    observation, best_blob, _task_name, candidates = build_object_observation_and_candidates()
    if candidates:
        config_id = state.current_object_config_id()
        target_x, target_y = build_object_target_point(config_id)
        _draw_debug_protocol_point(img, target_x, target_y)
    if best_blob is not None and state.current_detection_source != "predict":
        task_name, center_x, center_y, bottom_y, area = _candidate_values_for_blob(
            candidates,
            best_blob,
        )
        draw_selected_marker(img, best_blob, center_x, center_y)
        remember_object_tracking(
            task_name,
            best_blob,
            center_x,
            center_y,
            bottom_y,
            area,
            state.current_detection_source,
        )
    elif state.current_detection_source == "yolo":
        state.clear_track()
    if state.mode == RunMode.ORBIT_OBJECT:
        vx, vy = build_orbit_correction_velocity_from_observation(observation)
    else:
        vx, vy = build_search_velocity_from_observation(observation)
    frame_bytes = format_search_velocity_frame(vx, vy)
    write_data_line(frame_bytes)
    accept_observation(observation)
    if ASSISTANT_DEBUG_DISPLAY_ENABLED:
        draw_object_tracking_debug(img)
        flush = getattr(img, "flush", None)
        if flush is not None:
            flush()


def _process_return_line_frame(img):
    accept_observation()


def process_task_frame(img):
    if state.has_pending_event():
        event_frame = next_event_frame()
        if event_frame is not None:
            write_reliable_line(event_frame)
        return
    if state.current_sync is None and ASSISTANT_DEBUG_DISPLAY_ENABLED:
        observation, best_blob, task_name, candidates = build_object_observation_and_candidates()
        debug_log(
            "preview",
            "src=%s cand=%d best=%s conf=%d fail=%s" % (
                tracking_source_debug_name(state.current_detection_source).lower(),
                len(candidates),
                task_name if task_name is not None else "none",
                int(state.track_confidence),
                tracking_failure_debug_name(state.track_failure_reason),
            ),
        )
        if candidates:
            target_x, target_y = build_object_target_point(Task.SEARCH)
            _draw_debug_protocol_point(img, target_x, target_y)
        if best_blob is not None and state.current_detection_source != "predict":
            task_name, center_x, center_y, bottom_y, area = _candidate_values_for_blob(
                candidates,
                best_blob,
            )
            draw_selected_marker(img, best_blob, center_x, center_y)
            remember_object_tracking(
                task_name,
                best_blob,
                center_x,
                center_y,
                bottom_y,
                area,
                state.current_detection_source,
            )
        elif state.current_detection_source == "yolo":
            state.clear_track()
        draw_object_tracking_debug(img)
        flush = getattr(img, "flush", None)
        if flush is not None:
            flush()
        return
    if state.mode in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT):
        _process_object_frame(img)
    elif state.mode == RunMode.RETURN_LINE:
        _process_return_line_frame(img)
    else:
        _process_follow_frame(img)


def run():
    reset_runtime_state()
    state.uart_device = init_uart()
    init_sensor()
    state.yolo_net = load_yolo_model()
    debug_log(
        "boot",
        "debug=%d yolo=%d"
        % (1 if ASSISTANT_DEBUG_DISPLAY_ENABLED else 0, 1 if OBJECT_DETECTION_USE_YOLO else 0),
    )
    last_frame_ms = default_now_ms()
    while True:
        state.rx_buffer = process_uart_input(state.rx_buffer)
        img = sensor.snapshot()
        now_ms = default_now_ms()
        if now_ms >= last_frame_ms:
            state.current_frame_interval_ms = now_ms - last_frame_ms
        else:
            state.current_frame_interval_ms = reference_frame_interval_ms()
        last_frame_ms = now_ms
        img.lens_corr(strength=2.8, zoom=1.0)
        state.current_image = img
        state.current_image_width = int(img.width())
        state.current_image_height = int(img.height())
        state.current_detection_source = "miss"
        if state.pending_local_vision_control is not None:
            send_pending_local_vision_control()
        skip_task_frame = False
        if state.has_pending_event():
            state.current_yolo_candidates = ()
            state.current_object_candidates = ()
            debug_log("skip", "reason=pending_event")
        elif state.mode == RunMode.RETURN_LINE:
            state.current_yolo_candidates = ()
            state.current_object_candidates = ()
            debug_log("skip", "reason=return_line")
        elif (
            state.mode in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT)
            or (state.current_sync is None and ASSISTANT_DEBUG_DISPLAY_ENABLED)
        ):
            need_yolo = should_run_yolo_for_current_frame()
            if not need_yolo:
                if (
                    OBJECT_DETECTION_USE_YOLO
                    and state.yolo_only_skip_active_for_frame
                    and current_sync_is_yolo_only(state.current_sync)
                ):
                    state.current_detection_source = "yolo"
                    state.current_object_candidates = tuple(state.current_yolo_candidates)
                else:
                    state.current_yolo_candidates = ()
                    state.current_object_candidates = tuple(build_object_candidates(img, ()))
                    if state.current_sync is not None:
                        if current_sync_allows_yolo() and state.current_detection_source == "yolo":
                            need_yolo = True
                        elif state.current_detection_source == "yolo":
                            state.current_detection_source = "miss"
                            need_yolo = False
                        else:
                            need_yolo = False
                    else:
                        need_yolo = state.current_detection_source == "yolo"
            if need_yolo:
                yolo_requires_control = state.current_sync is not None
                if yolo_requires_control and not ensure_yolo_control_paused():
                    state.current_yolo_candidates = ()
                    state.current_object_candidates = ()
                    skip_task_frame = True
                else:
                    raw_yolo_candidates = tuple(yolo_detect(img))
                    if raw_yolo_candidates:
                        state.current_detection_source = "yolo"
                        record_yolo_frame_run()
                    state.current_yolo_candidates = raw_yolo_candidates
                    state.current_object_candidates = tuple(build_object_candidates(img, raw_yolo_candidates))
                    if (
                        state.current_sync is not None
                        and current_sync_is_entry_yolo_only(state.current_sync)
                    ):
                        state.entry_yolo_pending = False
                    if yolo_requires_control:
                        request_yolo_control_resume()
        else:
            state.current_yolo_candidates = ()
            state.current_object_candidates = ()
        try:
            if not skip_task_frame:
                process_task_frame(img)
        finally:
            state.record_frame_source(state.current_detection_source)
            gc.collect()


if __name__ == "__main__":
    run()
