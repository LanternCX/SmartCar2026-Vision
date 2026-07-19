"""辅车 OpenART 视觉入口 v2."""

# pyright: reportAttributeAccessIssue=false

import gc
import image
import sensor
import sys
import tf
import time
from machine import UART
from pyb import LED

# 是否启用辅车上电识别预览, 开启时绕过通信
ASSISTANT_DEBUG_DISPLAY_ENABLED = False

# 板载串口编号, 用于和主控通信
UART_ID = 12
# 串口波特率, 单位为 bit/s
UART_BAUDRATE = 115200
# 图像曝光时间, 单位为 us
EXP_TIME_US = 500
# 可靠消息重发间隔, 单位为 ms
RELIABLE_RESEND_INTERVAL_MS = 100
# 启动 READY 重发间隔, 单位为 ms
BOOT_READY_RESEND_INTERVAL_MS = 500
# 等待正式任务时蓝灯翻转周期, 单位为 ms
BOOT_WAIT_BLINK_INTERVAL_MS = 500
# 启动 READY 使用固定可靠序号, 与业务事件 topic 相互独立
BOOT_READY_SEQ = 0
# 最近一次致命错误的本地日志路径
ERROR_LOG_PATH = "/sd/vision_error.log"


class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    ASSISTANT_VISION_TASK_SYNC = 0x11
    ASSISTANT_VISION_EVENT_REPORT = 0x13
    VISION_BOOT_READY = 0x14
    VISION_BOOT_CONFIRM = 0x15


class RunMode:
    FOLLOW = "follow"
    APPROACH_OBJECT = "approach_object"
    ORBIT_OBJECT = "orbit_object"


class State:
    APPROACH_OBJECT = 2
    ORBIT = 3
    TRANSPORT_OBJECT = 4


class Target:
    OBJECT = 1


class Task:
    SEARCH = 1
    TRANSPORT = 2
    ORBIT = 3


class Event:
    TARGET_FOUND = 6
    ALIGNED = 7


# 协议消息体长度, 单位为 byte
FRAME_BODY_SIZE = 10
# 协议帧头标记
FRAME_HEAD = 0xA5
# 协议整帧长度, 单位为 byte
FRAME_SIZE = 15
# 是否启用 YOLO。False 时目标相关流程统一使用色块识别。
OBJECT_DETECTION_USE_YOLO = True
# YOLO 模型文件路径
YOLO_MODEL_PATH = "/sd/yolo.tflite"
# YOLO 输入图像复制缩放比例
YOLO_IMAGE_COPY_SCALE = 0.75
# YOLO 检测最小置信度阈值
YOLO_MIN_SCORE = 0.50
# YOLO 输出标签顺序, 需与模型保持一致
YOLO_LABELS = ("green", "red", "blue", "brown", "white")
# 视觉控制的参考帧率, 用于按时间尺度理解速度响应
VISION_REFERENCE_FPS = 25.0

# 跟随阶段的横向死区, 单位为 px
FOLLOW_X_DEADZONE_PX = 5.0
# 跟随阶段的目标纵向位置, 单位为 px
FOLLOW_TARGET_Y = 100.0
# 跟随阶段的纵向死区, 单位为 px
FOLLOW_Y_DEADZONE_PX = 8.0
# 跟随阶段横向控制比例系数
FOLLOW_CONTROL_KP_X = 0.03
# 跟随阶段纵向控制比例系数
FOLLOW_CONTROL_KP_Y = -0.05
# 跟随阶段最小输出速度
FOLLOW_CONTROL_MIN_SPEED = 0
# 跟随阶段纵向速度上限
FOLLOW_CONTROL_MAX_Y = 5

# 目标最小有效面积阈值
OBJECT_MIN_AREA = 50.0
# 目标丢失时的默认搜索横向速度
OBJECT_MISSING_SEARCH_VX = 0.0
# 目标丢失时的默认搜索纵向速度
OBJECT_MISSING_SEARCH_VY = 2.0
# 接近目标阶段横向控制比例系数
OBJECT_APPROACH_KP_X = 0.02
# 接近目标阶段纵向控制比例系数
OBJECT_APPROACH_KP_Y = -0.05
# 接近目标阶段最小输出速度
OBJECT_APPROACH_MIN_SPEED = 2
# 接近目标阶段横向死区, 单位为 px
OBJECT_APPROACH_DEADZONE_X_PX = 30.0
# 接近目标阶段纵向死区, 单位为 px
OBJECT_APPROACH_DEADZONE_Y_PX = 50.0
# 目标横向对正容差, 单位为 px
OBJECT_X_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_X_PX
# 目标纵向对正容差, 单位为 px
OBJECT_Y_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_Y_PX
# 判定目标稳定所需连续帧数
OBJECT_STABLE_FRAMES = 2

# 接近目标阶段横向速度上限
OBJECT_APPROACH_MAX_VX = 5.0
# 接近目标阶段纵向速度上限
OBJECT_APPROACH_MAX_VY = 5.0

# 绕目标阶段横向速度修正比例系数
OBJECT_ORBIT_KP_X = 0.015
# 绕目标阶段纵向速度修正比例系数
OBJECT_ORBIT_KP_Y = -0.05
# 绕目标阶段最小输出速度
OBJECT_ORBIT_MIN_SPEED = 0.0
# 绕目标阶段横向死区, 单位为 px
OBJECT_ORBIT_DEADZONE_X_PX = 30.0
# 绕目标阶段纵向死区, 单位为 px
OBJECT_ORBIT_DEADZONE_Y_PX = 15.0
# 绕目标阶段横向速度上限
OBJECT_ORBIT_MAX_VX = 5.0
# 绕目标阶段纵向速度上限
OBJECT_ORBIT_MAX_VY = 5.0

# 接近目标阶段期望的图像横向位置, 单位为 px
OBJECT_APPROACH_TARGET_X_PX = 160.0
# 接近目标阶段期望的图像纵向位置, 单位为 px
OBJECT_APPROACH_TARGET_Y_PX = 210.0
# 绕目标阶段期望的图像横向位置, 单位为 px
OBJECT_ORBIT_TARGET_X_PX = 160.0
# 绕目标阶段期望的图像纵向位置, 单位为 px
OBJECT_ORBIT_TARGET_Y_PX = 210.0
# 辅车运输阶段图像纵向命中线, 单位为 px
ASSISTANT_TRANSPORT_TARGET_Y_PX = 210.0

# 协议约定的图像宽度, 单位为 px
PROTOCOL_IMAGE_WIDTH = 320
# 协议约定的图像高度, 单位为 px
PROTOCOL_IMAGE_HEIGHT = 240
# 搬运候选筛选横向窗口, 单位为 px
OBJECT_TRANSPORT_WINDOW_X_PX = float(PROTOCOL_IMAGE_WIDTH)
# 搬运候选筛选纵向窗口, 单位为 px
OBJECT_TRANSPORT_WINDOW_Y_PX = float(PROTOCOL_IMAGE_HEIGHT)

# 帧序号环形空间大小, 取满 1 字节范围
SEQ_RING_SIZE = 256
# 半环阈值, 用于比较环形序号前后关系
SEQ_HALF_RING = 128

# 色块候选合并边距, 单位为 px
OBJECT_BLOB_MERGE_MARGIN = 0
# 色块最小像素数阈值
OBJECT_BLOB_PIXELS_THRESHOLD = 200
# 色块最小面积阈值
OBJECT_BLOB_AREA_THRESHOLD = 200
# 跟随任务的颜色阈值配置
FOLLOW_TASKS = (("marker", (30, 100, 70, 127, -128, 0)),)
# 目标相关任务的筛选参数配置
OBJECT_TASKS = (
    ('red', ((16, 39, 21, 60, 0, 49),), 3, 10, 15, 60, True),
    ('blue', ((32, 57, -11, 12, -50, -23),), 3, 10, 20, 60, True),
    ('brown', ((15, 37, -11, 20, 8, 31),), 3, 10, 50, 80, False),
    ('white', ((58, 70, -11, 9, -11, 9),), 3, 10, 30, 80, True),
    ('green', ((29, 89, -54, -29, 2, 84),), 3, 10, 15, 60, True),
)

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


def encode_assistant_vision_task_sync_body(state_value, target, arg):
    return bytes((_require_u8(state_value), _require_u8(target))) + _pack_i16(arg)


def decode_assistant_vision_task_sync_body(body):
    return {
        "state": int(body[0]),
        "target": int(body[1]),
        "arg": _unpack_i16(body, 2),
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


def format_boot_ready_frame():
    return encode_frame(Mode.TCP, Topic.VISION_BOOT_READY, BOOT_READY_SEQ, b"")


def format_boot_confirm_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.VISION_BOOT_CONFIRM, reliable_seq, b"")


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
    return {
        "reliable_seq": int(frame["seq"]),
        "state": int(packet["state"]),
        "target": int(packet["target"]),
        "arg": int(packet["arg"]),
    }


def parse_event_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.ASSISTANT_VISION_EVENT_REPORT:
        return None
    return {"reliable_seq": int(frame["seq"])}


def parse_boot_ready_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.VISION_BOOT_READY:
        return None
    return {"reliable_seq": int(frame["seq"])}


def parse_boot_confirm_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.TCP or frame["topic"] != Topic.VISION_BOOT_CONFIRM:
        return None
    return {"reliable_seq": int(frame["seq"])}


def is_newer_seq(seq, last_seq):
    if last_seq is None:
        return True
    diff = (int(seq) - int(last_seq)) % SEQ_RING_SIZE
    return diff != 0 and diff < SEQ_HALF_RING


def default_now_ms():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


def _sleep_ms(delay_ms):
    sleep_ms = getattr(time, "sleep_ms", None)
    if sleep_ms is not None:
        sleep_ms(int(delay_ms))
        return
    time.sleep(float(delay_ms) / 1000.0)


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
    return float(left), float(top), float(right), float(bottom)


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


def object_task_id(task_name):
    for index, task in enumerate(OBJECT_TASKS, 1):
        if task[0] == task_name:
            return index
    return 0


def current_blob_task_name():
    if state.object_task_name is not None:
        return state.object_task_name
    task_name = object_task_name_from_id(state.current_object_id())
    if task_name is not None:
        return task_name
    if not OBJECT_DETECTION_USE_YOLO and OBJECT_TASKS:
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
    return tf.load(YOLO_MODEL_PATH, load_to_fb=True)


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


def build_object_blob_candidates(img):
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
    if not thresholds:
        return ()
    blobs = _find_blobs_with_task_config(
        img,
        thresholds,
        pixels_threshold,
        area_threshold,
        merge_margin,
        None,
    )
    candidates = []
    for blob in blobs:
        if blob_area(blob) < float(area_threshold):
            continue
        if int(max_side_length) > 0 and blob_max_side_length(blob) > float(max_side_length):
            continue
        left, top, right, bottom = blob_rect_to_bbox(blob.rect())
        _, _, _, protocol_bottom = normalize_bbox_for_protocol(left, top, right, bottom)
        candidates.append((task_name, blob.cx(), blob.cy(), protocol_bottom, blob_area(blob), blob))
    return tuple(candidates)


def build_object_candidates(img, yolo_candidates):
    state.current_image = img
    state.current_image_width = int(img.width())
    state.current_image_height = int(img.height())
    task_name = current_blob_task_name()
    if OBJECT_DETECTION_USE_YOLO:
        state.current_detection_source = "yolo"
        if task_name is None:
            return tuple(yolo_candidates)
        return tuple(candidate for candidate in yolo_candidates if candidate[0] == task_name)
    state.current_detection_source = "blob"
    return build_object_blob_candidates(img)


def choose_best_candidate(candidates, target_x, target_y):
    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[3]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    return max(candidates, key=lambda item: float(item[4]))


def filter_candidates_in_target_window(candidates, target_x, target_y, tolerance_x, tolerance_y):
    return [
        candidate
        for candidate in candidates
        if abs(float(candidate[1]) - float(target_x)) <= float(tolerance_x)
        and abs(float(candidate[3]) - float(target_y)) <= float(tolerance_y)
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
            OBJECT_TRANSPORT_WINDOW_X_PX,
            OBJECT_TRANSPORT_WINDOW_Y_PX,
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
    _draw_debug_cross(img, x, y)


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


def draw_object_preview_debug(img, candidates):
    for task_name, _center_x, _center_y, _bottom_y, _area, blob in candidates:
        left, top, right, bottom = blob_rect_to_bbox(blob.rect())
        _draw_debug_rect(img, left, top, right, bottom, thickness=2)
        _draw_debug_text(img, left, top - 15, task_name)
    if candidates:
        target_x, target_y = build_object_target_point(Task.SEARCH)
        _task_name, center_x, center_y, _bottom_y, _area, best_blob = choose_best_candidate(
            candidates,
            target_x,
            target_y,
        )
        draw_selected_marker(img, best_blob, center_x, center_y)
        _draw_debug_protocol_point(img, target_x, target_y)
    flush = getattr(img, "flush", None)
    if flush is not None:
        flush()


def debug_log(tag, text):
    if not ASSISTANT_DEBUG_DISPLAY_ENABLED:
        return
    print("[assistant_v2][%s] %s" % (str(tag), str(text)))


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
    return _apply_min_speed(
        err_y * float(OBJECT_APPROACH_KP_Y) * current_frame_time_scale(),
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
    return _apply_min_speed(
        err_y * float(OBJECT_ORBIT_KP_Y) * current_frame_time_scale(),
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
        self._completed_event_sync_seq = None
        self.yolo_net = None
        self.uart_device = None
        self.rx_buffer = b""
        self.current_object_candidates = ()
        self.object_task_name = None
        self.current_image = None
        self.current_image_width = PROTOCOL_IMAGE_WIDTH
        self.current_image_height = PROTOCOL_IMAGE_HEIGHT
        self.current_frame_interval_ms = reference_frame_interval_ms()
        self.current_detection_source = "miss"
        self.boot_ready_acked = False
        self.boot_confirmed = False

    def handle_control_line(self, line):
        sync_packet = parse_task_sync_packet(line)
        if sync_packet is not None:
            return self._handle_sync_packet(sync_packet)
        ack_packet = parse_event_ack_packet(line)
        if ack_packet is not None:
            self._handle_ack_packet(ack_packet)
        return None

    def _handle_sync_packet(self, packet):
        reliable_seq = int(packet["reliable_seq"])
        if self._should_apply_sync(reliable_seq):
            self.current_sync = {
                "reliable_seq": reliable_seq,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            task_name = object_task_name_from_id(unpack_task_arg_object_id(packet["arg"]))
            if task_name is not None:
                self.object_task_name = task_name
            self._last_sync_seq = reliable_seq
            self._pending_event = None
            self._pending_event_last_sent_ms = None
            self.mode = self._mode_from_sync(self.current_sync)
            self._stable_count = 0
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

    def _observation_matches_target(self, x, y, value):
        if self.current_sync is not None:
            config_id = unpack_task_arg_config(self.current_sync["arg"])
            if int(config_id) == int(Task.TRANSPORT):
                return (
                    float(value) >= self.min_area
                    and abs(float(x)) <= self.tolerance_x
                    and abs(float(y)) <= self.tolerance_y
                )
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


def accept_observation(observation=None, line_y=None):
    _ = line_y
    if observation is None:
        observation = build_object_observation(0, 0, 0, 0)
    state.accept_object_observation(observation)


def blob_center_y(blob):
    cy_fn = getattr(blob, "cy", None)
    if cy_fn is not None:
        return float(cy_fn())
    _left, top, _right, bottom = blob_rect_to_bbox(blob.rect())
    return (float(top) + float(bottom)) / 2.0


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
    _write_all(frame_bytes)


def write_reliable_line(frame_bytes):
    return _write_all(frame_bytes)


def init_uart():
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(0, time=2000)
    sensor.set_auto_gain(False)  # pyright: ignore[reportCallIssue]
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width(), sensor.height()


def capture_image():
    return sensor.snapshot().replace(vflip=True, hmirror=True, transpose=False)


def warm_up_detection():
    """完成首帧准备和色块检测预热."""

    img = capture_image()
    img.lens_corr(strength=2.8, zoom=1.0)
    if not OBJECT_DETECTION_USE_YOLO:
        tuple(build_object_candidates(img, ()))
    gc.collect()


def init_status_lights():
    red = LED(1)
    green = LED(2)
    blue = LED(3)
    white = LED(4)
    red.off()
    blue.off()
    white.off()
    green.on()
    return red, green, blue, white


def show_fatal_lights(lights):
    red, green, blue, white = lights
    green.off()
    blue.off()
    white.off()
    red.on()


def write_fatal_error(error):
    """把最近一次致命异常写入 SD 卡, 写入失败时保留原异常."""

    try:
        gc.collect()
        with open(ERROR_LOG_PATH, "w") as log_file:
            sys.print_exception(error, log_file)
    except Exception as log_error:
        print("vision error log write failed:", log_error)


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
        if mode == Mode.ACK and topic == Topic.ASSISTANT_VISION_EVENT_REPORT:
            return index
        if mode == Mode.ACK and topic == Topic.VISION_BOOT_READY:
            return index
        if mode == Mode.TCP and topic == Topic.VISION_BOOT_CONFIRM:
            return index
    return -1


def handle_control_frame(frame_bytes):
    packet = parse_boot_ready_ack_packet(frame_bytes)
    if packet is not None:
        if int(packet["reliable_seq"]) == int(BOOT_READY_SEQ):
            state.boot_ready_acked = True
        return None

    packet = parse_boot_confirm_packet(frame_bytes)
    if packet is not None:
        state.boot_confirmed = True
        return format_boot_confirm_ack_frame(packet["reliable_seq"])

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
            if not write_reliable_line(reply):
                raise RuntimeError("uart reliable write failed")
    return rx_buffer


def perform_boot_handshake(green):
    ready_frame = format_boot_ready_frame()
    last_sent_ms = None
    while not state.boot_confirmed:
        state.rx_buffer = process_uart_input(state.rx_buffer)
        if state.boot_confirmed:
            return
        now_ms = default_now_ms()
        if not state.boot_ready_acked and should_resend(
            now_ms,
            last_sent_ms,
            BOOT_READY_RESEND_INTERVAL_MS,
        ):
            if not write_reliable_line(ready_frame):
                raise RuntimeError("vision boot ready write failed")
            green.toggle()
            last_sent_ms = now_ms
        _sleep_ms(1)


def wait_for_first_task(blue):
    blue.on()
    last_toggle_ms = default_now_ms()
    while state.current_sync is None:
        state.rx_buffer = process_uart_input(state.rx_buffer)
        now_ms = default_now_ms()
        if should_resend(now_ms, last_toggle_ms, BOOT_WAIT_BLINK_INTERVAL_MS):
            blue.toggle()
            last_toggle_ms = now_ms
        _sleep_ms(1)


def prepare_runtime():
    lights = init_status_lights()
    _, green, blue, _ = lights
    try:
        gc.collect()
        state.yolo_net = load_yolo_model()
        init_sensor()
        warm_up_detection()
        if ASSISTANT_DEBUG_DISPLAY_ENABLED:
            green.off()
            return lights
        state.uart_device = init_uart()
        perform_boot_handshake(green)
        green.off()
        wait_for_first_task(blue)
        return lights
    except Exception as error:
        write_fatal_error(error)
        show_fatal_lights(lights)
        raise


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
    if best_blob is not None:
        task_name, center_x, center_y, _bottom_y, _area = _candidate_values_for_blob(
            candidates,
            best_blob,
        )
        state.object_task_name = task_name
        draw_selected_marker(img, best_blob, center_x, center_y)
    if state.mode == RunMode.ORBIT_OBJECT:
        vx, vy = build_orbit_correction_velocity_from_observation(observation)
    else:
        vx, vy = build_search_velocity_from_observation(observation)
    frame_bytes = format_search_velocity_frame(vx, vy)
    write_data_line(frame_bytes)
    accept_observation(observation)
    if ASSISTANT_DEBUG_DISPLAY_ENABLED:
        flush = getattr(img, "flush", None)
        if flush is not None:
            flush()


def process_task_frame(img):
    if state.has_pending_event():
        event_frame = next_event_frame()
        if event_frame is not None:
            write_reliable_line(event_frame)
        return
    if state.mode in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT):
        _process_object_frame(img)
    else:
        _process_follow_frame(img)


def run():
    reset_runtime_state()
    lights = prepare_runtime()
    _, _, blue, _ = lights
    debug_log(
        "boot",
        "debug=%d yolo=%d"
        % (1 if ASSISTANT_DEBUG_DISPLAY_ENABLED else 0, 1 if OBJECT_DETECTION_USE_YOLO else 0),
    )
    last_frame_ms = default_now_ms()
    try:
        while True:
            if not ASSISTANT_DEBUG_DISPLAY_ENABLED:
                state.rx_buffer = process_uart_input(state.rx_buffer)
            img = capture_image()
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
            if ASSISTANT_DEBUG_DISPLAY_ENABLED:
                yolo_candidates = (
                    tuple(yolo_detect(img)) if OBJECT_DETECTION_USE_YOLO else ()
                )
                state.current_object_candidates = tuple(
                    build_object_candidates(img, yolo_candidates)
                )
                try:
                    draw_object_preview_debug(img, state.current_object_candidates)
                finally:
                    gc.collect()
                blue.toggle()
                continue
            if state.has_pending_event():
                state.current_object_candidates = ()
                debug_log("skip", "reason=pending_event")
            elif state.mode in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT):
                yolo_candidates = (
                    tuple(yolo_detect(img)) if OBJECT_DETECTION_USE_YOLO else ()
                )
                state.current_object_candidates = tuple(
                    build_object_candidates(img, yolo_candidates)
                )
            else:
                state.current_object_candidates = ()
            try:
                process_task_frame(img)
            finally:
                gc.collect()
            blue.toggle()
    except Exception as error:
        write_fatal_error(error)
        show_fatal_lights(lights)
        raise


if __name__ == "__main__":
    run()
