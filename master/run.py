"""主车 OpenART 视觉入口 v2."""

# pyright: reportAttributeAccessIssue=false

import image
import sensor
import sys
import tf
import time
from machine import UART
from pyb import LED
import gc

# 是否启用主车上电识别预览, 开启时绕过通信
MASTER_DEBUG_DISPLAY_ENABLED = False
# 是否使用决赛目标选择模式, False 使用初赛模式
IS_FINAL_ROUND = False

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

# 高频速度数据流与可靠协议模式编号分组
class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


# 主车视觉协议 topic 编号分组
class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    MASTER_VISION_TASK_SYNC = 0x10
    MASTER_VISION_EVENT_REPORT = 0x12
    VISION_BOOT_READY = 0x14
    VISION_BOOT_CONFIRM = 0x15


# 主车状态编号分组
class State:
    SEARCH_OBJECT = 1
    ORBITING = 2


# 主车目标编号分组
class Target:
    OBJECT = 1


# 主车任务配置编号分组
class Task:
    SEARCH = 1
    TRANSPORT = 2
    ORBIT = 4


# 主车事件编号分组
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
VISION_REFERENCE_FPS = 25

# 目标最小有效面积阈值
OBJECT_MIN_AREA = 50.0
# 搜索阶段无目标时的默认横向速度
MASTER_MISSING_SEARCH_VX = 0.0
# 搜索阶段无目标时的默认纵向速度
MASTER_MISSING_SEARCH_VY = 2.0
# 搜索阶段横向控制比例系数
MASTER_SEARCH_KP_X = 0.03
# 搜索阶段纵向控制比例系数
MASTER_SEARCH_KP_Y = -0.05
# 搜索阶段最小输出速度
MASTER_SEARCH_MIN_SPEED = 2
# 搜索阶段横向死区, 单位为 px
MASTER_SEARCH_DEADZONE_X_PX = 30.0
# 搜索阶段纵向死区, 单位为 px
MASTER_SEARCH_DEADZONE_Y_PX = 50.0
# 目标横向对齐容差, 单位为 px
OBJECT_X_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_X_PX
# 目标纵向对齐容差, 单位为 px
OBJECT_Y_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_Y_PX
# 判定目标稳定所需连续帧数
OBJECT_STABLE_FRAMES = 1
# 搜索阶段锁定类别所需连续帧数
OBJECT_SELECTION_STABLE_FRAMES = 3
# 搜索阶段锁定类别连续丢失后重新选择所需帧数
OBJECT_LOCK_MISS_FRAMES = 3

# 搜索阶段横向速度上限
MASTER_SEARCH_MAX_VX = 5.0
# 搜索阶段纵向速度上限
MASTER_SEARCH_MAX_VY = 5.0

# 绕目标阶段横向速度修正比例系数
MASTER_ORBIT_KP_X = 0.015
# 绕目标阶段纵向速度修正比例系数
MASTER_ORBIT_KP_Y = -0.05
# 绕目标阶段最小输出速度
MASTER_ORBIT_MIN_SPEED = 0.0
# 绕目标阶段横向死区, 单位为 px
MASTER_ORBIT_DEADZONE_X_PX = 30.0
# 绕目标阶段纵向死区, 单位为 px
MASTER_ORBIT_DEADZONE_Y_PX = 15.0
# 绕目标阶段横向速度上限
MASTER_ORBIT_MAX_VX = 5.0
# 绕目标阶段纵向速度上限
MASTER_ORBIT_MAX_VY = 5.0

# 搜索阶段期望的图像横向位置, 单位为 px
MASTER_SEARCH_TARGET_X_PX = 160.0
# 搜索阶段期望的图像纵向位置, 单位为 px
MASTER_SEARCH_TARGET_Y_PX = 210.0
# 绕目标阶段期望的图像横向位置, 单位为 px
MASTER_ORBIT_TARGET_X_PX = 160.0
# 绕目标阶段期望的图像纵向位置, 单位为 px
MASTER_ORBIT_TARGET_Y_PX = 210.0
# 主车运输阶段期望的图像纵向位置, 单位为 px
MASTER_TRANSPORT_TARGET_Y_PX = 210.0

# 协议约定的图像宽度, 单位为 px
PROTOCOL_IMAGE_WIDTH = 320
# 协议约定的图像高度, 单位为 px
PROTOCOL_IMAGE_HEIGHT = 240

# 帧序号环形空间大小, 取满 1 字节范围
SEQ_RING_SIZE = 256
# 半环阈值, 用于比较环形序号前后关系
SEQ_HALF_RING = 128

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


# 当前主车视觉运行态统一集中在单一状态对象里.
class RuntimeState:
    def __init__(self):
        self.reset()

    def reset(self, next_event_seq=1):
        self.current_task = None
        self.last_task_context_id = None
        self.stable_frame_count = 0
        self.object_selection_stable_count = 0
        self.object_lock_miss_count = 0
        self.next_event_seq = int(next_event_seq) % SEQ_RING_SIZE
        self.pending_event = None
        self.pending_event_last_sent_ms = None
        self.last_event_context_id = None
        self.rx_buffer = b""
        self.yolo_net = None
        self.uart_device = None
        self.current_object_candidates = ()
        self.object_task_name = None
        self.object_selection_task_name = None
        self.last_object_edge_group = 2
        self.current_image = None
        self.current_image_width = PROTOCOL_IMAGE_WIDTH
        self.current_image_height = PROTOCOL_IMAGE_HEIGHT
        self.current_frame_interval_ms = 0.0
        self.current_detection_source = "miss"
        self.boot_ready_acked = False
        self.boot_confirmed = False


state = RuntimeState()


def reset_runtime_state(next_event_seq=1):
    """重置主车视觉运行态."""
    state.reset(next_event_seq)


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
    payload = bytes([mode, topic, seq]) + body + (b"\x00" * (FRAME_BODY_SIZE - len(body)))
    return bytes([FRAME_HEAD]) + payload + bytes([_crc8(payload)])


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
        + bytes([1 if has_omega else 0])
    )


def decode_velocity_body(body):
    return {
        "vx": _unpack_scaled(body, 0),
        "vy": _unpack_scaled(body, 2),
        "omega": _unpack_scaled(body, 4),
        "has_omega": bool(body[6]),
    }


def encode_master_vision_task_sync_body(context_id, state, target, arg):
    return bytes([_require_u8(context_id), _require_u8(state), _require_u8(target)]) + _pack_i16(arg)


def decode_master_vision_task_sync_body(body):
    return {
        "context_id": int(body[0]),
        "state": int(body[1]),
        "target": int(body[2]),
        "arg": _unpack_i16(body, 3),
    }


def encode_master_vision_event_report_body(context_id, event, value):
    return bytes([_require_u8(context_id), _require_u8(event)]) + _pack_i16(value)


def decode_master_vision_event_report_body(body):
    return {
        "context_id": int(body[0]),
        "event": int(body[1]),
        "value": _unpack_i16(body, 2),
    }


def parse_task_sync_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.TCP or frame["topic"] != Topic.MASTER_VISION_TASK_SYNC:
        return None
    packet = decode_master_vision_task_sync_body(frame["body"])
    return {
        "reliable_seq": int(frame["seq"]),
        "context_id": int(packet["context_id"]),
        "state": int(packet["state"]),
        "target": int(packet["target"]),
        "arg": int(packet["arg"]),
    }


def parse_event_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.MASTER_VISION_EVENT_REPORT:
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


def should_resend(now_ms, last_sent_ms, interval_ms):
    if last_sent_ms is None:
        return True
    interval_ms = int(interval_ms)
    if interval_ms <= 0:
        return True
    if int(now_ms) < int(last_sent_ms):
        return True
    return int(now_ms) - int(last_sent_ms) >= interval_ms


def format_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.MASTER_VISION_TASK_SYNC, reliable_seq, b"")


def format_boot_ready_frame():
    return encode_frame(Mode.TCP, Topic.VISION_BOOT_READY, BOOT_READY_SEQ, b"")


def format_boot_confirm_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.VISION_BOOT_CONFIRM, reliable_seq, b"")


def format_search_velocity_frame(vx, vy):
    return encode_frame(
        Mode.UDP,
        Topic.LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_event_frame(reliable_seq, context_id, event, value):
    return encode_frame(
        Mode.TCP,
        Topic.MASTER_VISION_EVENT_REPORT,
        reliable_seq,
        encode_master_vision_event_report_body(context_id, event, value),
    )


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


def _find_control_frame_start(rx_buffer):
    limit = len(rx_buffer) - FRAME_SIZE + 1
    for index in range(limit):
        frame = decode_frame(rx_buffer[index : index + FRAME_SIZE])
        if frame is None:
            continue
        if frame["mode"] == Mode.TCP and frame["topic"] == Topic.MASTER_VISION_TASK_SYNC:
            return index
        if frame["mode"] == Mode.ACK and frame["topic"] == Topic.MASTER_VISION_EVENT_REPORT:
            return index
        if frame["mode"] == Mode.ACK and frame["topic"] == Topic.VISION_BOOT_READY:
            return index
        if frame["mode"] == Mode.TCP and frame["topic"] == Topic.VISION_BOOT_CONFIRM:
            return index
    return -1


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

    def area(self):
        return max(0.0, self._right - self._left) * max(0.0, self._bottom - self._top)


def blob_rect_to_bbox(rect):
    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom):
    return left, top, right, bottom


def blob_area(blob):
    return float(blob.area())


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
        0,
        0,
        0,
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


def object_task_id(task_name):
    for index, task in enumerate(OBJECT_TASKS, 1):
        if task[0] == task_name:
            return index
    return 0


def object_edge_group(task_name):
    object_id = object_task_id(task_name)
    return (object_id + 1) // 2


def current_blob_task_name():
    if state.object_task_name is not None:
        return state.object_task_name
    if not OBJECT_DETECTION_USE_YOLO and OBJECT_TASKS:
        return OBJECT_TASKS[0][0]
    return None


def is_unconfirmed_search_task():
    current_task = state.current_task
    return (
        current_task is not None
        and int(current_task["state"]) == int(State.SEARCH_OBJECT)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.SEARCH)
        and state.last_event_context_id != int(current_task["context_id"])
    )


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


def label_name(label):
    label = int(label)
    if 0 <= label < len(YOLO_LABELS):
        return YOLO_LABELS[label]
    return "unknown"


def yolo_detect(img):
    net = state.yolo_net
    if net is None:
        raise RuntimeError("yolo_net not loaded")
    detect_img = img.copy(YOLO_IMAGE_COPY_SCALE, 1)
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
        candidates.append((task_name, blob.cx(), protocol_bottom, blob.area(), blob))
    return candidates


def build_blob_object_candidates(img):
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
        candidates.append((task_name, blob.cx(), protocol_bottom, blob_area(blob), blob))
    return tuple(candidates)


def build_object_candidates(img, yolo_candidates):
    current_task = state.current_task
    state.current_image = img
    state.current_image_width = int(img.width())
    state.current_image_height = int(img.height())
    if (
        bool(OBJECT_DETECTION_USE_YOLO)
        and current_task is not None
        and int(current_task["target"]) == int(Target.OBJECT)
    ):
        state.current_detection_source = "yolo"
        if is_unconfirmed_search_task():
            locked_task_name = state.object_task_name
            if locked_task_name is None:
                state.object_lock_miss_count = 0
                return tuple(yolo_candidates)
            for candidate in yolo_candidates:
                if candidate[0] == locked_task_name:
                    state.object_lock_miss_count = 0
                    return tuple(yolo_candidates)
            state.object_lock_miss_count += 1
            if state.object_lock_miss_count < int(OBJECT_LOCK_MISS_FRAMES):
                return ()
            state.object_task_name = None
            state.object_selection_task_name = None
            state.object_selection_stable_count = 0
            state.object_lock_miss_count = 0
            return tuple(yolo_candidates)
        task_name = current_blob_task_name()
        if task_name is None:
            return tuple(yolo_candidates)
        return tuple(candidate for candidate in yolo_candidates if candidate[0] == task_name)
    if current_task is not None and int(current_task["target"]) != int(Target.OBJECT):
        state.current_detection_source = "miss"
        return ()
    if not bool(OBJECT_DETECTION_USE_YOLO):
        state.current_detection_source = "blob"
        return build_blob_object_candidates(img)
    return ()


def build_debug_threshold_candidates(img):
    state.current_image = img
    state.current_image_width = int(img.width())
    state.current_image_height = int(img.height())
    candidates = []
    for task in OBJECT_TASKS:
        (
            task_name,
            merge_margin,
            pixels_threshold,
            area_threshold,
            max_side_length,
            _require_all_thresholds,
        ) = object_task_parts(task)
        thresholds = object_task_thresholds(task)
        if not thresholds:
            continue
        blobs = _find_blobs_with_task_config(
            img,
            thresholds,
            pixels_threshold,
            area_threshold,
            merge_margin,
            None,
        )
        for blob in blobs:
            area = blob_area(blob)
            if area < float(area_threshold):
                continue
            if int(max_side_length) > 0 and blob_max_side_length(blob) > float(max_side_length):
                continue
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, protocol_bottom = normalize_bbox_for_protocol(left, top, right, bottom)
            center_x = (float(left) + float(right)) / 2.0
            candidates.append((task_name, center_x, protocol_bottom, area, blob))
    if candidates:
        state.current_detection_source = "blob"
    else:
        state.current_detection_source = "miss"
    return tuple(candidates)


def choose_best_candidate(candidates, target_x, target_y):
    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[2]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    return max(candidates, key=lambda item: float(item[3]))


def is_candidate_occluded(candidate, candidates):
    target_x = float(candidate[1])
    _, target_top, _, target_bottom = blob_rect_to_bbox(
        candidate[4].rect()
    )
    target_center_y = (float(target_top) + float(target_bottom)) / 2.0
    for other in candidates:
        if other is candidate or float(other[2]) <= float(candidate[2]):
            continue
        other_left, other_top, other_right, other_bottom = blob_rect_to_bbox(
            other[4].rect()
        )
        if (
            float(other_left) <= target_x <= float(other_right)
            and float(other_top) <= float(target_bottom)
            and float(other_bottom) >= target_center_y
        ):
            return True
    return False


def candidate_matches_target_side(candidate):
    last_edge = state.last_object_edge_group
    target_edge = object_edge_group(candidate[0])
    center_x = float(state.current_image_width) / 2.0
    candidate_x = float(candidate[1])
    if (last_edge == 1 and target_edge == 3) or (last_edge == 3 and target_edge == 2):
        return candidate_x < center_x
    if (last_edge == 2 and target_edge == 3) or (last_edge == 3 and target_edge == 1):
        return candidate_x > center_x
    return True


def choose_search_candidate(candidates):
    if len(candidates) == 1:
        return candidates[0]
    image_center_x = float(state.current_image_width) / 2.0
    outer_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate_matches_target_side(candidate)
            and not is_candidate_occluded(candidate, candidates)
        ),
        key=lambda item: abs(float(item[1]) - image_center_x),
        reverse=True,
    )[:2]
    if not outer_candidates:
        return None
    for candidate in outer_candidates:
        if object_edge_group(candidate[0]) == state.last_object_edge_group:
            return candidate
    return outer_candidates[0]


def update_object_selection(task_name):
    if not is_unconfirmed_search_task():
        state.object_task_name = task_name
        return
    if task_name != state.object_selection_task_name:
        state.object_selection_task_name = task_name
        state.object_selection_stable_count = 1
        state.stable_frame_count = 0
    else:
        state.object_selection_stable_count += 1
    if state.object_selection_stable_count >= int(OBJECT_SELECTION_STABLE_FRAMES):
        state.object_task_name = task_name


def debug_color_for_task_name(task_name):
    if task_name == "green":
        return (191, 255, 0)
    if task_name == "red":
        return (255, 0, 0)
    if task_name == "blue":
        return (0, 0, 255)
    if task_name == "brown":
        return (80, 40, 20)
    if task_name == "white":
        return (255, 255, 255)
    return (255, 255, 255)


def filter_candidates_in_target_window(candidates, target_y, tolerance_y):
    return [
        candidate
        for candidate in candidates
        if abs(float(candidate[2]) - float(target_y)) <= float(tolerance_y)
    ]


def build_search_target_point(config_id):
    target_x = float(MASTER_SEARCH_TARGET_X_PX)
    if int(config_id) == int(Task.ORBIT):
        return float(MASTER_ORBIT_TARGET_X_PX), float(MASTER_ORBIT_TARGET_Y_PX)
    if int(config_id) == int(Task.TRANSPORT):
        return target_x, float(MASTER_TRANSPORT_TARGET_Y_PX)
    return target_x, float(MASTER_SEARCH_TARGET_Y_PX)


def current_task_config_id():
    current_task = state.current_task
    if current_task is None:
        return Task.SEARCH
    return int(current_task["arg"])


def is_orbit_task_context():
    current_task = state.current_task
    return (
        current_task is not None
        and int(current_task["state"]) == int(State.ORBITING)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.ORBIT)
    )


def build_observation(valid, center_x, bottom_y, area):
    context_id = 0
    current_task = state.current_task
    if current_task is not None:
        context_id = int(current_task["context_id"])
    if int(valid) != 1:
        return context_id, 0.0, 0.0, 0.0
    target_x, target_y = build_search_target_point(current_task_config_id())
    return (
        context_id,
        float(center_x) - float(target_x),
        float(bottom_y) - float(target_y),
        float(area),
    )


def build_observation_and_candidates():
    current_object_candidates = state.current_object_candidates
    if not current_object_candidates:
        return build_observation(0, 0, 0, 0), None, None, current_object_candidates
    candidates = current_object_candidates
    selection_candidates = candidates
    if is_unconfirmed_search_task() and state.object_task_name is not None:
        locked_candidates = tuple(
            candidate
            for candidate in list(candidates)
            if candidate[0] == state.object_task_name
        )
        if locked_candidates:
            selection_candidates = locked_candidates
    target_x, target_y = build_search_target_point(current_task_config_id())
    current_task = state.current_task
    if (
        current_task is not None
        and int(current_task["state"]) == int(State.SEARCH_OBJECT)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.SEARCH)
        and IS_FINAL_ROUND
    ):
        selected = choose_search_candidate(selection_candidates)
        if selected is None:
            return build_observation(0, 0, 0, 0), None, None, candidates
    else:
        selected = choose_best_candidate(selection_candidates, target_x, target_y)
    task_name, center_x, bottom_y, area, best_blob = selected
    return (
        build_observation(1, center_x, bottom_y, area),
        best_blob,
        task_name,
        candidates,
    )


def draw_object_candidates_debug(img, candidates):
    for task_name, _, _, _, blob in candidates:
        color = debug_color_for_task_name(task_name)
        img.draw_rectangle(blob.rect(), color=color, thickness=2)
        rect_x, rect_y, _, _ = blob.rect()
        img.draw_string(
            int(rect_x),
            int(rect_y) - 15,
            task_name,
            color=color,
            scale=2,
            mono_space=False,
        )


def draw_selected_candidate_debug(img, task_name, blob):
    color = debug_color_for_task_name(task_name)
    img.draw_rectangle(blob.rect(), color=color, thickness=3)
    rect_x, rect_y, _, _ = blob.rect()
    img.draw_string(
        int(rect_x),
        int(rect_y) - 30,
        "SELECT",
        color=(255, 255, 255),
        scale=2,
        mono_space=False,
    )


def draw_protocol_target_point_debug(img, target_x, target_y):
    img.draw_cross(int(target_x), int(target_y), color=(255, 255, 0))


def debug_log(tag, text):
    if not MASTER_DEBUG_DISPLAY_ENABLED:
        return
    print("[master_v2][%s] %s" % (str(tag), str(text)))


def draw_search_preview_debug(img, candidates):
    draw_object_candidates_debug(img, candidates)
    if candidates:
        target_x, target_y = build_search_target_point(Task.SEARCH)
        if IS_FINAL_ROUND:
            selected = choose_search_candidate(candidates)
        else:
            selected = choose_best_candidate(candidates, target_x, target_y)
        if selected is not None:
            task_name, _, _, _, best_blob = selected
            draw_selected_candidate_debug(img, task_name, best_blob)
        draw_protocol_target_point_debug(img, target_x, target_y)
    img.flush()


def _clamp(value, limit):
    value = float(value)
    limit = abs(float(limit))
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _apply_min_speed(value, limit, min_speed):
    value = _clamp(value, limit)
    if value == 0.0:
        return 0.0
    min_speed = abs(float(min_speed))
    limit = abs(float(limit))
    if min_speed > limit:
        min_speed = limit
    if 0.0 < value < min_speed:
        return min_speed
    if -min_speed < value < 0.0:
        return -min_speed
    return value


def current_frame_time_scale():
    frame_interval_ms = float(state.current_frame_interval_ms)
    if frame_interval_ms <= 0.0:
        frame_interval_ms = reference_frame_interval_ms()
    return reference_frame_interval_ms() / frame_interval_ms


def _axis_p_velocity(error, deadzone, kp, limit, min_speed):
    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    if float(kp) == 0.0:
        return 0.0
    time_scale = current_frame_time_scale()
    return _apply_min_speed(
        error * float(kp) * time_scale,
        float(limit) * time_scale,
        float(min_speed) * time_scale,
    )


def reference_frame_interval_ms():
    reference_fps = float(VISION_REFERENCE_FPS)
    if reference_fps <= 0.0:
        reference_fps = 1.0
    return 1000.0 / reference_fps


def _build_search_y_velocity(err_y):
    err_y = float(err_y)
    if abs(err_y) <= float(MASTER_SEARCH_DEADZONE_Y_PX):
        return 0.0
    time_scale = current_frame_time_scale()
    return _apply_min_speed(
        err_y * float(MASTER_SEARCH_KP_Y) * time_scale,
        float(MASTER_SEARCH_MAX_VY) * time_scale,
        float(MASTER_SEARCH_MIN_SPEED) * time_scale,
    )


def build_search_velocity_from_observation(observation):
    _, x, y, value = observation
    if float(value) <= 0.0:
        return float(MASTER_MISSING_SEARCH_VX), float(MASTER_MISSING_SEARCH_VY)
    return (
        _axis_p_velocity(
            x,
            MASTER_SEARCH_DEADZONE_X_PX,
            MASTER_SEARCH_KP_X,
            MASTER_SEARCH_MAX_VX,
            MASTER_SEARCH_MIN_SPEED,
        ),
        _build_search_y_velocity(y),
    )


def _build_orbit_y_velocity(err_y):
    err_y = float(err_y)
    if abs(err_y) <= float(MASTER_ORBIT_DEADZONE_Y_PX):
        return 0.0
    if float(MASTER_ORBIT_KP_Y) == 0.0:
        return 0.0
    time_scale = current_frame_time_scale()
    return _apply_min_speed(
        err_y * float(MASTER_ORBIT_KP_Y) * time_scale,
        float(MASTER_ORBIT_MAX_VY) * time_scale,
        float(MASTER_ORBIT_MIN_SPEED) * time_scale,
    )


def build_orbit_correction_velocity_from_observation(observation):
    _, x, y, value = observation
    if float(value) <= 0.0:
        return 0.0, 0.0
    return (
        _axis_p_velocity(
            x,
            MASTER_ORBIT_DEADZONE_X_PX,
            MASTER_ORBIT_KP_X,
            MASTER_ORBIT_MAX_VX,
            MASTER_ORBIT_MIN_SPEED,
        ),
        _build_orbit_y_velocity(y),
    )


def build_task_event_value(img, best_blob, task_name=None):
    _ = img
    _ = best_blob
    current_task = state.current_task
    if (
        current_task is not None
        and int(current_task["state"]) == int(State.SEARCH_OBJECT)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.SEARCH)
        and task_name is not None
    ):
        return object_task_id(task_name)
    return None


def current_event_type():
    current_task = state.current_task
    if current_task is None:
        return None
    task_state = int(current_task["state"])
    target = int(current_task["target"])
    arg = int(current_task["arg"])
    if task_state == State.SEARCH_OBJECT and target == Target.OBJECT and arg == Task.SEARCH:
        return Event.TARGET_FOUND
    if task_state == State.SEARCH_OBJECT and target == Target.OBJECT and arg == Task.TRANSPORT:
        return Event.ALIGNED
    return None


def required_stable_frames():
    return int(OBJECT_STABLE_FRAMES)


def resolve_event_value(observation_value, event_value):
    current_task = state.current_task
    if (
        current_task is not None
        and int(current_task["state"]) == int(State.SEARCH_OBJECT)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.SEARCH)
        and event_value is not None
    ):
        return int(event_value)
    return int(float(observation_value))


def allocate_event_seq():
    reliable_seq = state.next_event_seq
    state.next_event_seq = (reliable_seq + 1) % SEQ_RING_SIZE
    return reliable_seq


def create_pending_event(context_id, event, value):
    state.pending_event = {
        "reliable_seq": allocate_event_seq(),
        "context_id": int(context_id),
        "event": int(event),
        "value": int(value),
    }
    state.pending_event_last_sent_ms = None
    state.last_event_context_id = int(context_id)


def next_event_frame():
    pending_event = state.pending_event
    if pending_event is None:
        return None
    now_ms = default_now_ms()
    if not should_resend(
        now_ms,
        state.pending_event_last_sent_ms,
        RELIABLE_RESEND_INTERVAL_MS,
    ):
        return None
    state.pending_event_last_sent_ms = now_ms
    return format_event_frame(
        pending_event["reliable_seq"],
        pending_event["context_id"],
        pending_event["event"],
        pending_event["value"],
    )


def _process_debug_preview_frame(img):
    if OBJECT_DETECTION_USE_YOLO:
        candidates = tuple(yolo_detect(img))
        state.current_detection_source = "yolo"
    else:
        candidates = build_debug_threshold_candidates(img)
    state.current_object_candidates = candidates
    draw_search_preview_debug(img, candidates)


def accept_observation(observation, img, event_value=None):
    current_task = state.current_task
    if current_task is None:
        return
    context_id = int(current_task["context_id"])
    observed_context_id, error_x, error_y, observation_value = observation
    if int(observed_context_id) != context_id:
        return
    event_type = current_event_type()
    if event_type is None:
        state.stable_frame_count = 0
        return
    if (
        state.pending_event is not None
        or state.last_event_context_id == context_id
    ):
        return
    if (
        float(observation_value) >= float(OBJECT_MIN_AREA)
        and abs(float(error_x)) <= float(OBJECT_X_TOLERANCE_PX)
        and abs(float(error_y)) <= float(OBJECT_Y_TOLERANCE_PX)
    ):
        state.stable_frame_count += 1
    else:
        state.stable_frame_count = 0
        return
    if state.stable_frame_count >= required_stable_frames():
        create_pending_event(
            context_id,
            event_type,
            resolve_event_value(observation_value, event_value),
        )


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

    packet = parse_task_sync_packet(frame_bytes)
    if packet is not None:
        context_id = int(packet["context_id"])
        if is_newer_seq(context_id, state.last_task_context_id):
            state.current_task = {
                "context_id": context_id,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            state.last_task_context_id = context_id
            state.stable_frame_count = 0
            if int(packet["arg"]) == int(Task.SEARCH):
                state.object_lock_miss_count = 0
                state.object_selection_task_name = None
                state.object_selection_stable_count = 0
                if state.object_task_name is not None:
                    state.last_object_edge_group = object_edge_group(state.object_task_name)
                state.object_task_name = None
        return format_ack_frame(packet["reliable_seq"])

    packet = parse_event_ack_packet(frame_bytes)
    pending_event = state.pending_event
    if packet is not None and pending_event is not None:
        if int(packet["reliable_seq"]) == int(pending_event["reliable_seq"]):
            state.pending_event = None
            state.pending_event_last_sent_ms = None
        return None

    return None


def process_uart_input(rx_buffer):
    uart_device = state.uart_device
    if uart_device is None:
        return rx_buffer
    size = uart_device.any()
    if not size:
        return rx_buffer
    data = uart_device.read(size)
    if data is None:
        return rx_buffer
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
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


def process_task_frame(img):
    if state.current_task is None and MASTER_DEBUG_DISPLAY_ENABLED:
        _process_debug_preview_frame(img)
        return
    if state.pending_event is not None:
        event_frame = next_event_frame()
        if event_frame is not None:
            write_reliable_line(event_frame)
        return
    if state.current_task is None:
        return
    observation, best_blob, task_name, _candidates = build_observation_and_candidates()
    if best_blob is not None:
        update_object_selection(task_name)
    elif state.object_task_name is None:
        state.object_selection_task_name = None
        state.object_selection_stable_count = 0
        state.stable_frame_count = 0
    if is_orbit_task_context():
        velocity = build_orbit_correction_velocity_from_observation(observation)
    else:
        velocity = build_search_velocity_from_observation(observation)
    write_data_line(format_search_velocity_frame(*velocity))
    event_value = build_task_event_value(img, best_blob, task_name)
    accept_observation(
        observation,
        img,
        event_value=event_value,
    )
    if MASTER_DEBUG_DISPLAY_ENABLED:
        draw_object_candidates_debug(img, _candidates)
        if best_blob is not None and task_name is not None:
            draw_selected_candidate_debug(img, task_name, best_blob)
        target_x, target_y = build_search_target_point(current_task_config_id())
        draw_protocol_target_point_debug(img, target_x, target_y)
        img.flush()


def init_uart():
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def load_yolo_model():
    if not OBJECT_DETECTION_USE_YOLO:
        return None
    return tf.load(YOLO_MODEL_PATH, load_to_fb=True)


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
    while state.current_task is None:
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
        if MASTER_DEBUG_DISPLAY_ENABLED:
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


def run():
    reset_runtime_state()
    lights = prepare_runtime()
    _, _, blue, _ = lights
    debug_log(
        "boot",
        "debug=%d yolo=%d"
        % (1 if MASTER_DEBUG_DISPLAY_ENABLED else 0, 1 if OBJECT_DETECTION_USE_YOLO else 0),
    )
    last_frame_ms = default_now_ms()

    try:
        while True:
            if not MASTER_DEBUG_DISPLAY_ENABLED:
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
            if MASTER_DEBUG_DISPLAY_ENABLED:
                try:
                    _process_debug_preview_frame(img)
                finally:
                    gc.collect()
                blue.toggle()
                continue
            if state.pending_event is not None:
                state.current_object_candidates = ()
                debug_log("skip", "reason=pending_event")
            elif (
                state.current_task is not None
                and int(state.current_task["target"]) == int(Target.OBJECT)
            ):
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
