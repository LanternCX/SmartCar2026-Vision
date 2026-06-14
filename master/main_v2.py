"""主车 OpenART 视觉入口 v2."""

import image
import sensor
import tf
import time
from machine import UART
import gc

# 主车视觉固定使用 UART6 对应的 OpenART 串口 2.
UART_ID = 2
# 主辅车本地视觉链路统一波特率.
UART_BAUDRATE = 115200
# 相机固定曝光时间, 单位为微秒.
EXP_TIME_US = 500
# 未确认可靠事件的重发间隔, 单位为毫秒.
RELIABLE_RESEND_INTERVAL_MS = 100

# 高频速度数据流与可靠协议模式编号分组.
class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


# 主车视觉协议 topic 编号分组.
class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    MASTER_VISION_TASK_SYNC = 0x10
    MASTER_VISION_EVENT_REPORT = 0x12

# 主车状态编号分组.
class State:
    SEARCH_OBJECT = 1
    ORBITING = 2
    TRANSPORT_OBJECT = 4
    RETURN_GARAGE_RETREAT = 6
    RETURN_GARAGE_LINE = 7


# 主车目标编号分组.
class Target:
    OBJECT = 1
    EDGE_LINE = 3


# 主车任务配置编号分组.
class Task:
    SEARCH = 1
    TRANSPORT = 2
    TRANSPORT_FINISH = 3
    ORBIT = 4
    RETURN_GARAGE_LINE = 5


# 主车事件编号分组.
class Event:
    TARGET_FOUND = 6
    ALIGNED = 7
    ARRIVED = 8
    RETURN_LINE_ALIGNED = 10
    RETURN_GARAGE_FINISHED = 12


# ROI 跟踪失败原因编号分组.
class TrackFailureReason:
    NONE = 0
    NO_CANDIDATE = 1
    OUT_OF_WINDOW = 2
    AREA_JUMP = 3
    EDGE_TOUCH = 4
    POOR_SEPARATION = 5

# 固定短帧的 body 槽位长度, 单位为字节.
FRAME_BODY_SIZE = 8
# 固定短帧的帧头字节.
FRAME_HEAD = 0xA5
# 固定短帧总长度, 单位为字节.
FRAME_SIZE = 13

# YOLO 主线模型在板端的固定路径.
YOLO_MODEL_PATH = "/sd/yolo.tflite"
# YOLO 推理前复制图像时使用的缩放比例.
YOLO_IMAGE_COPY_SCALE = 0.75
# 低于该置信度的检测框直接丢弃.
YOLO_MIN_SCORE = 0.50
# YOLO 标签编号到任务名的稳定映射.
YOLO_LABELS = ("tennis", "red", "blue", "brown", "white")
# 调试模式打开后在屏幕上显示识别框和目标点.
MASTER_DEBUG_DISPLAY_ENABLED = False
# 旧版高帧率手感对应的参考帧率.
VISION_REFERENCE_FPS = 30

# 候选框被认为有效目标的最小面积阈值.
OBJECT_MIN_AREA = 50.0
# 搜索阶段无目标时的默认横向速度.
MASTER_MISSING_SEARCH_VX = 0.0
# 搜索阶段无目标时的默认纵向速度.
MASTER_MISSING_SEARCH_VY = 2.0
# 搜索阶段横向像素误差到速度的比例增益.
MASTER_SEARCH_KP_X = 0.05
# 搜索阶段纵向像素误差到速度的比例增益.
MASTER_SEARCH_KP_Y = -0.15
# 搜索阶段非零速度的最小输出幅值.
MASTER_SEARCH_MIN_SPEED = 2.0
# 搜索阶段横向像素死区.
MASTER_SEARCH_DEADZONE_X_PX = 15.0
# 搜索阶段纵向像素死区.
MASTER_SEARCH_DEADZONE_Y_PX = 8.0
# 事件判定使用的横向容差, 直接复用搜索横向死区.
OBJECT_X_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_X_PX
# 事件判定使用的纵向容差, 直接复用搜索纵向死区.
OBJECT_Y_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_Y_PX
# 连续满足目标窗口和面积条件的稳定帧数.
OBJECT_STABLE_FRAMES = 3

# 允许在两次 YOLO 之间连续使用 ROI 的最大帧数。
ROI_TRACKING_MAX_FRAMES = 15
# ROI 连续失手达到该值后立即回退到 YOLO。
ROI_TRACKING_FAILURE_TO_YOLO_FRAMES = 1
# 搜索阶段横向速度限幅.
MASTER_SEARCH_MAX_VX = 5.0
# 搜索阶段纵向速度限幅.
MASTER_SEARCH_MAX_VY = 5.0

# 绕行修正阶段横向像素误差到速度的比例增益.
MASTER_ORBIT_KP_X = 0.05
# 绕行修正阶段纵向像素误差到速度的比例增益.
MASTER_ORBIT_KP_Y = -0.30
# 绕行修正阶段非零速度的最小输出幅值.
MASTER_ORBIT_MIN_SPEED = 0.0
# 绕行修正阶段横向像素死区.
MASTER_ORBIT_DEADZONE_X_PX = 15.0
# 绕行修正阶段纵向像素死区.
MASTER_ORBIT_DEADZONE_Y_PX = 8.0
# 绕行修正阶段横向速度限幅.
MASTER_ORBIT_MAX_VX = 5.0
# 绕行修正阶段纵向速度限幅.
MASTER_ORBIT_MAX_VY = 5.0

# 搜索阶段目标点横向坐标, 单位为协议像素.
MASTER_SEARCH_TARGET_X_PX = 160.0
# 搜索阶段目标点纵向坐标, 单位为协议像素.
MASTER_SEARCH_TARGET_Y_PX = 210.0
# 绕行修正阶段目标点横向坐标, 单位为协议像素.
MASTER_ORBIT_TARGET_X_PX = 160.0
# 绕行修正阶段目标点纵向坐标, 单位为协议像素.
MASTER_ORBIT_TARGET_Y_PX = 210.0
# 搬运入口对正阶段目标点纵向坐标, 单位为协议像素.
MASTER_TRANSPORT_TARGET_Y_PX = 240.0

# 搬运收尾环带相对目标框的外扩像素.
FINISH_HOOK_RING_EXPAND_PX = 5
# 搬运收尾使用的黄色阈值.
FINISH_HOOK_YELLOW_THRESHOLD = (58, 87, -32, -12, 64, 84)
# 搬运收尾判定黄色接触的占比阈值.
FINISH_HOOK_YELLOW_RATIO_THRESHOLD = 0.1
# 搬运收尾在脱离接触后需要保持的稳定帧数.
FINISH_HOOK_STABLE_FRAMES = 2

# 回库黄线识别使用的黄色阈值.
RETURN_GARAGE_LINE_YELLOW_THRESHOLD = (58, 87, -32, -12, 64, 84)
# 回库黄线中心采样区域的半宽, 单位为像素.
RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX = 5
# 回库黄线目标纵向坐标, 单位为协议像素.
RETURN_GARAGE_LINE_TARGET_Y_PX = 220.0
# 回库黄线纵向控制死区, 单位为像素.
RETURN_GARAGE_LINE_DEADZONE_Y_PX = 4.0
# 回库黄线对正事件使用的纵向容差, 单位为像素.
RETURN_GARAGE_LINE_ALIGN_TOLERANCE_PX = 4.0
# 回库黄线纵向像素误差到速度的比例增益.
RETURN_GARAGE_LINE_KP_Y = -0.05
# 回库黄线纵向速度限幅.
RETURN_GARAGE_LINE_MAX_VY = 5.0
# 回库黄线纵向非零速度的最小输出幅值.
RETURN_GARAGE_LINE_MIN_SPEED = 0.0
# 单列黄线采样允许参与中心计算的最大厚度, 单位为像素.
RETURN_GARAGE_LINE_MAX_THICKNESS_PX = 30
# 黄线点被认定为有效连通线段所需的最小水平连通长度, 单位为像素.
RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50
# 回库完成判定使用的固定采样行坐标, 单位为协议像素.
RETURN_LINE_FINISH_ROW_Y_PX = 160
# 回库完成判定使用的固定采样列坐标, 单位为协议像素.
RETURN_LINE_FINISH_COLUMN_X_PX = 270
# 回库完成判定所需的连续满足帧数.
RETURN_LINE_MISSING_FINISH_FRAMES = 5
PROTOCOL_IMAGE_WIDTH = 320
PROTOCOL_IMAGE_HEIGHT = 240

# 可靠序号环空间总长度.
SEQ_RING_SIZE = 256
# 判断环形序号新旧关系时使用的半环长度.
SEQ_HALF_RING = 128

# 当前主线开放的物体任务名与物体编号映射.
OBJECT_TASKS = (
    ('red', ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),
)

# 固定点编码允许的最小 i16 值.
_I16_MIN = -32768
# 固定点编码允许的最大 i16 值.
_I16_MAX = 32767
# 浮点速度和观测量编码到 i16 时使用的缩放倍数.
_SCALE = 1000


# 当前主车视觉运行态统一集中在单一状态对象里.
class RuntimeState:
    def __init__(self):
        self.current_task = None
        self.last_task_context_id = None
        self.stable_frame_count = 0
        self.next_event_seq = 1
        self.pending_event = None
        self.pending_event_last_sent_ms = None
        self.last_event_context_id = None
        self.finish_contact_seen = False
        self.last_return_line_y = None
        self.return_line_finish_missing_count = 0
        self.rx_buffer = b""
        self.yolo_net = None
        self.uart_device = None
        self.current_yolo_candidates = ()
        self.current_object_candidates = ()
        self.current_image = None
        self.current_image_width = PROTOCOL_IMAGE_WIDTH
        self.current_image_height = PROTOCOL_IMAGE_HEIGHT
        self.current_frame_interval_ms = 0.0
        self.current_detection_source = "miss"
        self.clear_track()

    def clear_track(self):
        self.track_task_name = None
        self.track_object_id = 0
        self.track_center_x = None
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

    def reset(self, next_event_seq=1):
        self.current_task = None
        self.last_task_context_id = None
        self.stable_frame_count = 0
        self.next_event_seq = int(next_event_seq) % SEQ_RING_SIZE
        self.pending_event = None
        self.pending_event_last_sent_ms = None
        self.last_event_context_id = None
        self.finish_contact_seen = False
        self.last_return_line_y = None
        self.return_line_finish_missing_count = 0
        self.rx_buffer = b""
        self.yolo_net = None
        self.uart_device = None
        self.current_yolo_candidates = ()
        self.current_object_candidates = ()
        self.current_image = None
        self.current_image_width = PROTOCOL_IMAGE_WIDTH
        self.current_image_height = PROTOCOL_IMAGE_HEIGHT
        self.current_frame_interval_ms = reference_frame_interval_ms()
        self.current_detection_source = "miss"
        self.clear_track()


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


def is_newer_seq(seq, last_seq):
    if last_seq is None:
        return True
    diff = (int(seq) - int(last_seq)) % SEQ_RING_SIZE
    return diff != 0 and diff < SEQ_HALF_RING


def default_now_ms():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


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

    def area(self):
        return max(0.0, self._right - self._left) * max(0.0, self._bottom - self._top)


def blob_rect_to_bbox(rect):
    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom):
    image_height = state.current_image_height
    normalized_top = image_height - bottom
    normalized_bottom = image_height - top
    return left, normalized_top, right, normalized_bottom


def blob_area(blob):
    return float(blob.area())


def blob_max_side_length(blob):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(float(right - left), float(bottom - top))


def task_thresholds(thresholds):
    if len(thresholds) == 6 and not isinstance(thresholds[0], (tuple, list)):
        return (thresholds,)
    return thresholds


def object_task_parts(task):
    if len(task) >= 7:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            max(0, int(task[5])),
            bool(task[6]),
        )
    if len(task) >= 6:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            0,
            bool(task[5]),
        )
    if len(task) >= 5:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            0,
            True,
        )
    return (
        task[0],
        task_thresholds(task[1]),
        0,
        0,
        0,
        0,
        True,
    )


def object_task_id(task_name):
    for index, task in enumerate(OBJECT_TASKS, 1):
        if task[0] == task_name:
            return index
    return 0


def object_thresholds_for_task_name(task_name):
    for task in OBJECT_TASKS:
        if task[0] == task_name:
            return task_thresholds(task[1])
    return ()


def _find_blobs_with_task_config(img, thresholds, pixels_threshold, area_threshold, merge_margin):
    try:
        return img.find_blobs(
            list(thresholds),
            pixels_threshold=pixels_threshold,
            area_threshold=1,
            merge=True,
            margin=max(0, int(merge_margin)),
        )
    except TypeError:
        return img.find_blobs(
            list(thresholds),
            pixels_threshold=pixels_threshold,
            area_threshold=1,
            merge=True,
        )


def _blob_code(blob):
    code_fn = getattr(blob, "code", None)
    if code_fn is not None:
        return int(code_fn())
    try:
        return int(blob[8])
    except Exception:
        return None


def _blob_matches_required_thresholds(blob, threshold_count, require_all_thresholds):
    if int(threshold_count) <= 1 or not bool(require_all_thresholds):
        return True
    code = _blob_code(blob)
    if code is None:
        return True
    expected_code = (1 << int(threshold_count)) - 1
    return (code & expected_code) == expected_code


def blob_bbox_overlaps(left, top, right, bottom, other_blob):
    other_left, other_top, other_right, other_bottom = blob_rect_to_bbox(other_blob.rect())
    return (
        min(float(right), float(other_right)) > max(float(left), float(other_left))
        and min(float(bottom), float(other_bottom)) > max(float(top), float(other_top))
    )


def blob_matches_all_thresholds(img, blob, thresholds, pixels_threshold, area_threshold, merge):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    for threshold in thresholds[1:]:
        blobs = img.find_blobs(
            [threshold],
            pixels_threshold=pixels_threshold,
            area_threshold=area_threshold,
            merge=merge,
        )
        matched = False
        for other_blob in blobs:
            if blob_bbox_overlaps(left, top, right, bottom, other_blob):
                matched = True
                break
        if not matched:
            return False
    return True


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
    for detected in tf.detect(net, detect_img):
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


def _build_blob_object_candidates(img):
    candidates = []
    for task in OBJECT_TASKS:
        (
            task_name,
            thresholds,
            merge_margin,
            pixels_threshold,
            area_threshold,
            max_side_length,
            require_all_thresholds,
        ) = object_task_parts(task)
        if len(thresholds) <= 0:
            continue
        blobs = _find_blobs_with_task_config(
            img,
            thresholds,
            pixels_threshold,
            area_threshold,
            merge_margin,
        )
        for blob in blobs:
            if not _blob_matches_required_thresholds(blob, len(thresholds), require_all_thresholds):
                continue
            if len(thresholds) > 1 and not blob_matches_all_thresholds(
                img,
                blob,
                thresholds,
                pixels_threshold,
                area_threshold,
                True,
            ):
                continue
            if blob_area(blob) < float(area_threshold):
                continue
            if int(max_side_length) > 0 and blob_max_side_length(blob) > float(max_side_length):
                continue
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, protocol_bottom = normalize_bbox_for_protocol(left, top, right, bottom)
            candidates.append((task_name, blob.cx(), protocol_bottom, blob_area(blob), blob))
    return candidates


def _tracked_rect():
    rect = state.track_rect
    if rect is None:
        return None
    left, top, right, bottom = rect
    return float(left), float(top), float(right), float(bottom)


def _tracked_target_window():
    rect = _tracked_rect()
    if rect is None:
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
    _task_name, center_x, bottom_y, area, blob = candidate
    if abs(float(center_x) - predicted_center_x) > tolerance_x:
        return TrackFailureReason.OUT_OF_WINDOW
    if abs(float(bottom_y) - predicted_bottom_y) > tolerance_y:
        return TrackFailureReason.OUT_OF_WINDOW
    if float(state.track_area) > 0.0:
        area_ratio = float(area) / float(state.track_area)
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
    predicted_center_x, predicted_bottom_y, _, _, _, _, _, _ = _tracked_target_window()
    _task_name, center_x, bottom_y, area, _blob = candidate
    return (
        abs(float(center_x) - predicted_center_x) + abs(float(bottom_y) - predicted_bottom_y),
        abs(float(area) - float(state.track_area)),
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
    task_name, center_x, bottom_y, area, blob = candidate
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
        filtered_bottom_y,
        filtered_area,
        filtered_blob,
    )


def should_use_blob_tracking():
    if state.current_task is None and not MASTER_DEBUG_DISPLAY_ENABLED:
        return False
    if is_return_line_task_context():
        return False
    if state.track_task_name is None or state.track_rect is None:
        return False
    if state.track_frames_since_yolo >= int(ROI_TRACKING_MAX_FRAMES):
        return False
    if state.track_roi_failure_frames >= int(ROI_TRACKING_FAILURE_TO_YOLO_FRAMES):
        return False
    return True


def should_run_yolo_for_current_frame():
    if state.pending_event is not None:
        return False
    if is_return_line_task_context():
        return False
    if state.current_task is None and not MASTER_DEBUG_DISPLAY_ENABLED:
        return False
    return not should_use_blob_tracking()


def _build_predicted_object_candidates():
    rect = _tracked_rect()
    if rect is None or state.track_task_name is None:
        return ()
    left, top, right, bottom = rect
    width = float(right) - float(left)
    height = float(bottom) - float(top)
    predicted_center_x = float(state.track_center_x) + float(state.track_velocity_x)
    predicted_bottom_y = float(state.track_bottom_y) + float(state.track_velocity_bottom_y)
    predicted_left = predicted_center_x - width / 2.0
    predicted_top = float(state.current_image_height) - predicted_bottom_y - height
    state.track_frames_since_yolo += 1
    return (
        (
            state.track_task_name,
            predicted_center_x,
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


def build_object_candidates(img, yolo_candidates):
    if should_use_blob_tracking():
        candidates = _build_blob_object_candidates(img)
        if state.track_task_name is not None:
            candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
        filtered_candidates = []
        failure_reason = TrackFailureReason.NO_CANDIDATE
        if candidates:
            for candidate in candidates:
                reason = _candidate_tracking_failure_reason(candidate)
                if reason == TrackFailureReason.NONE:
                    filtered_candidates.append(candidate)
                    continue
                if failure_reason == TrackFailureReason.NO_CANDIDATE:
                    failure_reason = reason
        candidates = filtered_candidates
        if candidates:
            state.current_detection_source = "roi"
            filtered_candidates = []
            failure_reason = TrackFailureReason.NO_CANDIDATE
            for candidate in candidates:
                if not _blob_is_separable_from_background(img, candidate[0], candidate[4]):
                    if failure_reason == TrackFailureReason.NO_CANDIDATE:
                        failure_reason = TrackFailureReason.POOR_SEPARATION
                    continue
                filtered_candidates.append(candidate)
            if filtered_candidates:
                best = min(filtered_candidates, key=_tracked_candidate_sort_key)
                state.record_roi_attempt(True)
                state.track_failure_reason = TrackFailureReason.NONE
                return (_filter_candidate_by_track(best),)
            state.track_failure_reason = failure_reason
        state.track_failure_reason = failure_reason
        state.record_roi_attempt(False)
        state.track_roi_failure_frames += 1
        if state.track_roi_failure_frames < int(ROI_TRACKING_FAILURE_TO_YOLO_FRAMES):
            state.current_detection_source = "predict"
            state.track_confidence = max(0, int(state.track_confidence) - 20)
            return _build_predicted_object_candidates()
        state.record_roi_fallback()
    state.current_detection_source = "yolo"
    candidates = tuple(yolo_candidates)
    if state.track_rect is None:
        state.track_failure_reason = TrackFailureReason.NONE
        return candidates
    tracked_candidates = _prefer_tracked_yolo_candidates(candidates)
    if tracked_candidates is not None:
        state.track_failure_reason = TrackFailureReason.NONE
        return tracked_candidates
    if state.track_task_name is not None:
        same_task_candidates = [candidate for candidate in candidates if candidate[0] == state.track_task_name]
        if same_task_candidates:
            best = min(same_task_candidates, key=_tracked_candidate_sort_key)
            state.track_failure_reason = TrackFailureReason.OUT_OF_WINDOW
            return (_filter_candidate_by_track(best),)
    state.track_failure_reason = TrackFailureReason.NO_CANDIDATE
    return candidates


def remember_object_tracking(task_name, blob, center_x, bottom_y, area, source):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    previous_center_x = state.track_center_x
    previous_bottom_y = state.track_bottom_y
    state.track_task_name = task_name
    state.track_object_id = object_task_id(task_name)
    state.track_center_x = float(center_x)
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
    elif source == "roi":
        state.track_confidence = min(100, max(int(state.track_confidence), 60) + 10)
    state.track_failure_reason = TrackFailureReason.NONE

def choose_best_candidate(candidates, target_x, target_y):
    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[2]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    return max(candidates, key=lambda item: float(item[3]))


def debug_color_for_task_name(task_name):
    if task_name == "tennis":
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
    if int(config_id) in (
        int(Task.TRANSPORT),
        int(Task.TRANSPORT_FINISH),
    ):
        return target_x, float(MASTER_TRANSPORT_TARGET_Y_PX)
    return target_x, float(MASTER_SEARCH_TARGET_Y_PX)


def current_task_config_id():
    current_task = state.current_task
    if current_task is None:
        return Task.SEARCH
    return int(current_task["arg"])


def is_finish_task_context():
    current_task = state.current_task
    return (
        current_task is not None
        and int(current_task["state"]) == int(State.TRANSPORT_OBJECT)
        and int(current_task["target"]) == int(Target.EDGE_LINE)
        and int(current_task["arg"]) == int(Task.TRANSPORT_FINISH)
    )


def is_orbit_task_context():
    current_task = state.current_task
    return (
        current_task is not None
        and int(current_task["state"]) == int(State.ORBITING)
        and int(current_task["target"]) == int(Target.OBJECT)
        and int(current_task["arg"]) == int(Task.ORBIT)
    )


def is_return_line_task_context():
    current_task = state.current_task
    return (
        current_task is not None
        and int(current_task["target"]) == int(Target.EDGE_LINE)
        and int(current_task["arg"]) == int(Task.RETURN_GARAGE_LINE)
        and int(current_task["state"]) in (
            int(State.RETURN_GARAGE_RETREAT),
            int(State.RETURN_GARAGE_LINE),
        )
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


def build_return_line_observation(line_y):
    context_id = 0
    current_task = state.current_task
    if current_task is not None:
        context_id = int(current_task["context_id"])
    if line_y is None:
        return context_id, 0.0, 0.0, 0.0
    return (
        context_id,
        0.0,
        float(line_y) - float(RETURN_GARAGE_LINE_TARGET_Y_PX),
        float(line_y),
    )


def build_observation_and_candidates():
    current_object_candidates = state.current_object_candidates
    if not current_object_candidates:
        return build_observation(0, 0, 0, 0), None, None, current_object_candidates
    candidates = current_object_candidates
    target_x, target_y = build_search_target_point(current_task_config_id())
    if is_finish_task_context():
        candidates = filter_candidates_in_target_window(
            candidates,
            target_y,
            OBJECT_Y_TOLERANCE_PX,
        )
        if not candidates:
            return build_observation(0, 0, 0, 0), None, None, candidates
        task_name, center_x, bottom_y, area, best_blob = choose_largest_area_candidate(candidates)
    else:
        task_name, center_x, bottom_y, area, best_blob = choose_best_candidate(
            candidates,
            target_x,
            target_y,
        )
    return (
        build_observation(1, center_x, bottom_y, area),
        best_blob,
        task_name,
        candidates,
    )


def build_finish_task_ring_rois(blob, img):
    image_width = int(img.width())
    image_height = int(img.height())
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    expand = int(FINISH_HOOK_RING_EXPAND_PX)
    outer_left = max(0, int(left) - expand)
    outer_top = max(0, int(top) - expand)
    outer_right = min(int(image_width), int(right) + expand)
    outer_bottom = min(int(image_height), int(bottom) + expand)
    rois = []
    if outer_top < int(top):
        rois.append((outer_left, outer_top, outer_right - outer_left, int(top) - outer_top))
    if int(bottom) < outer_bottom:
        rois.append((outer_left, int(bottom), outer_right - outer_left, outer_bottom - int(bottom)))
    if outer_left < int(left):
        rois.append((outer_left, int(top), int(left) - outer_left, int(bottom) - int(top)))
    if int(right) < outer_right:
        rois.append((int(right), int(top), outer_right - int(right), int(bottom) - int(top)))
    ring_area = 0
    valid_rois = []
    for roi in rois:
        _, _, width, height = roi
        if width <= 0 or height <= 0:
            continue
        ring_area += int(width) * int(height)
        valid_rois.append(roi)
    return valid_rois, ring_area


def _count_yellow_pixels_in_roi(img, roi):
    roi_area = int(roi[2]) * int(roi[3])
    if roi_area <= 0:
        return 0
    blobs = img.find_blobs(
        [FINISH_HOOK_YELLOW_THRESHOLD],
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


def build_finish_task_yellow_ratio_percent(img, blob):
    if blob is None:
        return 0.0
    rois, ring_area = build_finish_task_ring_rois(blob, img)
    if ring_area <= 0:
        return 0.0
    yellow_pixels = 0
    for roi in rois:
        yellow_pixels += _count_yellow_pixels_in_roi(img, roi)
    return float(yellow_pixels) * 100.0 / float(ring_area)


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
    image_width = int(img.width())
    image_height = int(img.height())
    draw_x = image_width - 1 - int(target_x)
    draw_y = image_height - 1 - int(target_y)
    img.draw_cross(draw_x, draw_y, color=(255, 255, 0))


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
    if not MASTER_DEBUG_DISPLAY_ENABLED:
        return
    print("[master_v2][%s] %s" % (str(tag), str(text)))


def draw_tracking_state_debug(img):
    predicted_roi = state.track_predicted_roi
    if predicted_roi is not None:
        left, top, right, bottom = predicted_roi
        width = int(right) - int(left)
        height = int(bottom) - int(top)
        if width > 0 and height > 0:
            img.draw_rectangle((int(left), int(top), width, height), color=(0, 255, 255), thickness=1)
    if state.track_predicted_center_x is not None and state.track_predicted_bottom_y is not None:
        draw_protocol_target_point_debug(
            img,
            state.track_predicted_center_x,
            state.track_predicted_bottom_y,
        )
    img.draw_string(
        2,
        62,
        "src=%s conf=%d gap=%d" % (
            tracking_source_debug_name(state.current_detection_source),
            int(state.track_confidence),
            int(state.track_frames_since_yolo),
        ),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )
    img.draw_string(
        2,
        74,
        "fail=%s rf=%d id=%d" % (
            tracking_failure_debug_name(state.track_failure_reason),
            int(state.track_roi_failure_frames),
            int(state.track_object_id),
        ),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )
    img.draw_string(
        2,
        86,
        "fps t=%d y=%d r=%d p=%d fb=%d" % (
            int(state.last_second_total_frames),
            int(state.last_second_yolo_frames),
            int(state.last_second_roi_frames),
            int(state.last_second_predict_frames),
            int(state.last_second_roi_fallbacks),
        ),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )
    img.draw_string(
        2,
        98,
        "roi ok=%d/%d" % (
            int(state.last_second_roi_success_frames),
            int(state.last_second_roi_attempt_frames),
        ),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )


def draw_finish_task_debug(img, blob, yellow_ratio):
    if blob is None:
        img.draw_string(2, 50, "finish ratio=0.0", color=(255, 255, 255), scale=1, mono_space=False)
        return
    rois, _ = build_finish_task_ring_rois(blob, img)
    for roi in rois:
        img.draw_rectangle(roi, color=(255, 255, 0), thickness=1)
    img.draw_string(
        2,
        50,
        "finish ratio=%.1f" % float(yellow_ratio),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )


def draw_return_line_debug(img, line_y, velocity):
    image_width = int(img.width())
    image_height = int(img.height())
    center_x = int(image_width / 2)
    half_width = int(RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX)
    target_y = int(RETURN_GARAGE_LINE_TARGET_Y_PX)
    flip_x = lambda x: image_width - 1 - int(x)
    flip_y = lambda y: image_height - 1 - int(y)
    img.draw_line(
        flip_x(center_x - half_width),
        flip_y(0),
        flip_x(center_x - half_width),
        flip_y(image_height - 1),
        color=(255, 255, 0),
    )
    img.draw_line(
        flip_x(center_x + half_width),
        flip_y(0),
        flip_x(center_x + half_width),
        flip_y(image_height - 1),
        color=(255, 255, 0),
    )
    img.draw_line(
        flip_x(0),
        flip_y(target_y),
        flip_x(image_width - 1),
        flip_y(target_y),
        color=(255, 255, 0),
    )
    if line_y is not None:
        img.draw_line(
            flip_x(0),
            flip_y(int(line_y)),
            flip_x(image_width - 1),
            flip_y(int(line_y)),
            color=(255, 0, 0),
        )
    img.draw_string(
        2,
        2,
        "line_y=%s vy=%.1f" % ("none" if line_y is None else "%.1f" % float(line_y), float(velocity[1])),
        color=(255, 255, 255),
        scale=1,
        mono_space=False,
    )


def draw_search_preview_debug(img, candidates):
    draw_object_candidates_debug(img, candidates)
    if candidates:
        target_x, target_y = build_search_target_point(Task.SEARCH)
        task_name, _, _, _, best_blob = choose_best_candidate(candidates, target_x, target_y)
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
    image_height = float(state.current_image_height)
    scaled_error = err_y * (
        float(MASTER_SEARCH_MAX_VY)
        / abs(float(MASTER_SEARCH_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(MASTER_SEARCH_KP_Y) * time_scale,
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
    image_height = float(state.current_image_height)
    scaled_error = err_y * (
        float(MASTER_ORBIT_MAX_VY)
        / abs(float(MASTER_ORBIT_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(MASTER_ORBIT_KP_Y) * time_scale,
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


def _pixel_matches_threshold(pixel, threshold):
    lab = image.rgb_to_lab(pixel)
    return (
        float(threshold[0]) <= float(lab[0]) <= float(threshold[1])
        and float(threshold[2]) <= float(lab[1]) <= float(threshold[3])
        and float(threshold[4]) <= float(lab[2]) <= float(threshold[5])
    )


def _pixel_matches_any_threshold(pixel, thresholds):
    for threshold in thresholds:
        if _pixel_matches_threshold(pixel, threshold):
            return True
    return False


def _sample_roi_positions(start, end, sample_count):
    start = int(start)
    end = int(end)
    count = max(1, int(sample_count))
    span = max(1, end - start)
    for index in range(count):
        yield start + (span * (index * 2 + 1)) // (count * 2)


def _sample_threshold_match_ratio(img, rois, thresholds):
    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None:
        return None
    samples = 0
    matches = 0
    for left, top, right, bottom in rois:
        if right <= left or bottom <= top:
            continue
        for y in _sample_roi_positions(top, bottom, 3):
            for x in _sample_roi_positions(left, right, 3):
                samples += 1
                if _pixel_matches_any_threshold(get_pixel(int(x), int(y)), thresholds):
                    matches += 1
    if samples <= 0:
        return None
    return float(matches) / float(samples)


def _blob_background_ring_rois(blob):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    expand_x = max(1, int((right - left) / 2))
    expand_y = max(1, int((bottom - top) / 2))
    outer_left = max(0, int(left) - expand_x)
    outer_top = max(0, int(top) - expand_y)
    outer_right = min(int(state.current_image_width), int(right) + expand_x)
    outer_bottom = min(int(state.current_image_height), int(bottom) + expand_y)
    return (
        (outer_left, outer_top, outer_right, int(top)),
        (outer_left, int(bottom), outer_right, outer_bottom),
        (outer_left, int(top), int(left), int(bottom)),
        (int(right), int(top), outer_right, int(bottom)),
    )


def _blob_is_separable_from_background(img, task_name, blob):
    thresholds = object_thresholds_for_task_name(task_name)
    if not thresholds:
        return True
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    inner_ratio = _sample_threshold_match_ratio(img, ((left, top, right, bottom),), thresholds)
    ring_ratio = _sample_threshold_match_ratio(img, _blob_background_ring_rois(blob), thresholds)
    if inner_ratio is None or ring_ratio is None:
        return True
    if inner_ratio <= 0.0:
        return True
    return ring_ratio < max(0.5, inner_ratio * 0.8)


def _return_line_pixel_matches(img, x, y):
    image_width = int(img.width())
    image_height = int(img.height())
    max_x = int(image_width) - 1
    max_y = int(image_height) - 1
    return _pixel_matches_threshold(
        img.get_pixel(max_x - int(x), max_y - int(y)),
        RETURN_GARAGE_LINE_YELLOW_THRESHOLD,
    )


def _return_line_has_horizontal_connected_at(img, x, y, required_connected):
    image_width = int(img.width())
    if not _return_line_pixel_matches(img, x, y):
        return False
    connected = 0
    left = int(x) - 1
    while left >= 0 and _return_line_pixel_matches(img, left, y):
        connected += 1
        if connected >= int(required_connected):
            return True
        left -= 1
    right = int(x) + 1
    max_x = int(image_width) - 1
    while right <= max_x and _return_line_pixel_matches(img, right, y):
        connected += 1
        if connected >= int(required_connected):
            return True
        right += 1
    return False


def _return_line_sample_columns(center_x, half_width):
    yield int(center_x)
    for offset in range(1, int(half_width) + 1):
        yield int(center_x) - offset
        yield int(center_x) + offset


def _return_line_y_on_column(img, x):
    image_height = int(img.height())
    top = None
    bottom = None
    for y in range(0, int(image_height)):
        if not _return_line_pixel_matches(img, x, y):
            continue
        if top is None:
            top = int(y)
        bottom = int(y)
    if top is None or bottom is None:
        return None
    max_thickness = int(RETURN_GARAGE_LINE_MAX_THICKNESS_PX)
    if max_thickness > 0 and int(bottom) - int(top) > max_thickness:
        top = int(bottom) - max_thickness
    return (float(top) + float(bottom)) / 2.0


def build_return_line_y_from_image(img, previous_line_y=None):
    image_width = int(img.width())
    image_height = int(img.height())
    center_x = int(int(image_width) / 2)
    half_width = int(RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX)
    saw_candidate = False
    for x in _return_line_sample_columns(center_x, half_width):
        if x < 0 or x >= int(image_width):
            continue
        line_y = _return_line_y_on_column(img, x)
        if line_y is None:
            continue
        saw_candidate = True
        if _return_line_has_horizontal_connected_at(img, x, int(round(line_y)), RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX):
            return line_y
    if saw_candidate:
        return previous_line_y
    return None


def build_return_line_velocity_from_y(line_y):
    if line_y is None:
        return 0.0, 0.0
    err_y = float(line_y) - float(RETURN_GARAGE_LINE_TARGET_Y_PX)
    return (
        0.0,
        _axis_p_velocity(
            err_y,
            RETURN_GARAGE_LINE_DEADZONE_Y_PX,
            RETURN_GARAGE_LINE_KP_Y,
            RETURN_GARAGE_LINE_MAX_VY,
            RETURN_GARAGE_LINE_MIN_SPEED,
        ),
    )


def build_task_event_value(img, best_blob, task_name=None):
    if is_finish_task_context():
        return build_finish_task_yellow_ratio_percent(img, best_blob)
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
    if task_state == State.TRANSPORT_OBJECT and target == Target.EDGE_LINE and arg == Task.TRANSPORT_FINISH:
        return Event.ARRIVED
    if task_state == State.RETURN_GARAGE_RETREAT and target == Target.EDGE_LINE and arg == Task.RETURN_GARAGE_LINE:
        return Event.RETURN_LINE_ALIGNED
    if task_state == State.RETURN_GARAGE_LINE and target == Target.EDGE_LINE and arg == Task.RETURN_GARAGE_LINE:
        return Event.RETURN_GARAGE_FINISHED
    return None


def required_stable_frames():
    if is_finish_task_context():
        return int(FINISH_HOOK_STABLE_FRAMES)
    return int(OBJECT_STABLE_FRAMES)


def resolve_event_value(observation_value, event_value):
    if is_finish_task_context():
        return int(float(event_value))
    if is_return_line_task_context():
        if current_event_type() == Event.RETURN_GARAGE_FINISHED:
            return 0
        return int(float(observation_value))
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
    state.clear_track()


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


def _accept_finish_task_observation(context_id, observation_value, yellow_ratio, event_type):
    if float(observation_value) <= 0.0:
        state.stable_frame_count = 0
        return
    if not state.finish_contact_seen:
        if float(yellow_ratio) > float(FINISH_HOOK_YELLOW_RATIO_THRESHOLD) * 100.0:
            state.finish_contact_seen = True
        state.stable_frame_count = 0
        return
    if float(yellow_ratio) > 0.0:
        state.stable_frame_count = 0
        return
    state.stable_frame_count += 1
    if state.stable_frame_count >= required_stable_frames():
        create_pending_event(
            context_id,
            event_type,
            resolve_event_value(observation_value, yellow_ratio),
        )


def _return_line_row_has_yellow(img):
    image_width = int(img.width())
    for x in range(0, int(image_width)):
        if _return_line_pixel_matches(img, x, RETURN_LINE_FINISH_ROW_Y_PX):
            return True
    return False


def _return_line_column_has_yellow(img):
    image_height = int(img.height())
    for y in range(0, int(image_height)):
        if _return_line_pixel_matches(img, RETURN_LINE_FINISH_COLUMN_X_PX, y):
            return True
    return False


def _accept_return_line_observation(context_id, observation_value, img, event_type):
    if event_type == Event.RETURN_LINE_ALIGNED:
        if float(observation_value) > 0.0 and float(observation_value) <= float(RETURN_GARAGE_LINE_TARGET_Y_PX):
            state.stable_frame_count += 1
            if state.stable_frame_count >= required_stable_frames():
                create_pending_event(
                    context_id,
                    event_type,
                    resolve_event_value(observation_value, None),
                )
            return
        state.stable_frame_count = 0
        return

    state.stable_frame_count = 0
    if _return_line_row_has_yellow(img) and not _return_line_column_has_yellow(img):
        state.return_line_finish_missing_count += 1
    else:
        state.return_line_finish_missing_count = 0
        return
    if state.return_line_finish_missing_count >= int(RETURN_LINE_MISSING_FINISH_FRAMES):
        create_pending_event(context_id, event_type, 0)


def accept_observation(observation, img, event_value=None):
    current_task = state.current_task
    if current_task is None:
        return
    if state.current_detection_source == "predict":
        state.stable_frame_count = 0
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
    if is_finish_task_context():
        _accept_finish_task_observation(context_id, observation_value, event_value, event_type)
        return
    if is_return_line_task_context():
        _accept_return_line_observation(
            context_id,
            observation_value,
            img,
            event_type,
        )
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
            state.finish_contact_seen = False
            state.last_return_line_y = None
            state.return_line_finish_missing_count = 0
            state.clear_track()
        return format_ack_frame(packet["reliable_seq"])

    packet = parse_event_ack_packet(frame_bytes)
    pending_event = state.pending_event
    if packet is not None and pending_event is not None:
        if int(packet["reliable_seq"]) == int(pending_event["reliable_seq"]):
            state.pending_event = None
            state.pending_event_last_sent_ms = None
    return None


def process_uart_input(rx_buffer):
    uart_device = state.uart_device
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
            write_reliable_line(reply)
    return rx_buffer


def _process_return_line_frame(img):
    line_y = build_return_line_y_from_image(img, state.last_return_line_y)
    if line_y is not None:
        state.last_return_line_y = float(line_y)
    velocity = build_return_line_velocity_from_y(line_y)
    write_data_line(format_search_velocity_frame(*velocity))
    accept_observation(build_return_line_observation(line_y), img)
    if MASTER_DEBUG_DISPLAY_ENABLED:
        draw_return_line_debug(img, line_y, velocity)
        img.flush()


def process_task_frame(img):
    if state.pending_event is not None:
        event_frame = next_event_frame()
        if event_frame is not None:
            write_reliable_line(event_frame)
        if not is_return_line_task_context():
            return
    if state.current_task is None:
        if MASTER_DEBUG_DISPLAY_ENABLED:
            observation, best_blob, task_name, _candidates = build_observation_and_candidates()
            debug_log(
                "preview",
                "src=%s cand=%d best=%s conf=%d fail=%s" % (
                    tracking_source_debug_name(state.current_detection_source).lower(),
                    len(_candidates),
                    task_name if task_name is not None else "none",
                    int(state.track_confidence),
                    tracking_failure_debug_name(state.track_failure_reason),
                ),
            )
            if best_blob is not None and state.current_detection_source != "predict":
                remember_object_tracking(
                    task_name,
                    best_blob,
                    best_blob.cx(),
                    observation[2] + build_search_target_point(current_task_config_id())[1],
                    observation[3],
                    state.current_detection_source,
                )
            elif state.current_detection_source == "yolo":
                state.clear_track()
            draw_object_candidates_debug(img, _candidates)
            if best_blob is not None and task_name is not None:
                draw_selected_candidate_debug(img, task_name, best_blob)
            target_x, target_y = build_search_target_point(current_task_config_id())
            draw_protocol_target_point_debug(img, target_x, target_y)
            draw_tracking_state_debug(img)
            img.flush()
        return
    if is_return_line_task_context():
        _process_return_line_frame(img)
        return
    observation, best_blob, task_name, _candidates = build_observation_and_candidates()
    debug_log(
        "object",
        "src=%s cand=%d best=%s conf=%d fail=%s" % (
            tracking_source_debug_name(state.current_detection_source).lower(),
            len(_candidates),
            task_name if task_name is not None else "none",
            int(state.track_confidence),
            tracking_failure_debug_name(state.track_failure_reason),
        ),
    )
    if best_blob is not None and state.current_detection_source != "predict":
        remember_object_tracking(
            task_name,
            best_blob,
            best_blob.cx(),
            observation[2] + build_search_target_point(current_task_config_id())[1],
            observation[3],
            state.current_detection_source,
        )
    elif state.current_detection_source == "yolo":
        state.clear_track()
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
        draw_tracking_state_debug(img)
        if is_finish_task_context():
            draw_finish_task_debug(img, best_blob, event_value)
        img.flush()


def init_uart():
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width(), sensor.height()


def run():
    reset_runtime_state()
    state.uart_device = init_uart()
    init_sensor()
    state.yolo_net = tf.load(YOLO_MODEL_PATH)
    debug_log("boot", "debug=%d yolo=%d" % (1 if MASTER_DEBUG_DISPLAY_ENABLED else 0, 1))
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
        if state.pending_event is not None:
            state.current_yolo_candidates = ()
            state.current_object_candidates = ()
            debug_log("skip", "reason=pending_event")
        elif is_return_line_task_context():
            state.current_yolo_candidates = ()
            state.current_object_candidates = ()
            debug_log("skip", "reason=return_line")
        else:
            if state.current_task is not None or MASTER_DEBUG_DISPLAY_ENABLED:
                need_yolo = should_run_yolo_for_current_frame()
                if not need_yolo:
                    state.current_yolo_candidates = ()
                    state.current_object_candidates = tuple(build_object_candidates(img, ()))
                    need_yolo = state.current_detection_source == "yolo"
                if need_yolo:
                    raw_yolo_candidates = tuple(yolo_detect(img))
                    if raw_yolo_candidates:
                        state.current_detection_source = "yolo"
                    state.current_yolo_candidates = raw_yolo_candidates
                    state.current_object_candidates = tuple(
                        build_object_candidates(img, raw_yolo_candidates)
                    )
            else:
                state.current_yolo_candidates = ()
                state.current_object_candidates = ()
        try:
            process_task_frame(img)
        finally:
            state.record_frame_source(state.current_detection_source)
            gc.collect()


if __name__ == "__main__":
    run()
