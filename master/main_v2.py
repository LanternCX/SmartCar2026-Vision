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

# 高频速度数据流使用的模式编号.
MODE_UDP = 0x01
# 低频可靠同步与事件回报使用的模式编号.
MODE_TCP = 0x02
# 可靠确认帧使用的模式编号.
MODE_ACK = 0x03
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
YOLO_MIN_SCORE = 0.90
# YOLO 标签编号到任务名的稳定映射.
YOLO_LABELS = ("tennis", "red", "blue", "brown", "white")
# 调试模式打开后在屏幕上显示识别框和目标点.
MASTER_DEBUG_DISPLAY_ENABLED = False

# OpenART 下发主车本地视觉速度的 topic 编号.
TOPIC_LOCAL_VISION_VELOCITY = 0x01
# RT1021 下发主车本地视觉任务同步的 topic 编号.
TOPIC_MASTER_VISION_TASK_SYNC = 0x10
# OpenART 回报主车视觉可靠事件的 topic 编号.
TOPIC_MASTER_VISION_EVENT_REPORT = 0x12

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

# 主车搜索状态编号.
STATE_SEARCH_OBJECT = 1
# 主车绕行状态编号.
STATE_ORBITING = 2
# 主车搬运状态编号.
STATE_TRANSPORT_OBJECT = 4
# 主车回库后退找黄线状态编号.
STATE_RETURN_GARAGE_RETREAT = 6
# 主车回库黄线平移状态编号.
STATE_RETURN_GARAGE_LINE = 7

# 物体目标类型编号.
TARGET_OBJECT = 1
# 边线目标类型编号.
TARGET_EDGE_LINE = 3

# 主车搜索任务配置编号.
MASTER_SEARCH_TASK_CONFIG_ID = 1
# 主车搬运入口对正任务配置编号.
MASTER_TRANSPORT_TASK_CONFIG_ID = 2
# 主车搬运收尾任务配置编号.
MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID = 3
# 主车绕行修正任务配置编号.
MASTER_ORBIT_TASK_CONFIG_ID = 4
# 主车回库黄线任务配置编号.
MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID = 5

# 发现目标事件编号.
EVENT_TARGET_FOUND = 6
# 搬运入口对正完成事件编号.
EVENT_ALIGNED = 7
# 搬运收尾到位事件编号.
EVENT_ARRIVED = 8
# 回库后退对正黄线完成事件编号.
EVENT_RETURN_LINE_ALIGNED = 10
# 回库完成事件编号.
EVENT_RETURN_GARAGE_FINISHED = 12

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


# 当前生效的主车本地视觉任务.
CURRENT_TASK = None
# 最近一次应用的任务上下文编号.
LAST_TASK_CONTEXT_ID = None
# 当前任务稳定命中已累计的连续帧数.
STABLE_FRAME_COUNT = 0
# 下一次可靠事件回报将使用的序号.
NEXT_EVENT_SEQ = 1
# 当前等待 ACK 的可靠事件内容.
PENDING_EVENT = None
# 当前等待 ACK 的可靠事件上次发送时间, 单位为毫秒.
PENDING_EVENT_LAST_SENT_MS = None
# 当前上下文已经成功回报过事件时记录的上下文编号.
LAST_EVENT_CONTEXT_ID = None
# 搬运收尾是否已经观察到黄色接触.
FINISH_CONTACT_SEEN = False
# 最近一次有效回库黄线中心 Y.
LAST_RETURN_LINE_Y = None
# 回库完成判定已累计的连续满足帧数.
RETURN_LINE_FINISH_MISSING_COUNT = 0
# 串口输入残片缓冲区.
RX_BUFFER = b""
# 启动后复用的 YOLO 网络对象.
YOLO_NET = None
# 主车视觉脚本运行时唯一使用的串口对象.
UART_DEVICE = None
# 当前帧复用的 YOLO 候选缓存.
CURRENT_YOLO_CANDIDATES = ()


def reset_runtime_state(next_event_seq=1):
    """重置主车视觉运行态."""

    global CURRENT_TASK
    global LAST_TASK_CONTEXT_ID
    global STABLE_FRAME_COUNT
    global NEXT_EVENT_SEQ
    global PENDING_EVENT
    global PENDING_EVENT_LAST_SENT_MS
    global LAST_EVENT_CONTEXT_ID
    global FINISH_CONTACT_SEEN
    global LAST_RETURN_LINE_Y
    global RETURN_LINE_FINISH_MISSING_COUNT
    global RX_BUFFER
    global YOLO_NET
    global UART_DEVICE
    global CURRENT_YOLO_CANDIDATES

    CURRENT_TASK = None
    LAST_TASK_CONTEXT_ID = None
    STABLE_FRAME_COUNT = 0
    NEXT_EVENT_SEQ = int(next_event_seq) % SEQ_RING_SIZE
    PENDING_EVENT = None
    PENDING_EVENT_LAST_SENT_MS = None
    LAST_EVENT_CONTEXT_ID = None
    FINISH_CONTACT_SEEN = False
    LAST_RETURN_LINE_Y = None
    RETURN_LINE_FINISH_MISSING_COUNT = 0
    RX_BUFFER = b""
    YOLO_NET = None
    UART_DEVICE = None
    CURRENT_YOLO_CANDIDATES = ()


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
    if frame["mode"] != MODE_TCP or frame["topic"] != TOPIC_MASTER_VISION_TASK_SYNC:
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
    if frame["mode"] != MODE_ACK or frame["topic"] != TOPIC_MASTER_VISION_EVENT_REPORT:
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
    return encode_frame(MODE_ACK, TOPIC_MASTER_VISION_TASK_SYNC, reliable_seq, b"")


def format_search_velocity_frame(vx, vy):
    return encode_frame(
        MODE_UDP,
        TOPIC_LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_event_frame(reliable_seq, context_id, event, value):
    return encode_frame(
        MODE_TCP,
        TOPIC_MASTER_VISION_EVENT_REPORT,
        reliable_seq,
        encode_master_vision_event_report_body(context_id, event, value),
    )


def _write_all(frame_bytes):
    if not isinstance(frame_bytes, bytes):
        frame_bytes = bytes(frame_bytes)
    remaining = frame_bytes
    while remaining:
        written = UART_DEVICE.write(remaining)
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
        if frame["mode"] == MODE_TCP and frame["topic"] == TOPIC_MASTER_VISION_TASK_SYNC:
            return index
        if frame["mode"] == MODE_ACK and frame["topic"] == TOPIC_MASTER_VISION_EVENT_REPORT:
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


def normalize_bbox_for_protocol(left, top, right, bottom, image_height):
    normalized_top = image_height - bottom
    normalized_bottom = image_height - top
    return left, normalized_top, right, normalized_bottom


def blob_area(blob):
    return float(blob.area())


def object_task_id(task_name):
    for index, task in enumerate(OBJECT_TASKS, 1):
        if task[0] == task_name:
            return index
    return 0


def label_name(label):
    label = int(label)
    if 0 <= label < len(YOLO_LABELS):
        return YOLO_LABELS[label]
    return "unknown"


def yolo_detect(img):
    net = YOLO_NET
    if net is None:
        raise RuntimeError("YOLO_NET not loaded")
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
        _, _, _, protocol_bottom = normalize_bbox_for_protocol(
            left,
            top,
            right,
            bottom,
            image_height,
        )
        candidates.append((task_name, blob.cx(), protocol_bottom, blob.area(), blob))
    return candidates

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
    if int(config_id) == int(MASTER_ORBIT_TASK_CONFIG_ID):
        return float(MASTER_ORBIT_TARGET_X_PX), float(MASTER_ORBIT_TARGET_Y_PX)
    if int(config_id) in (
        int(MASTER_TRANSPORT_TASK_CONFIG_ID),
        int(MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID),
    ):
        return target_x, float(MASTER_TRANSPORT_TARGET_Y_PX)
    return target_x, float(MASTER_SEARCH_TARGET_Y_PX)


def current_task_config_id():
    if CURRENT_TASK is None:
        return MASTER_SEARCH_TASK_CONFIG_ID
    return int(CURRENT_TASK["arg"])


def is_finish_task_context():
    return (
        CURRENT_TASK is not None
        and int(CURRENT_TASK["state"]) == int(STATE_TRANSPORT_OBJECT)
        and int(CURRENT_TASK["target"]) == int(TARGET_EDGE_LINE)
        and int(CURRENT_TASK["arg"]) == int(MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID)
    )


def is_orbit_task_context():
    return (
        CURRENT_TASK is not None
        and int(CURRENT_TASK["state"]) == int(STATE_ORBITING)
        and int(CURRENT_TASK["target"]) == int(TARGET_OBJECT)
        and int(CURRENT_TASK["arg"]) == int(MASTER_ORBIT_TASK_CONFIG_ID)
    )


def is_return_line_task_context():
    return (
        CURRENT_TASK is not None
        and int(CURRENT_TASK["target"]) == int(TARGET_EDGE_LINE)
        and int(CURRENT_TASK["arg"]) == int(MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID)
        and int(CURRENT_TASK["state"]) in (
            int(STATE_RETURN_GARAGE_RETREAT),
            int(STATE_RETURN_GARAGE_LINE),
        )
    )


def build_observation(valid, center_x, bottom_y, area):
    context_id = 0
    if CURRENT_TASK is not None:
        context_id = int(CURRENT_TASK["context_id"])
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
    if CURRENT_TASK is not None:
        context_id = int(CURRENT_TASK["context_id"])
    if line_y is None:
        return context_id, 0.0, 0.0, 0.0
    return (
        context_id,
        0.0,
        float(line_y) - float(RETURN_GARAGE_LINE_TARGET_Y_PX),
        float(line_y),
    )


def build_observation_and_candidates():
    if not CURRENT_YOLO_CANDIDATES:
        return build_observation(0, 0, 0, 0), None, None, CURRENT_YOLO_CANDIDATES
    candidates = CURRENT_YOLO_CANDIDATES
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


def build_finish_task_ring_rois(blob, image_width, image_height):
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
    image_width = int(img.width())
    image_height = int(img.height())
    rois, ring_area = build_finish_task_ring_rois(blob, image_width, image_height)
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


def draw_finish_task_debug(img, blob, yellow_ratio):
    if blob is None:
        img.draw_string(2, 50, "finish ratio=0.0", color=(255, 255, 255), scale=1, mono_space=False)
        return
    rois, _ = build_finish_task_ring_rois(blob, int(img.width()), int(img.height()))
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
        target_x, target_y = build_search_target_point(MASTER_SEARCH_TASK_CONFIG_ID)
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


def _axis_p_velocity(error, deadzone, kp, limit, min_speed):
    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    if float(kp) == 0.0:
        return 0.0
    return _apply_min_speed(error * float(kp), limit, min_speed)


def _build_search_y_velocity(err_y, image_height):
    err_y = float(err_y)
    if abs(err_y) <= float(MASTER_SEARCH_DEADZONE_Y_PX):
        return 0.0
    scaled_error = err_y * (
        float(MASTER_SEARCH_MAX_VY)
        / abs(float(MASTER_SEARCH_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(MASTER_SEARCH_KP_Y),
        MASTER_SEARCH_MAX_VY,
        MASTER_SEARCH_MIN_SPEED,
    )


def build_search_velocity_from_observation(observation, image_height):
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
        _build_search_y_velocity(y, image_height),
    )


def _build_orbit_y_velocity(err_y, image_height):
    err_y = float(err_y)
    if abs(err_y) <= float(MASTER_ORBIT_DEADZONE_Y_PX):
        return 0.0
    if float(MASTER_ORBIT_KP_Y) == 0.0:
        return 0.0
    scaled_error = err_y * (
        float(MASTER_ORBIT_MAX_VY)
        / abs(float(MASTER_ORBIT_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(MASTER_ORBIT_KP_Y),
        MASTER_ORBIT_MAX_VY,
        MASTER_ORBIT_MIN_SPEED,
    )


def build_orbit_correction_velocity_from_observation(observation, image_height):
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
        _build_orbit_y_velocity(y, image_height),
    )


def _pixel_matches_threshold(pixel, threshold):
    lab = image.rgb_to_lab(pixel)
    return (
        float(threshold[0]) <= float(lab[0]) <= float(threshold[1])
        and float(threshold[2]) <= float(lab[1]) <= float(threshold[3])
        and float(threshold[4]) <= float(lab[2]) <= float(threshold[5])
    )


def _return_line_pixel_matches(img, x, y, image_width, image_height):
    max_x = int(image_width) - 1
    max_y = int(image_height) - 1
    return _pixel_matches_threshold(
        img.get_pixel(max_x - int(x), max_y - int(y)),
        RETURN_GARAGE_LINE_YELLOW_THRESHOLD,
    )


def _return_line_has_horizontal_connected_at(img, x, y, image_width, image_height, required_connected):
    if not _return_line_pixel_matches(img, x, y, image_width, image_height):
        return False
    connected = 0
    left = int(x) - 1
    while left >= 0 and _return_line_pixel_matches(img, left, y, image_width, image_height):
        connected += 1
        if connected >= int(required_connected):
            return True
        left -= 1
    right = int(x) + 1
    max_x = int(image_width) - 1
    while right <= max_x and _return_line_pixel_matches(img, right, y, image_width, image_height):
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


def _return_line_y_on_column(img, x, image_width, image_height):
    top = None
    bottom = None
    for y in range(0, int(image_height)):
        if not _return_line_pixel_matches(img, x, y, image_width, image_height):
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


def build_return_line_y_from_image(img, image_width, image_height, previous_line_y=None):
    center_x = int(int(image_width) / 2)
    half_width = int(RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX)
    saw_candidate = False
    for x in _return_line_sample_columns(center_x, half_width):
        if x < 0 or x >= int(image_width):
            continue
        line_y = _return_line_y_on_column(img, x, image_width, image_height)
        if line_y is None:
            continue
        saw_candidate = True
        if _return_line_has_horizontal_connected_at(
            img,
            x,
            int(round(line_y)),
            image_width,
            image_height,
            RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX,
        ):
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
    if (
        CURRENT_TASK is not None
        and int(CURRENT_TASK["state"]) == int(STATE_SEARCH_OBJECT)
        and int(CURRENT_TASK["target"]) == int(TARGET_OBJECT)
        and int(CURRENT_TASK["arg"]) == int(MASTER_SEARCH_TASK_CONFIG_ID)
        and task_name is not None
    ):
        return object_task_id(task_name)
    return None


def current_event_type():
    if CURRENT_TASK is None:
        return None
    state = int(CURRENT_TASK["state"])
    target = int(CURRENT_TASK["target"])
    arg = int(CURRENT_TASK["arg"])
    if state == STATE_SEARCH_OBJECT and target == TARGET_OBJECT and arg == MASTER_SEARCH_TASK_CONFIG_ID:
        return EVENT_TARGET_FOUND
    if state == STATE_SEARCH_OBJECT and target == TARGET_OBJECT and arg == MASTER_TRANSPORT_TASK_CONFIG_ID:
        return EVENT_ALIGNED
    if state == STATE_TRANSPORT_OBJECT and target == TARGET_EDGE_LINE and arg == MASTER_TRANSPORT_FINISH_TASK_CONFIG_ID:
        return EVENT_ARRIVED
    if state == STATE_RETURN_GARAGE_RETREAT and target == TARGET_EDGE_LINE and arg == MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID:
        return EVENT_RETURN_LINE_ALIGNED
    if state == STATE_RETURN_GARAGE_LINE and target == TARGET_EDGE_LINE and arg == MASTER_RETURN_GARAGE_LINE_TASK_CONFIG_ID:
        return EVENT_RETURN_GARAGE_FINISHED
    return None


def required_stable_frames():
    if is_finish_task_context():
        return int(FINISH_HOOK_STABLE_FRAMES)
    return int(OBJECT_STABLE_FRAMES)


def resolve_event_value(observation_value, event_value):
    if is_finish_task_context():
        return int(float(event_value))
    if is_return_line_task_context():
        if current_event_type() == EVENT_RETURN_GARAGE_FINISHED:
            return 0
        return int(float(observation_value))
    if (
        CURRENT_TASK is not None
        and int(CURRENT_TASK["state"]) == int(STATE_SEARCH_OBJECT)
        and int(CURRENT_TASK["target"]) == int(TARGET_OBJECT)
        and int(CURRENT_TASK["arg"]) == int(MASTER_SEARCH_TASK_CONFIG_ID)
        and event_value is not None
    ):
        return int(event_value)
    return int(float(observation_value))


def allocate_event_seq():
    global NEXT_EVENT_SEQ
    reliable_seq = NEXT_EVENT_SEQ
    NEXT_EVENT_SEQ = (NEXT_EVENT_SEQ + 1) % SEQ_RING_SIZE
    return reliable_seq


def create_pending_event(context_id, event, value):
    global PENDING_EVENT
    global PENDING_EVENT_LAST_SENT_MS
    global LAST_EVENT_CONTEXT_ID

    PENDING_EVENT = {
        "reliable_seq": allocate_event_seq(),
        "context_id": int(context_id),
        "event": int(event),
        "value": int(value),
    }
    PENDING_EVENT_LAST_SENT_MS = None
    LAST_EVENT_CONTEXT_ID = int(context_id)


def next_event_frame():
    global PENDING_EVENT_LAST_SENT_MS

    if PENDING_EVENT is None:
        return None
    now_ms = default_now_ms()
    if not should_resend(now_ms, PENDING_EVENT_LAST_SENT_MS, RELIABLE_RESEND_INTERVAL_MS):
        return None
    PENDING_EVENT_LAST_SENT_MS = now_ms
    return format_event_frame(
        PENDING_EVENT["reliable_seq"],
        PENDING_EVENT["context_id"],
        PENDING_EVENT["event"],
        PENDING_EVENT["value"],
    )


def _accept_finish_task_observation(context_id, observation_value, yellow_ratio, event_type):
    global STABLE_FRAME_COUNT
    global FINISH_CONTACT_SEEN

    if float(observation_value) <= 0.0:
        STABLE_FRAME_COUNT = 0
        return
    if not FINISH_CONTACT_SEEN:
        if float(yellow_ratio) > float(FINISH_HOOK_YELLOW_RATIO_THRESHOLD) * 100.0:
            FINISH_CONTACT_SEEN = True
        STABLE_FRAME_COUNT = 0
        return
    if float(yellow_ratio) > 0.0:
        STABLE_FRAME_COUNT = 0
        return
    STABLE_FRAME_COUNT += 1
    if STABLE_FRAME_COUNT >= required_stable_frames():
        create_pending_event(
            context_id,
            event_type,
            resolve_event_value(observation_value, yellow_ratio),
        )


def _return_line_row_has_yellow(img, image_width, image_height):
    for x in range(0, int(image_width)):
        if _return_line_pixel_matches(img, x, RETURN_LINE_FINISH_ROW_Y_PX, image_width, image_height):
            return True
    return False


def _return_line_column_has_yellow(img, image_width, image_height):
    for y in range(0, int(image_height)):
        if _return_line_pixel_matches(img, RETURN_LINE_FINISH_COLUMN_X_PX, y, image_width, image_height):
            return True
    return False


def _accept_return_line_observation(context_id, observation_value, img, event_type):
    global STABLE_FRAME_COUNT
    global RETURN_LINE_FINISH_MISSING_COUNT
    image_width = int(img.width())
    image_height = int(img.height())

    if event_type == EVENT_RETURN_LINE_ALIGNED:
        if float(observation_value) > 0.0 and float(observation_value) <= float(RETURN_GARAGE_LINE_TARGET_Y_PX):
            STABLE_FRAME_COUNT += 1
            if STABLE_FRAME_COUNT >= required_stable_frames():
                create_pending_event(
                    context_id,
                    event_type,
                    resolve_event_value(observation_value, None),
                )
            return
        STABLE_FRAME_COUNT = 0
        return

    STABLE_FRAME_COUNT = 0
    if _return_line_row_has_yellow(img, image_width, image_height) and not _return_line_column_has_yellow(
        img,
        image_width,
        image_height,
    ):
        RETURN_LINE_FINISH_MISSING_COUNT += 1
    else:
        RETURN_LINE_FINISH_MISSING_COUNT = 0
        return
    if RETURN_LINE_FINISH_MISSING_COUNT >= int(RETURN_LINE_MISSING_FINISH_FRAMES):
        create_pending_event(context_id, event_type, 0)


def accept_observation(observation, img, event_value=None):
    global STABLE_FRAME_COUNT

    if CURRENT_TASK is None:
        return
    context_id = int(CURRENT_TASK["context_id"])
    observed_context_id, error_x, error_y, observation_value = observation
    if int(observed_context_id) != context_id:
        return
    event_type = current_event_type()
    if event_type is None:
        STABLE_FRAME_COUNT = 0
        return
    if PENDING_EVENT is not None or LAST_EVENT_CONTEXT_ID == context_id:
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
        STABLE_FRAME_COUNT += 1
    else:
        STABLE_FRAME_COUNT = 0
        return
    if STABLE_FRAME_COUNT >= required_stable_frames():
        create_pending_event(
            context_id,
            event_type,
            resolve_event_value(observation_value, event_value),
        )


def handle_control_frame(frame_bytes):
    global CURRENT_TASK
    global LAST_TASK_CONTEXT_ID
    global STABLE_FRAME_COUNT
    global FINISH_CONTACT_SEEN
    global LAST_RETURN_LINE_Y
    global RETURN_LINE_FINISH_MISSING_COUNT
    global PENDING_EVENT
    global PENDING_EVENT_LAST_SENT_MS

    packet = parse_task_sync_packet(frame_bytes)
    if packet is not None:
        context_id = int(packet["context_id"])
        if is_newer_seq(context_id, LAST_TASK_CONTEXT_ID):
            CURRENT_TASK = {
                "context_id": context_id,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            LAST_TASK_CONTEXT_ID = context_id
            STABLE_FRAME_COUNT = 0
            FINISH_CONTACT_SEEN = False
            LAST_RETURN_LINE_Y = None
            RETURN_LINE_FINISH_MISSING_COUNT = 0
        return format_ack_frame(packet["reliable_seq"])

    packet = parse_event_ack_packet(frame_bytes)
    if packet is not None and PENDING_EVENT is not None:
        if int(packet["reliable_seq"]) == int(PENDING_EVENT["reliable_seq"]):
            PENDING_EVENT = None
            PENDING_EVENT_LAST_SENT_MS = None
    return None


def process_uart_input(rx_buffer):
    size = UART_DEVICE.any()
    if not size:
        return rx_buffer
    data = UART_DEVICE.read(size)
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
    global LAST_RETURN_LINE_Y
    image_width = int(img.width())
    image_height = int(img.height())
    line_y = build_return_line_y_from_image(
        img,
        image_width,
        image_height,
        LAST_RETURN_LINE_Y,
    )
    if line_y is not None:
        LAST_RETURN_LINE_Y = float(line_y)
    velocity = build_return_line_velocity_from_y(line_y)
    write_data_line(format_search_velocity_frame(*velocity))
    accept_observation(
        build_return_line_observation(line_y),
        img,
    )
    if MASTER_DEBUG_DISPLAY_ENABLED:
        draw_return_line_debug(img, line_y, velocity)
        img.flush()


def process_task_frame(img):
    if PENDING_EVENT is not None:
        event_frame = next_event_frame()
        if event_frame is not None:
            write_reliable_line(event_frame)
        if not is_return_line_task_context():
            return
    if CURRENT_TASK is None:
        if MASTER_DEBUG_DISPLAY_ENABLED:
            draw_search_preview_debug(img, CURRENT_YOLO_CANDIDATES)
        return
    if is_return_line_task_context():
        _process_return_line_frame(img)
        return
    image_height = int(img.height())
    observation, best_blob, task_name, _candidates = build_observation_and_candidates()
    if is_orbit_task_context():
        velocity = build_orbit_correction_velocity_from_observation(observation, image_height)
    else:
        velocity = build_search_velocity_from_observation(observation, image_height)
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
    global YOLO_NET
    global RX_BUFFER
    global UART_DEVICE
    global CURRENT_YOLO_CANDIDATES

    reset_runtime_state()
    UART_DEVICE = init_uart()
    init_sensor()
    YOLO_NET = tf.load(YOLO_MODEL_PATH)

    while True:
        RX_BUFFER = process_uart_input(RX_BUFFER)
        img = sensor.snapshot()
        img.lens_corr(strength=2.8, zoom=1.0)
        CURRENT_YOLO_CANDIDATES = tuple(yolo_detect(img))
        process_task_frame(img)
        gc.collect()


if __name__ == "__main__":
    run()
