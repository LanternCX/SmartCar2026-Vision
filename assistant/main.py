"""! @file main.py
@brief OpenART 辅车视觉入口
@details 负责跟随色标与找物体两种模式的图像处理, 输出速度短包并维护本地可靠事件
"""

import time

try:
    import image as omv_image
except ImportError:
    omv_image = None

try:
    import sensor
except ImportError:
    sensor = None

try:
    import tf
except ImportError:
    tf = None

try:
    from machine import UART
except ImportError:
    UART = None


# OpenART 与 RT1021 通信使用的串口编号。
UART_ID = 2
# 串口波特率，需要与车端 UART6 保持一致。
UART_BAUDRATE = 115200
# 摄像头固定曝光时间，单位为微秒。
EXP_TIME_US = 500
# 可靠事件默认重发间隔，单位为毫秒。
RELIABLE_RESEND_INTERVAL_MS = 100

# 固定帧模式编号分组。
class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


# 辅车视觉协议 topic 编号分组。
class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    ASSISTANT_VISION_TASK_SYNC = 0x11
    ASSISTANT_VISION_EVENT_REPORT = 0x13

# 辅车运行模式分组。
class RunMode:
    FOLLOW = "follow"
    APPROACH_OBJECT = "approach_object"
    ORBIT_OBJECT = "orbit_object"
    RETURN_LINE = "return_line"


# 辅车找物体状态编号分组。
class State:
    APPROACH_OBJECT = 2
    ORBIT = 3
    TRANSPORT_OBJECT = 4
    RETURN_FOLLOW = 6


# 辅车目标编号分组。
class Target:
    NONE = 0
    OBJECT = 1


# 辅车任务编号分组。
class Task:
    SEARCH = 1
    TRANSPORT = 2
    ORBIT = 3
    RETURN_GARAGE_LINE = 5


# 辅车事件编号分组。
class Event:
    TARGET_FOUND = 6
    ALIGNED = 7
    RETURN_GARAGE_FINISHED = 12


# 可靠序号使用 0..255 环形空间。
SEQ_RING_SIZE = 256
# 判断序号新旧使用的半环长度。
SEQ_HALF_RING = 128
# 固定帧 body 槽位长度。
FRAME_BODY_SIZE = 10
FRAME_HEAD = 0xA5
# 固定帧总长度。
FRAME_SIZE = 15
# 物体识别开关。False 使用色块阈值，True 使用 YOLO。
OBJECT_DETECTION_USE_YOLO = False
# YOLO 模型文件路径，对应部署到 OpenART SD 卡根目录的模型文件。
YOLO_MODEL_PATH = "/sd/yolo.tflite"
# YOLO 检测前对图像做缩放复制，与模型验证脚本保持一致。
YOLO_IMAGE_COPY_SCALE = 0.75
# 物体识别最低置信度。
YOLO_MIN_SCORE = 0.90
# YOLO 标签编号映射。
YOLO_LABELS = ("tennis", "red", "blue", "brown", "white")

# ChromaForge 导出的色块合并间距。
OBJECT_BLOB_MERGE_MARGIN = 0
# ChromaForge 导出的最小识别色块面积。
OBJECT_BLOB_PIXELS_THRESHOLD = 200
# ChromaForge 导出的最小识别目标面积。
OBJECT_BLOB_AREA_THRESHOLD = 200
# 跟随模式使用的色标阈值。
FOLLOW_TASKS = (("marker", (37, 57, 64, 95, -64, 20)),)
# 找物体模式使用的红色目标阈值，与主车保持一致。
OBJECT_TASKS = (
    ('red', ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),
)
# 回库黄线使用的黄色阈值，与主车回库黄线保持一致。
RETURN_LINE_YELLOW_THRESHOLD = (58, 87, -32, -12, 64, 84)

# 跟随控制使用的横向死区，单位为像素。
FOLLOW_X_DEADZONE_PX = 5.0
# 跟随控制使用的纵向目标尺度量，单位为像素。
FOLLOW_TARGET_Y = 45.0
# 跟随控制使用的纵向死区，单位为像素。
FOLLOW_Y_DEADZONE_PX = 8.0
# 跟随控制使用的横向速度修正量增益。
FOLLOW_CONTROL_KP_X = 0.04
# 跟随控制使用的纵向速度修正量增益。
FOLLOW_CONTROL_KP_Y = -0.10
# 跟随控制误差超出死区后的最小有效速度量。
FOLLOW_CONTROL_MIN_SPEED = 0
# 跟随控制纵向速度修正量上限。
FOLLOW_CONTROL_MAX_Y = 5

# 找物体模式无目标时的横向搜索速度。
OBJECT_MISSING_SEARCH_VX = 0.0
# 找物体模式无目标时的前向搜索速度。
OBJECT_MISSING_SEARCH_VY = 0.0
# 找物体模式横向速度 P 环增益。
OBJECT_APPROACH_KP_X = 0.05
# 找物体模式纵向速度 P 环增益。
OBJECT_APPROACH_KP_Y = -0.2
# 找物体模式误差超出死区后的最小有效速度量。
OBJECT_APPROACH_MIN_SPEED = 2
# 找物体模式横向误差死区，单位为像素。
OBJECT_APPROACH_DEADZONE_X_PX = 15.0
# 找物体模式纵向误差死区，单位为像素。
OBJECT_APPROACH_DEADZONE_Y_PX = 8.0
# 找物体模式横向速度限幅。
OBJECT_APPROACH_MAX_VX = 5.0
# 找物体模式纵向速度限幅。
OBJECT_APPROACH_MAX_VY = 5.0
# 绕行修正横向速度 P 环增益。
OBJECT_ORBIT_KP_X = 0.05
# 绕行修正纵向速度 P 环增益。
OBJECT_ORBIT_KP_Y = -0.15
# 绕行修正误差超出死区后的最小有效速度量。
OBJECT_ORBIT_MIN_SPEED = 0
# 绕行修正横向误差死区，单位为像素。
OBJECT_ORBIT_DEADZONE_X_PX = 15.0
# 绕行修正纵向误差死区，单位为像素。
OBJECT_ORBIT_DEADZONE_Y_PX = 8.0
# 绕行修正横向速度限幅。
OBJECT_ORBIT_MAX_VX = 5.0
# 绕行修正纵向速度限幅。
OBJECT_ORBIT_MAX_VY = 5.0
# 找物体目标点横向像素坐标。当前图像为 QVGA 320x240, 默认中线 x=160; 若修改图像宽度请同步调整。
OBJECT_APPROACH_TARGET_X_PX = 160.0
# 找物体目标点纵向像素坐标。当前图像为 QVGA 320x240, 默认底边 y=240; 若修改图像高度请同步调整。
OBJECT_APPROACH_TARGET_Y_PX = 200.0
# 绕行修正目标点横向像素坐标。
OBJECT_ORBIT_TARGET_X_PX = 160.0
# 绕行修正目标点纵向像素坐标。
OBJECT_ORBIT_TARGET_Y_PX = 200.0
# 搬运入口目标点纵向像素坐标。当前图像为 QVGA 320x240, 推行前对正使用底边 y=240。
ASSISTANT_TRANSPORT_TARGET_Y_PX = 240.0
# TARGET_FOUND 最小面积阈值。
OBJECT_MIN_AREA = 50.0
# TARGET_FOUND 横向容差，单位为像素。
OBJECT_X_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_X_PX
# TARGET_FOUND 纵向容差，单位为像素。
OBJECT_Y_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_Y_PX
# 连续满足 hook 条件多少帧后确认找到目标。
OBJECT_STABLE_FRAMES = 3
# 回库黄线采样半宽，单位像素。
RETURN_LINE_SAMPLE_HALF_WIDTH_PX = 5
# 回库黄线目标 Y 坐标。
RETURN_LINE_TARGET_Y_PX = 220.0
# 回库黄线 Y 死区，单位像素。
RETURN_LINE_DEADZONE_Y_PX = 4.0
# 回库黄线纵向速度 P 环增益。
RETURN_LINE_KP_Y = -0.05
# 回库黄线纵向速度限幅。
RETURN_LINE_MAX_VY = 5.0
# 回库黄线纵向最小有效速度。
RETURN_LINE_MIN_SPEED = 0.0
# 回库黄线参与中心计算的最大厚度，单位像素。
RETURN_LINE_MAX_THICKNESS_PX = 30
# 回库黄线候选点左右水平联通黄线的最小合计长度，单位像素。
RETURN_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50
# 回库黄线连续丢线停车帧数。
RETURN_LINE_MISSING_FINISH_FRAMES = 5

_I16_MIN = -32768
_I16_MAX = 32767
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
    """! @brief 编码固定长度短帧"""

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
    """! @brief 解码固定长度短帧"""

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
    """! @brief 计算固定帧 CRC8"""

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
    """! @brief 编码本地视觉速度 body"""

    return (
        _pack_scaled(vx)
        + _pack_scaled(vy)
        + _pack_scaled(omega)
        + bytes([1 if has_omega else 0])
    )


def decode_velocity_body(body):
    """! @brief 解码本地视觉速度 body"""

    return {
        "vx": _unpack_scaled(body, 0),
        "vy": _unpack_scaled(body, 2),
        "omega": _unpack_scaled(body, 4),
        "has_omega": bool(body[6]),
    }


def encode_assistant_vision_task_sync_body(state, target, arg):
    """! @brief 编码辅车视觉任务同步 body"""

    return bytes([_require_u8(state), _require_u8(target)]) + _pack_i16(arg)


def decode_assistant_vision_task_sync_body(body):
    """! @brief 解码辅车视觉任务同步 body"""

    return {
        "state": int(body[0]),
        "target": int(body[1]),
        "arg": _unpack_i16(body, 2),
    }


def pack_task_arg(config_id, object_id):
    """! @brief 把配置号与物体编号打包进同步参数槽位"""

    packed = (int(config_id) & 0xFF) | ((int(object_id) & 0xFF) << 8)
    if packed >= 0x8000:
        packed -= 0x10000
    return packed


def unpack_task_arg_config(arg):
    """! @brief 读取同步参数中的配置号"""

    return int(arg) & 0xFF


def unpack_task_arg_object_id(arg):
    """! @brief 读取同步参数中的物体编号"""

    return (int(arg) >> 8) & 0xFF


def encode_assistant_vision_event_report_body(event, value):
    """! @brief 编码辅车视觉事件回报 body"""

    return bytes([_require_u8(event)]) + _pack_i16(value)


def decode_assistant_vision_event_report_body(body):
    """! @brief 解码辅车视觉事件回报 body"""

    return {
        "event": int(body[0]),
        "value": _unpack_i16(body, 1),
    }


def format_vision_frame(vx, vy):
    """! @brief 将当前速度修正量编码为固定长度短帧

    @param vx 车体系 x 方向视觉速度修正量
    @param vy 车体系 y 方向视觉速度修正量
    @return 当前主线速度短帧
    """

    return encode_frame(
        Mode.UDP,
        Topic.LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_ack_frame(reliable_seq):
    """! @brief 格式化本地同步确认帧

    @param reliable_seq 被确认的同步序号
    @return ACK 短帧
    """

    return encode_frame(Mode.ACK, Topic.ASSISTANT_VISION_TASK_SYNC, reliable_seq, b"")


def format_event_frame(reliable_seq, event, value):
    """! @brief 格式化可靠事件回报帧

    @param reliable_seq 当前同步序号
    @param event 事件编号
    @param value 事件附加值
    @return 事件短帧
    """

    return encode_frame(
        Mode.TCP,
        Topic.ASSISTANT_VISION_EVENT_REPORT,
        reliable_seq,
        encode_assistant_vision_event_report_body(event, value),
    )


def parse_sync_packet(frame_bytes):
    """! @brief 解析 RT1021 下发的本地同步短帧

    @param frame_bytes 原始固定帧
    @return 同步包字段字典，输入无效时返回 None
    """

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


def parse_ack_packet(frame_bytes):
    """! @brief 解析可靠事件确认短帧

    @param frame_bytes 原始固定帧
    @return 确认包字段字典，输入无效时返回 None
    """

    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.ASSISTANT_VISION_EVENT_REPORT:
        return None
    return {"reliable_seq": int(frame["seq"])}


def is_newer_seq(seq, last_seq):
    """! @brief 按 0..255 半环规则判断序号是否更新

    @param seq 待判断序号
    @param last_seq 已记录序号，None 表示没有历史序号
    @return seq 是否新于 last_seq
    """

    if last_seq is None:
        return True
    diff = (int(seq) - int(last_seq)) % SEQ_RING_SIZE
    return diff != 0 and diff < SEQ_HALF_RING


def default_now_ms():
    """! @brief 读取毫秒时间戳

    @return 当前毫秒时间戳
    """

    ticks_ms = getattr(time, "ticks_ms", None)
    if ticks_ms is not None:
        return int(ticks_ms())
    return int(time.time() * 1000)


def should_resend(now_ms, last_sent_ms, interval_ms):
    """! @brief 判断可靠包是否到达重复发送时间

    @param now_ms 当前毫秒时间戳
    @param last_sent_ms 上次发送毫秒时间戳，None 表示尚未发送
    @param interval_ms 重发间隔，单位毫秒
    @return 是否应该发送或重发
    """

    if last_sent_ms is None:
        return True
    interval_ms = int(interval_ms)
    if interval_ms <= 0:
        return True
    now_ms = int(now_ms)
    last_sent_ms = int(last_sent_ms)
    if now_ms < last_sent_ms:
        return True
    return now_ms - last_sent_ms >= interval_ms


def _apply_min_speed(value, min_speed, limit=None):
    """! @brief 对非零速度量施加最小幅值和可选上限

    @param value 原始速度量
    @param min_speed 最小速度幅值
    @param limit 可选速度幅值上限
    @return 处理后的速度量
    """

    value = float(value)
    if value == 0.0:
        return 0.0
    min_speed = abs(float(min_speed))
    if limit is not None:
        limit = abs(float(limit))
        if min_speed > limit:
            min_speed = limit
    if 0.0 < value < min_speed:
        value = min_speed
    if -min_speed < value < 0.0:
        value = -min_speed
    if limit is not None:
        if value > limit:
            value = limit
        if value < -limit:
            value = -limit
    return value


def _clamp(value, limit):
    """! @brief 按对称上下限约束数值

    @param value 原始数值
    @param limit 绝对值上限
    @return 限幅后的数值
    """

    value = float(value)
    limit = abs(float(limit))
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _build_axis_velocity(error, kp, limit=None):
    """! @brief 根据跟随误差和增益生成单轴速度量"""

    return _apply_min_speed(
        float(error) * float(kp),
        FOLLOW_CONTROL_MIN_SPEED,
        limit,
    )


def _axis_p_velocity(error, deadzone, kp, limit, min_speed):
    """! @brief 根据死区和 P 环参数生成单轴速度量

    @param error 当前轴误差
    @param deadzone 当前轴死区
    @param kp 当前轴增益
    @param limit 当前轴速度限幅
    @param min_speed 当前轴最小有效速度
    @return 当前轴速度控制量
    """

    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    return _apply_min_speed(
        _clamp(error * float(kp), limit),
        min_speed,
        limit,
    )


def build_follow_command(valid, err_x, err_y):
    """! @brief 在 ART 端完成辅车跟随阶段判断与速度修正量生成

    @param valid 当前帧是否存在有效目标
    @param err_x 目标中心相对画面中心的横向像素差值
    @param err_y 目标尺度量相对目标尺度量的纵向差值
    @return 包含阶段名和速度修正量的字典
    """

    if int(valid) != 1:
        return {
            "phase": "MARKER_MISSING",
            "command_vx": 0.0,
            "command_vy": 0.0,
        }

    err_x = float(err_x)
    err_y = float(err_y)
    deadzone_x = float(FOLLOW_X_DEADZONE_PX)
    deadzone_y = float(FOLLOW_Y_DEADZONE_PX)
    if abs(err_x) <= deadzone_x and abs(err_y) <= deadzone_y:
        return {
            "phase": "CENTER_HOLD",
            "command_vx": 0.0,
            "command_vy": 0.0,
        }

    x_active = abs(err_x) > deadzone_x
    y_active = abs(err_y) > deadzone_y

    if x_active and y_active:
        max_y = float(FOLLOW_CONTROL_MAX_Y)
        return {
            "phase": "ALIGN_XY",
            "command_vx": _build_axis_velocity(err_x, FOLLOW_CONTROL_KP_X),
            "command_vy": _build_axis_velocity(err_y, FOLLOW_CONTROL_KP_Y, max_y),
        }

    if x_active:
        return {
            "phase": "ALIGN_X",
            "command_vx": _build_axis_velocity(err_x, FOLLOW_CONTROL_KP_X),
            "command_vy": 0.0,
        }

    max_y = float(FOLLOW_CONTROL_MAX_Y)
    return {
        "phase": "ALIGN_Y",
        "command_vx": 0.0,
        "command_vy": _build_axis_velocity(err_y, FOLLOW_CONTROL_KP_Y, max_y),
    }


def blob_rect_to_bbox(rect):
    """! @brief 将 OpenMV 的 x,y,w,h 矩形转换为边界框四元组

    @param rect blob.rect() 返回的矩形元组
    @return left, top, right, bottom 形式的边界框
    """

    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    """! @brief 将边界框统一到地板在下方的位置判断坐标系

    @param left 原始识别框左边界
    @param top 原始识别框上边界
    @param right 原始识别框右边界
    @param bottom 原始识别框下边界
    @param img_height 当前图像高度
    @return left, top, right, bottom 形式的归一化边界框
    """

    normalized_top = img_height - bottom
    normalized_bottom = img_height - top
    return left, normalized_top, right, normalized_bottom


def compute_lateral_error(blob_cx, cx_screen):
    """! @brief 计算跟随控制使用的横向偏差

    @param blob_cx 当前目标横向中心像素坐标
    @param cx_screen 画面横向中心像素坐标
    @return 当前目标中心相对画面中心的横向整数偏差
    """

    return int(round(float(blob_cx) - float(cx_screen)))


def edge_length(p0, p1):
    """! @brief 计算两点间边长

    @param p0 第一端点 x, y
    @param p1 第二端点 x, y
    @return 两点间欧氏距离
    """

    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    return (dx * dx + dy * dy) ** 0.5


def get_marker_corners(blob):
    """! @brief 返回用于距离量和显示的大角点集合

    @param blob 当前色块对象
    @return 优先使用最小外接旋转矩形的四个角点
    """

    return tuple(blob.min_corners())


def compute_marker_span(corners):
    """! @brief 基于梯形上底和下底平均值计算当前目标尺度量

    @param corners 按顺时针排序的四个角点
    @return 当前目标的纵向距离代理尺度量
    """

    top_width = edge_length(corners[0], corners[1])
    bottom_width = edge_length(corners[3], corners[2])
    return (top_width + bottom_width) / 2.0


def compute_vertical_error(marker_span, target_span):
    """! @brief 计算跟随控制使用的纵向偏差

    @param marker_span 当前目标的尺度量
    @param target_span 纵向目标尺度量
    @return 当前目标尺度量相对目标尺度量的纵向整数偏差
    """

    return int(round(float(marker_span) - float(target_span)))


def blob_area(blob):
    """! @brief 读取候选物体面积

    @param blob 候选色块对象
    @return 候选目标面积
    """

    area_fn = getattr(blob, "area", None)
    if area_fn is not None:
        return float(area_fn())
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return float((right - left) * (bottom - top))


class YoloDetectionBlob:
    """! @brief 让模型检测框复用现有物体候选接口"""

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
    """! @brief 读取候选物体外接框的最大边长"""

    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(float(right - left), float(bottom - top))


def task_thresholds(thresholds):
    """! @brief 统一读取单 LAB 与多 LAB 任务配置"""

    if len(thresholds) == 6 and not isinstance(thresholds[0], (tuple, list)):
        return (thresholds,)
    return thresholds


def object_task_parts(task):
    """! @brief 兼容读取旧版与新版找物体任务配置"""

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
        OBJECT_BLOB_MERGE_MARGIN,
        OBJECT_BLOB_PIXELS_THRESHOLD,
        OBJECT_BLOB_AREA_THRESHOLD,
        0,
        True,
    )


def object_task_name_from_id(object_id):
    """! @brief 根据物体编号返回配置中的任务名称"""

    object_id = int(object_id)
    if object_id <= 0:
        return None
    index = object_id - 1
    if index >= len(OBJECT_TASKS):
        return None
    return OBJECT_TASKS[index][0]


def load_yolo_model():
    """! @brief 在开启开关时加载 YOLO 模型"""

    if not OBJECT_DETECTION_USE_YOLO or tf is None:
        return None
    return tf.load(YOLO_MODEL_PATH)


def _copy_image_for_yolo(img):
    """! @brief 生成 YOLO 推理使用的图像副本"""

    copy_fn = getattr(img, "copy", None)
    if copy_fn is None:
        return img
    return copy_fn(YOLO_IMAGE_COPY_SCALE, 1)


def _image_width(img):
    """! @brief 读取图像宽度"""

    width_fn = getattr(img, "width", None)
    if width_fn is not None:
        return float(width_fn())
    return 320.0


def _image_height(img):
    """! @brief 读取图像高度"""

    height_fn = getattr(img, "height", None)
    if height_fn is not None:
        return float(height_fn())
    return 240.0


def _label_name(label):
    """! @brief 返回 YOLO 标签名"""

    label = int(label)
    if 0 <= label < len(YOLO_LABELS):
        return YOLO_LABELS[label]
    return "unknown"


def _build_yolo_object_candidates(img, yolo_net=None):
    """! @brief 从 YOLO 检测结果生成物体候选"""

    net = yolo_net
    if net is None:
        net = load_yolo_model()
    if net is None or tf is None:
        return []
    detect_img = _copy_image_for_yolo(img)
    image_width = _image_width(img)
    image_height = _image_height(img)
    allowed_task_names = {task[0] for task in OBJECT_TASKS}
    candidates = []
    for detected in tf.detect(net, detect_img):
        x1, y1, x2, y2, label, score = detected
        if float(score) <= float(YOLO_MIN_SCORE):
            continue
        task_name = _label_name(label)
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
            left, top, right, bottom, image_height
        )
        candidates.append(
            (task_name, blob.cx(), blob.cy(), protocol_bottom, blob.area(), blob)
        )
    return candidates


def _find_blobs_with_task_config(
    img, thresholds, pixels_threshold, area_threshold, merge_margin
):
    """! @brief 按当前任务配置调用板端找色块接口"""

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
    """! @brief 读取合并色块的颜色码"""

    code_fn = getattr(blob, "code", None)
    if code_fn is not None:
        return int(code_fn())
    try:
        return int(blob[8])
    except Exception:
        return None


def _blob_matches_required_thresholds(blob, threshold_count, require_all_thresholds):
    """! @brief 判断色块是否满足当前任务的颜色簇命中要求"""

    if int(threshold_count) <= 1 or not bool(require_all_thresholds):
        return True
    code = _blob_code(blob)
    if code is None:
        return True
    expected_code = (1 << int(threshold_count)) - 1
    return (code & expected_code) == expected_code


def blob_bbox_overlaps(left, top, right, bottom, other_blob):
    """! @brief 判断两个候选框是否存在有效重叠"""

    other_left, other_top, other_right, other_bottom = blob_rect_to_bbox(
        other_blob.rect()
    )
    return (
        min(float(right), float(other_right)) > max(float(left), float(other_left))
        and min(float(bottom), float(other_bottom)) > max(float(top), float(other_top))
    )


def blob_matches_all_thresholds(img, blob, thresholds, pixels_threshold, area_threshold, merge):
    """! @brief 判断候选色块是否被同一目标的全部 LAB 阈值命中"""

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


def build_blob_candidates(img):
    """! @brief 提取跟随模式颜色候选目标的重心、底边与尺度量信息

    @param img 当前帧图像对象
    @return 候选目标列表，元素格式为 名称, cx, cy, bottom, span, blob
    """

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
            _, _, _, bottom = normalize_bbox_for_protocol(
                left, top, right, bottom, img_height
            )
            marker_span = compute_marker_span(get_marker_corners(blob))
            candidates.append(
                (task_name, blob.cx(), blob.cy(), bottom, marker_span, blob)
            )
    return candidates


def build_object_blob_candidates(img, yolo_net=None):
    """! @brief 提取找物体模式候选目标的重心、底边与面积信息

    @param img 当前帧图像对象
    @return 候选目标列表，元素格式为 名称, cx, cy, bottom, area, blob
    """

    if OBJECT_DETECTION_USE_YOLO:
        return _build_yolo_object_candidates(img, yolo_net)
    candidates = []
    img_height = img.height()
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
            if not _blob_matches_required_thresholds(
                blob,
                len(thresholds),
                require_all_thresholds,
            ):
                continue
            if blob_area(blob) < float(area_threshold):
                continue
            if int(max_side_length) > 0 and blob_max_side_length(blob) > float(
                max_side_length
            ):
                continue
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, bottom = normalize_bbox_for_protocol(
                left, top, right, bottom, img_height
            )
            candidates.append(
                (task_name, blob.cx(), blob.cy(), bottom, blob_area(blob), blob)
            )
    return candidates


def _pixel_to_lab(pixel):
    if pixel is None:
        return None
    if omv_image is not None:
        try:
            return omv_image.rgb_to_lab(pixel)
        except Exception:
            pass
    return pixel


def _pixel_matches_threshold(pixel, threshold):
    lab = _pixel_to_lab(pixel)
    if lab is None:
        return False
    try:
        if len(lab) < 3:
            return False
    except TypeError:
        return False
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
        RETURN_LINE_YELLOW_THRESHOLD,
    )


def _return_line_has_horizontal_connected_at(
    img, x, y, image_width, image_height, required_connected
):
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
        # 显示坐标按翻转后坐标标注, 但 get_pixel 读取表现为原始坐标。
        if not _return_line_pixel_matches(img, x, y, image_width, image_height):
            continue
        if top is None:
            top = int(y)
        bottom = int(y)
    if top is None or bottom is None:
        return None
    max_thickness = int(RETURN_LINE_MAX_THICKNESS_PX)
    if max_thickness > 0 and int(bottom) - int(top) > max_thickness:
        top = int(bottom) - max_thickness
    return (float(top) + float(bottom)) / 2.0


def _build_return_line_y_from_pixels(img, image_width, image_height, previous_line_y=None):
    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None:
        return None
    center_x = int(float(image_width) / 2.0)
    half_width = int(RETURN_LINE_SAMPLE_HALF_WIDTH_PX)
    saw_candidate = False
    for x in _return_line_sample_columns(center_x, half_width):
        if x < 0 or x >= int(image_width):
            continue
        line_y = _return_line_y_on_column(img, x, image_width, image_height)
        if line_y is None:
            continue
        saw_candidate = True
        has_horizontal_connected = _return_line_has_horizontal_connected_at(
            img,
            x,
            int(round(line_y)),
            image_width,
            image_height,
            RETURN_LINE_MIN_HORIZONTAL_CONNECTED_PX,
        )
        if has_horizontal_connected:
            return line_y
    if saw_candidate:
        return previous_line_y
    return None


def build_return_line_y_from_image(img, image_width, image_height, previous_line_y=None):
    return _build_return_line_y_from_pixels(
        img, image_width, image_height, previous_line_y
    )


def build_return_line_velocity_from_y(line_y):
    if line_y is None:
        return 0.0, 0.0
    err_y = float(line_y) - float(RETURN_LINE_TARGET_Y_PX)
    if abs(err_y) <= float(RETURN_LINE_DEADZONE_Y_PX):
        return 0.0, 0.0
    return 0.0, _apply_min_speed(
        err_y * float(RETURN_LINE_KP_Y),
        RETURN_LINE_MIN_SPEED,
        RETURN_LINE_MAX_VY,
    )


def choose_best_candidate(candidates, target_x, target_y):
    """! @brief 选择最靠近当前目标点的候选目标

    @param candidates 候选目标列表
    @param target_x 目标点 x 坐标
    @param target_y 目标点 y 坐标
    @return 被选中的候选目标
    """

    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[3]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    """! @brief 选择面积最大的候选目标"""

    return max(candidates, key=lambda item: float(item[4]))


def filter_candidates_in_target_window(candidates, target_x, target_y, tolerance_x, tolerance_y):
    """! @brief 保留底边命中当前目标窗口的候选目标"""

    _ = (target_x, tolerance_x)
    return [
        candidate
        for candidate in candidates
        if abs(float(candidate[3]) - float(target_y)) <= float(tolerance_y)
    ]


def should_filter_candidates_by_target_window(state):
    """! @brief 判断当前上下文是否只接受命中目标窗口的候选目标"""

    if state.current_sync is None:
        return False
    config_id = state.current_object_config_id()
    return (
        int(state.current_sync["state"]) == int(State.TRANSPORT_OBJECT)
        and int(config_id) == int(Task.TRANSPORT)
    )


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    """! @brief 在调试画面上绘制当前选中目标的矩形框

    @param img 当前图像对象
    @param blob 当前选中的色块对象
    @param pixel_x 当前目标中心 x
    @param pixel_y 当前目标中心 y
    """

    _ = pixel_x
    _ = pixel_y
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


def build_object_observation(
    valid,
    center_x,
    bottom_y,
    area,
    image_width,
    image_height,
    config_id=Task.SEARCH,
):
    """! @brief 根据物体中心、底边和面积生成找物体观测

    @param valid 当前帧是否存在有效目标
    @param center_x 目标中心 x 坐标
    @param bottom_y 目标底边 y 坐标
    @param area 目标面积
    @param image_width 图像宽度
    @param image_height 图像高度
    @param config_id 当前找物体配置编号
    @return x, y, value 观测字段元组
    """

    if int(valid) != 1:
        return 0.0, 0.0, 0.0
    target_x, target_y = build_object_target_point(
        image_width,
        image_height,
        config_id,
    )
    return (
        float(center_x) - target_x,
        float(bottom_y) - target_y,
        float(area),
    )


def build_object_target_point(
    image_width,
    image_height,
    config_id=Task.SEARCH,
):
    """! @brief 根据当前配置生成找物体目标点"""

    _ = image_width
    _ = image_height
    target_x = float(OBJECT_APPROACH_TARGET_X_PX)
    if int(config_id) == int(Task.ORBIT):
        return float(OBJECT_ORBIT_TARGET_X_PX), float(OBJECT_ORBIT_TARGET_Y_PX)
    if int(config_id) == int(Task.TRANSPORT):
        return target_x, float(ASSISTANT_TRANSPORT_TARGET_Y_PX)
    return target_x, float(OBJECT_APPROACH_TARGET_Y_PX)


def _build_object_y_velocity(err_y, image_height):
    """! @brief 根据底边纵向误差生成找物体纵向速度"""

    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_APPROACH_DEADZONE_Y_PX):
        return 0.0
    scaled_error = err_y * (
        float(OBJECT_APPROACH_MAX_VY)
        / abs(float(OBJECT_APPROACH_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_APPROACH_KP_Y),
        OBJECT_APPROACH_MIN_SPEED,
        OBJECT_APPROACH_MAX_VY,
    )


def build_object_approach_velocity_from_error(err_x, err_y, image_height):
    """! @brief 根据找物体误差生成速度控制量

    @param err_x 目标中心相对画面中心的横向误差
    @param err_y 目标底边相对图像底边的纵向误差
    @param image_height 图像高度
    @return vx, vy 速度控制量
    """

    return (
        _axis_p_velocity(
            err_x,
            OBJECT_APPROACH_DEADZONE_X_PX,
            OBJECT_APPROACH_KP_X,
            OBJECT_APPROACH_MAX_VX,
            OBJECT_APPROACH_MIN_SPEED,
        ),
        _build_object_y_velocity(err_y, image_height),
    )


def build_object_approach_velocity_from_observation(observation, image_height):
    """! @brief 根据找物体观测生成速度控制量

    @param observation x, y, value 观测字段元组
    @param image_height 图像高度
    @return vx, vy 速度控制量
    """

    x, y, value = observation
    if float(value) <= 0.0:
        return float(OBJECT_MISSING_SEARCH_VX), float(OBJECT_MISSING_SEARCH_VY)
    return build_object_approach_velocity_from_error(x, y, image_height)


def build_object_orbit_velocity_from_observation(observation, image_height):
    """! @brief 根据绕行目标物体观测生成平移修正量

    @param observation x, y, value 观测字段元组
    @param image_height 图像高度
    @return vx, vy 平移修正量
    """

    _, _, value = observation
    if float(value) <= 0.0:
        return 0.0, 0.0
    x, y, _ = observation
    return (
        _axis_p_velocity(
            x,
            OBJECT_ORBIT_DEADZONE_X_PX,
            OBJECT_ORBIT_KP_X,
            OBJECT_ORBIT_MAX_VX,
            OBJECT_ORBIT_MIN_SPEED,
        ),
        _build_object_orbit_y_velocity(y, image_height),
    )


def _build_object_orbit_y_velocity(err_y, image_height):
    """! @brief 根据底边纵向误差生成绕行修正纵向速度"""

    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_ORBIT_DEADZONE_Y_PX):
        return 0.0
    if float(OBJECT_ORBIT_KP_Y) == 0.0:
        return 0.0
    scaled_error = err_y * (
        float(OBJECT_ORBIT_MAX_VY)
        / abs(float(OBJECT_ORBIT_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_ORBIT_KP_Y),
        OBJECT_ORBIT_MIN_SPEED,
        OBJECT_ORBIT_MAX_VY,
    )


class AssistantVisionState:
    """! @brief 辅车视觉轻量状态

    @details 维护当前模式、最后同步序号、稳定计数和待确认事件
    """

    def __init__(
        self,
        min_area=OBJECT_MIN_AREA,
        tolerance_x=OBJECT_X_TOLERANCE_PX,
        tolerance_y=OBJECT_Y_TOLERANCE_PX,
        stable_frames=OBJECT_STABLE_FRAMES,
        now_ms=None,
        event_resend_interval_ms=RELIABLE_RESEND_INTERVAL_MS,
    ):
        """! @brief 初始化辅车视觉轻量状态

        @param min_area 找到目标所需的最小面积
        @param tolerance_x 横向容差，单位像素
        @param tolerance_y 纵向容差，单位像素
        @param stable_frames 连续满足条件的最小帧数
        @param now_ms 毫秒时钟函数，None 时使用默认时钟
        @param event_resend_interval_ms 可靠事件重发间隔，单位毫秒
        """

        self.min_area = float(min_area)
        self.tolerance_x = float(tolerance_x)
        self.tolerance_y = float(tolerance_y)
        self.required_stable_frames = int(stable_frames)
        self.mode = RunMode.FOLLOW
        self.current_sync = None
        self._last_sync_seq = None
        self._stable_count = 0
        self._pending_event = None
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = None
        self._now_ms = now_ms or default_now_ms
        self._event_resend_interval_ms = int(event_resend_interval_ms)
        self._last_return_line_y = None

    def handle_control_line(self, line):
        """! @brief 处理 RT1021 发来的同步或确认短帧

        @param line 原始控制短帧
        @return 需要回复的 ACK 帧，无需回复时返回 None
        """

        sync_packet = parse_sync_packet(line)
        if sync_packet is not None:
            return self._handle_sync_packet(sync_packet)
        ack_packet = parse_ack_packet(line)
        if ack_packet is not None:
            self._handle_ack_packet(ack_packet)
        return None

    def _handle_sync_packet(self, packet):
        """! @brief 处理本地同步包

        @param packet 解析后的同步包字段
        @return 同步确认帧文本
        """

        reliable_seq = int(packet["reliable_seq"])
        if self._should_apply_sync(reliable_seq):
            self.current_sync = {
                "reliable_seq": reliable_seq,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            self._last_sync_seq = reliable_seq
            self.mode = self._mode_from_sync(self.current_sync)
            self._stable_count = 0
            self._last_return_line_y = None
        return format_ack_frame(reliable_seq)

    def _should_apply_sync(self, reliable_seq):
        """! @brief 判断同步序号是否应覆盖当前模式

        @param reliable_seq 待判断的同步序号
        @return 是否应用该同步包
        """

        if self._last_sync_seq is None:
            return True
        if int(reliable_seq) == int(self._last_sync_seq):
            return False
        return is_newer_seq(reliable_seq, self._last_sync_seq)

    def _mode_from_sync(self, sync):
        """! @brief 根据同步字段解析当前模式

        @param sync 当前同步字段
        @return 模式名
        """

        if (
            int(sync["state"]) == State.APPROACH_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and (
                unpack_task_arg_config(sync["arg"]) == Task.SEARCH
                or unpack_task_arg_config(sync["arg"])
                == Task.TRANSPORT
            )
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.TRANSPORT_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and unpack_task_arg_config(sync["arg"]) == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and unpack_task_arg_config(sync["arg"]) == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and unpack_task_arg_config(sync["arg"]) == Task.ORBIT
        ):
            return RunMode.ORBIT_OBJECT
        if (
            int(sync["state"]) == State.RETURN_FOLLOW
            and int(sync["target"]) == Target.NONE
            and unpack_task_arg_config(sync["arg"]) == Task.RETURN_GARAGE_LINE
        ):
            return RunMode.RETURN_LINE
        return RunMode.FOLLOW

    def current_object_config_id(self):
        """! @brief 返回当前找物体阶段使用的目标点配置编号"""

        if self.current_sync is None:
            return Task.SEARCH
        return unpack_task_arg_config(self.current_sync["arg"])

    def current_object_id(self):
        """! @brief 返回当前指定的物体编号"""

        if self.current_sync is None:
            return 0
        return unpack_task_arg_object_id(self.current_sync["arg"])

    def last_return_line_y(self):
        """! @brief 返回上一帧有效回库黄线 Y"""

        return self._last_return_line_y

    def remember_return_line_y(self, line_y):
        """! @brief 保存当前有效回库黄线 Y"""

        if line_y is not None:
            self._last_return_line_y = float(line_y)

    def _handle_ack_packet(self, packet):
        """! @brief 处理可靠事件确认包

        @param packet 解析后的确认包字段
        """

        if self._pending_event is None:
            return
        if int(packet["reliable_seq"]) == int(self._pending_event["reliable_seq"]):
            self._pending_event = None
            self._pending_event_last_sent_ms = None

    def has_pending_event(self):
        """! @brief 判断是否存在等待确认的可靠事件"""

        return self._pending_event is not None

    def accept_object_observation(self, observation):
        """! @brief 累计找物体稳定条件并按需创建可靠事件

        @param observation x, y, value 观测字段元组
        """

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

    def accept_return_line_observation(self, line_y):
        """! @brief 回库黄线阶段保持跟线并清理完成事件累计"""

        _ = line_y
        self._stable_count = 0
        return

        if self.mode != RunMode.RETURN_LINE or self.current_sync is None:
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
        if line_y is not None:
            self._stable_count = 0
            return
        self._stable_count += 1
        if self._stable_count >= int(RETURN_LINE_MISSING_FINISH_FRAMES):
            self._create_event(current_sync_seq, event_id, 0)

    def _observation_matches_target(self, x, y, value):
        """! @brief 判断当前观测是否满足找到目标条件

        @param x 横向误差
        @param y 纵向误差
        @param value 目标面积
        @return 是否满足目标稳定条件
        """

        return (
            float(value) >= self.min_area
            and abs(float(x)) <= self.tolerance_x
            and abs(float(y)) <= self.tolerance_y
        )

    def _current_event_id(self):
        """! @brief 返回当前模式应回报的事件编号

        @return 事件编号, 当前模式不支持时返回 None
        """

        if self.current_sync is None:
            return None
        state = int(self.current_sync["state"])
        target = int(self.current_sync["target"])
        config_id = unpack_task_arg_config(self.current_sync["arg"])
        if (
            state == State.APPROACH_OBJECT
            and target == Target.OBJECT
            and config_id == Task.SEARCH
        ):
            return Event.TARGET_FOUND
        if (
            state == State.APPROACH_OBJECT
            and target == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return Event.ALIGNED
        if (
            state == State.ORBIT
            and target == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return Event.ALIGNED
        if (
            state == State.RETURN_FOLLOW
            and target == Target.NONE
            and config_id == Task.RETURN_GARAGE_LINE
        ):
            return Event.RETURN_GARAGE_FINISHED
        return None

    def _create_event(self, reliable_seq, event, value):
        """! @brief 创建待确认事件

        @param reliable_seq 当前同步序号
        @param event 事件编号
        @param value 事件附加值
        """

        self._pending_event = {
            "reliable_seq": int(reliable_seq),
            "event": int(event),
            "value": int(float(value)),
        }
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = int(reliable_seq)

    def next_event_frame(self):
        """! @brief 返回待确认事件帧，没有事件时返回 None

        @return 事件帧文本，未到发送时机或无待确认事件时返回 None
        """

        if self._pending_event is None:
            return None
        now_ms = self._now_ms()
        if not should_resend(
            now_ms, self._pending_event_last_sent_ms, self._event_resend_interval_ms
        ):
            return None
        self._pending_event_last_sent_ms = now_ms
        return format_event_frame(
            self._pending_event["reliable_seq"],
            self._pending_event["event"],
            self._pending_event["value"],
        )


def write_line(uart, line):
    """! @brief 按协议发送固定长度短帧

    @param uart 当前使用的串口对象
    @param line 待发送的固定帧 bytes
    """

    if not isinstance(line, bytes):
        line = bytes(line)
    remaining = line
    while remaining:
        written = uart.write(remaining)
        if written is None:
            written = len(remaining)
        written = int(written)
        if written <= 0:
            return
        remaining = remaining[written:]


def init_uart():
    """! @brief 初始化主通信串口

    @return 配置完成的 UART 对象
    @raises RuntimeError 主机侧导入且无 UART 平台支持时抛出
    """

    if UART is None:
        raise RuntimeError("UART unavailable in host environment")
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    """! @brief 初始化 OpenART 摄像头参数

    @return image_width, image_height 图像尺寸
    @raises RuntimeError 主机侧导入且无 sensor 平台支持时抛出
    """

    if sensor is None:
        raise RuntimeError("sensor unavailable in host environment")

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)  # type: ignore
    sensor.set_auto_gain(False)  # type: ignore
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width(), sensor.height()


def _find_control_frame_start(rx_buffer):
    """! @brief 查找辅车视觉控制链路合法帧起点"""

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
    return -1


def process_uart_input(uart, rx_buffer, state):
    """! @brief 处理 RT1021 发来的控制短帧

    @param uart 控制链路串口对象
    @param rx_buffer 上一轮遗留的未完整输入 bytes
    @param state 辅车视觉状态对象
    @return 更新后的接收缓冲区
    """

    any_fn = getattr(uart, "any", None)
    if any_fn is None:
        return rx_buffer
    size = uart.any()
    if not size:
        return rx_buffer
    try:
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
    except Exception:
        return rx_buffer
    while len(rx_buffer) >= FRAME_SIZE:
        frame_start = _find_control_frame_start(rx_buffer)
        if frame_start < 0:
            return rx_buffer[-(FRAME_SIZE - 1):]
        if frame_start > 0:
            rx_buffer = rx_buffer[frame_start:]
        if len(rx_buffer) < FRAME_SIZE:
            return rx_buffer
        line = rx_buffer[:FRAME_SIZE]
        rx_buffer = rx_buffer[FRAME_SIZE:]
        reply = state.handle_control_line(line)
        if reply is not None:
            write_line(uart, reply)
    return rx_buffer


def process_follow_frame(uart, img, image_width, image_height):
    """! @brief 处理单帧跟随模式速度输出

    @param uart 辅车视觉串口
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    candidates = build_blob_candidates(img)
    if not candidates:
        follow_command = build_follow_command(valid=0, err_x=0, err_y=0)
        write_line(
            uart,
            format_vision_frame(
                vx=follow_command["command_vx"],
                vy=follow_command["command_vy"],
            ),
        )
        return

    cx_screen = float(image_width) / 2.0
    _, pixel_x, pixel_y, _, marker_span, best_blob = choose_best_candidate(
        candidates, cx_screen, image_height
    )
    err_x = compute_lateral_error(blob_cx=pixel_x, cx_screen=cx_screen)
    err_y = compute_vertical_error(marker_span=marker_span, target_span=FOLLOW_TARGET_Y)
    follow_command = build_follow_command(valid=1, err_x=err_x, err_y=err_y)
    draw_selected_marker(img=img, blob=best_blob, pixel_x=pixel_x, pixel_y=pixel_y)
    write_line(
        uart,
        format_vision_frame(
            vx=follow_command["command_vx"],
            vy=follow_command["command_vy"],
        ),
    )


def process_object_frame(uart, state, img, image_width, image_height, yolo_net=None):
    """! @brief 处理单帧找物体模式速度输出与可靠事件

    @param uart 辅车视觉串口
    @param state 辅车视觉状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    candidates = build_object_blob_candidates(img, yolo_net)
    object_id = state.current_object_id()
    if object_id > 0:
        selected_task_name = object_task_name_from_id(object_id)
        if selected_task_name is not None:
            candidates = [
                candidate for candidate in candidates if candidate[0] == selected_task_name
            ]
    if not candidates:
        observation = build_object_observation(0, 0, 0, 0, image_width, image_height)
    else:
        config_id = state.current_object_config_id()
        target_x, target_y = build_object_target_point(
            image_width,
            image_height,
            config_id,
        )
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
            observation = build_object_observation(0, 0, 0, 0, image_width, image_height)
        else:
            if use_target_window_filter:
                _, pixel_x, pixel_y, bottom_y, area, best_blob = (
                    choose_largest_area_candidate(candidates)
                )
            else:
                _, pixel_x, pixel_y, bottom_y, area, best_blob = choose_best_candidate(
                    candidates,
                    target_x,
                    target_y,
                )
            observation = build_object_observation(
                1,
                pixel_x,
                bottom_y,
                area,
                image_width,
                image_height,
                config_id,
            )
            _draw_debug_protocol_point(img, image_width, image_height, target_x, target_y)
            draw_selected_marker(img=img, blob=best_blob, pixel_x=pixel_x, pixel_y=pixel_y)
    if state.mode == RunMode.ORBIT_OBJECT:
        vx, vy = build_object_orbit_velocity_from_observation(observation, image_height)
    else:
        vx, vy = build_object_approach_velocity_from_observation(
            observation,
            image_height,
        )
    write_line(uart, format_vision_frame(vx, vy))
    state.accept_object_observation(observation)


def process_return_line_frame(uart, state, img, image_width, image_height):
    """! @brief 处理辅车回库黄线巡线速度与完成事件"""

    line_y = build_return_line_y_from_image(
        img,
        image_width,
        image_height,
        state.last_return_line_y(),
    )
    state.remember_return_line_y(line_y)
    vx, vy = build_return_line_velocity_from_y(line_y)
    write_line(uart, format_vision_frame(vx, vy))
    state.accept_return_line_observation(line_y)


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


def _draw_debug_protocol_point(img, image_width, image_height, x, y):
    """! @brief 将协议坐标点映射到调试画面并绘制十字"""

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


def draw_assistant_return_line_debug(img, image_width, image_height, line_y, vx, vy):
    """! @brief 绘制辅车回库黄线判定调试信息"""

    center_x = int(float(image_width) / 2.0)
    half_width = int(RETURN_LINE_SAMPLE_HALF_WIDTH_PX)
    max_thickness = int(RETURN_LINE_MAX_THICKNESS_PX)
    target_y = int(RETURN_LINE_TARGET_Y_PX)
    bottom_y = int(image_height) - 1

    def flip_x(x):
        return int(image_width) - 1 - int(x)

    def flip_y(y):
        return int(image_height) - 1 - int(y)

    def draw_line(x0, y0, x1, y1):
        _draw_debug_line(img, flip_x(x0), flip_y(y0), flip_x(x1), flip_y(y1))

    draw_line(center_x - half_width, 0, center_x - half_width, bottom_y)
    draw_line(center_x + half_width, 0, center_x + half_width, bottom_y)
    draw_line(0, target_y, int(image_width) - 1, target_y)
    if line_y is not None:
        draw_line(0, int(line_y), int(image_width) - 1, int(line_y))
        _draw_debug_cross(img, flip_x(center_x), flip_y(int(line_y)))
        line_text = "line_y=%.1f" % float(line_y)
    else:
        line_text = "line_y=none"

    _draw_debug_text(img, 2, 2, "assistant return line debug")
    _draw_debug_text(img, 2, 14, "max h=%d x=%d..%d" % (
        max_thickness,
        center_x - half_width,
        center_x + half_width,
    ))
    _draw_debug_text(img, 2, 26, "%s found=%d" % (line_text, 1 if line_y is not None else 0))
    _draw_debug_text(img, 2, 38, "vx=%.1f vy=%.1f" % (float(vx), float(vy)))


def run_assistant_return_line_debug():
    """! @brief 只显示辅车回库黄线判定调试画面"""

    image_width, image_height = init_sensor()
    last_line_y = None
    while True:
        img = sensor.snapshot()  # type: ignore
        apply_lens_correction(img)
        line_y = build_return_line_y_from_image(
            img,
            image_width,
            image_height,
            last_line_y,
        )
        if line_y is not None:
            last_line_y = line_y
        vx, vy = build_return_line_velocity_from_y(line_y)
        draw_assistant_return_line_debug(img, image_width, image_height, line_y, vx, vy)


def process_frame(uart, state, img, image_width, image_height, yolo_net=None):
    """! @brief 按当前模式处理单帧视觉输出

    @param uart 辅车视觉串口
    @param state 辅车视觉状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    if state.has_pending_event():
        event_frame = state.next_event_frame()
        if event_frame is not None:
            write_line(uart, event_frame)
        return

    if state.mode == RunMode.APPROACH_OBJECT or state.mode == RunMode.ORBIT_OBJECT:
        process_object_frame(uart, state, img, image_width, image_height, yolo_net)
    elif state.mode == RunMode.RETURN_LINE:
        process_return_line_frame(uart, state, img, image_width, image_height)
    else:
        process_follow_frame(uart, img, image_width, image_height)


def apply_lens_correction(img):
    """! @brief 执行当前帧镜头畸变校准"""

    try:
        img.lens_corr(strength=2.8, zoom=1.0)
    except MemoryError:
        pass


def run():
    """! @brief 持续检测目标并逐帧发送视觉短包"""

    uart = init_uart()
    image_width, image_height = init_sensor()
    yolo_net = load_yolo_model()
    state = AssistantVisionState()
    rx_buffer = b""

    while True:
        rx_buffer = process_uart_input(uart, rx_buffer, state)
        img = sensor.snapshot()  # type: ignore
        apply_lens_correction(img)
        if OBJECT_DETECTION_USE_YOLO:
            process_frame(uart, state, img, image_width, image_height, yolo_net)
        else:
            process_frame(uart, state, img, image_width, image_height)


if __name__ == "__main__":
    run()
