"""! @file main.py
@brief OpenART Vision master 主车物体搜索视觉入口
@details 负责接收 RT1021 视觉上下文, 输出搜索速度, 并在 hook 条件满足时可靠回报事件
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
    from machine import UART
except ImportError:
    UART = None


# OpenART 与 RT1021 通信使用的串口编号。
UART_ID = 2
# 串口波特率，需要与车端 UART6 保持一致。
UART_BAUDRATE = 115200
# 摄像头固定曝光时间，单位为微秒。
EXP_TIME_US = 300
# TARGET_FOUND 未确认时的重复发送间隔，单位为毫秒。
RELIABLE_RESEND_INTERVAL_MS = 100
# 固定帧模式编号。
MODE_UDP = 0x01
MODE_TCP = 0x02
MODE_ACK = 0x03
# 固定帧 body 槽位长度。
FRAME_BODY_SIZE = 8
FRAME_HEAD = 0xA5
# 固定帧总长度。
FRAME_SIZE = 13
# 本地视觉速度 topic。
TOPIC_LOCAL_VISION_VELOCITY = 0x01
# 主车视觉同步 topic。
TOPIC_MASTER_VISION_HOOK_SYNC = 0x10
# 主车视觉事件回报 topic。
TOPIC_MASTER_VISION_EVENT_REPORT = 0x12
# 红色候选目标的最小面积，小于该值不会触发找到事件。
OBJECT_MIN_AREA = 50.0
# 主车搜索目标丢失时输出的配置横向速度。
MASTER_MISSING_SEARCH_VX = 0.0
# 主车搜索目标丢失时输出的配置纵向速度。
MASTER_MISSING_SEARCH_VY = 2
# 主车搜索横向速度 P 环增益。
MASTER_SEARCH_KP_X = 0.05
# 主车搜索纵向速度 P 环增益。
MASTER_SEARCH_KP_Y = -0.15
# 主车搜索误差超出死区后的最小有效速度量。
MASTER_SEARCH_MIN_SPEED = 2.0
# 主车搜索横向误差死区, 单位为像素。
MASTER_SEARCH_DEADZONE_X_PX = 15.0
# 主车搜索纵向误差死区, 单位为像素。
MASTER_SEARCH_DEADZONE_Y_PX = 8.0
# 目标中心允许偏离搜索目标点的最大横向像素误差。
OBJECT_X_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_X_PX
# 目标底边允许偏离搜索目标点的最大纵向像素误差。
OBJECT_Y_TOLERANCE_PX = MASTER_SEARCH_DEADZONE_Y_PX
# 连续满足面积与位置条件多少帧后确认找到目标。
OBJECT_STABLE_FRAMES = 3
# 主车搜索横向速度限幅。
MASTER_SEARCH_MAX_VX = 5.0
# 主车搜索纵向速度限幅。
MASTER_SEARCH_MAX_VY = 5.0
# 主车绕行修正横向速度 P 环增益。
MASTER_ORBIT_KP_X = 0.05
# 主车绕行修正纵向速度 P 环增益。
MASTER_ORBIT_KP_Y = -0.30
# 主车绕行修正误差超出死区后的最小有效速度量。
MASTER_ORBIT_MIN_SPEED = 0
# 主车绕行修正横向误差死区, 单位为像素。
MASTER_ORBIT_DEADZONE_X_PX = 15.0
# 主车绕行修正纵向误差死区, 单位为像素。
MASTER_ORBIT_DEADZONE_Y_PX = 8.0
# 主车绕行修正横向速度限幅。
MASTER_ORBIT_MAX_VX = 5.0
# 主车绕行修正纵向速度限幅。
MASTER_ORBIT_MAX_VY = 5.0
# 主车搜索目标点横向像素坐标。当前图像为 QVGA 320x240, 默认中线 x=160; 若修改图像宽度请同步调整。
MASTER_SEARCH_TARGET_X_PX = 160.0
# 主车搜索目标点纵向像素坐标。当前图像为 QVGA 320x240, 默认底边 y=240; 若修改图像高度请同步调整。
MASTER_SEARCH_TARGET_Y_PX = 210.0
# 主车绕行修正目标点横向像素坐标。
MASTER_ORBIT_TARGET_X_PX = 160.0
# 主车绕行修正目标点纵向像素坐标。
MASTER_ORBIT_TARGET_Y_PX = 210.0
# 主车搬运入口目标点纵向像素坐标。当前图像为 QVGA 320x240, 推行前对正使用底边 y=240。
MASTER_TRANSPORT_TARGET_Y_PX = 240.0
# 主车收尾判定环带外扩像素。
FINISH_HOOK_RING_EXPAND_PX = 5
# 主车收尾判定黄色占比阈值。
FINISH_HOOK_YELLOW_RATIO_THRESHOLD = 0.2
FINISH_HOOK_STABLE_FRAMES = 2
# 车端协议中的主车搜索状态编号。
STATE_SEARCH_OBJECT = 1
# 车端协议中的主车绕行状态编号。
STATE_ORBITING = 2
# 车端协议中的主车搬运状态编号。
STATE_TRANSPORT_OBJECT = 4
# 车端协议中的主车回库后退状态编号。
STATE_RETURN_GARAGE_RETREAT = 6
# 车端协议中的主车回库黄线平移状态编号。
STATE_RETURN_GARAGE_LINE = 7
# 车端协议中的物体目标编号。
TARGET_OBJECT = 1
# 车端协议中的边线目标编号。
TARGET_EDGE_LINE = 3
# 车端下发的主车搜索 hook 配置编号。
MASTER_SEARCH_HOOK_CONFIG_ID = 1
# 车端下发的主车搬运 hook 配置编号。
MASTER_TRANSPORT_HOOK_CONFIG_ID = 2
# 车端下发的主车收尾判定 hook 配置编号。
MASTER_TRANSPORT_FINISH_HOOK_CONFIG_ID = 3
# 车端下发的主车绕行视觉修正配置编号。
MASTER_ORBIT_HOOK_CONFIG_ID = 4
# 车端下发的主车回库黄线配置编号。
MASTER_RETURN_GARAGE_LINE_HOOK_CONFIG_ID = 5
# 车端协议中的目标找到事件编号。
EVENT_TARGET_FOUND = 6
# 车端协议中的对正完成事件编号。
EVENT_ALIGNED = 7
# 车端协议中的收尾到达事件编号。
EVENT_ARRIVED = 8
# 车端协议中的回库黄线对正事件编号。
EVENT_RETURN_LINE_ALIGNED = 10
# 车端协议中的回库完成事件编号。
EVENT_RETURN_GARAGE_FINISHED = 12
# 可靠包序号的环形范围大小。
SEQ_RING_SIZE = 256
# 判断环形序号新旧关系使用的半环长度。
SEQ_HALF_RING = 128

# ChromaForge 导出的色块合并间距。
OBJECT_BLOB_MERGE_MARGIN = 0
# ChromaForge 导出的最小识别色块面积。
OBJECT_BLOB_PIXELS_THRESHOLD = 200
# ChromaForge 导出的最小识别目标面积。
OBJECT_BLOB_AREA_THRESHOLD = 200
# 红色沙包候选目标的颜色阈值，格式为 OpenART LAB 阈值。
TASKS = (
    ('brown', ((15, 37, -11, 20, 8, 31),), 3, 10, 50, 80, False),
    ('red', ((16, 39, 21, 60, 0, 49),), 3, 10, 15, 60, True),
    ('green', ((29, 89, -54, -29, 2, 84),), 3, 10, 15, 60, True),
    ('blue', ((32, 57, -11, 12, -50, -23),), 3, 10, 20, 60, True),
    ('white', ((58, 70, -11, 9, -11, 9),), 3, 10, 30, 80, True),
)
# 收尾判定使用的黄色阈值，格式为 OpenART LAB 阈值。
FINISH_HOOK_YELLOW_THRESHOLD = (46, 75, -32, -1, 19, 70)
# 回库黄线使用的黄色阈值，格式为 OpenART LAB 阈值。
RETURN_GARAGE_LINE_YELLOW_THRESHOLD = (0, 100, -40, 10, 20, 127)
# 回库黄线采样半宽, 单位像素。
RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX = 5
# 回库黄线目标 Y 坐标。
RETURN_GARAGE_LINE_TARGET_Y_PX = 220.0
# 回库黄线 Y 死区, 单位像素。
RETURN_GARAGE_LINE_DEADZONE_Y_PX = 4.0
# 回库黄线对正容差, 单位像素。
RETURN_GARAGE_LINE_ALIGN_TOLERANCE_PX = 4.0
# 回库黄线纵向速度 P 环增益。
RETURN_GARAGE_LINE_KP_Y = -0.05
# 回库黄线纵向速度限幅。
RETURN_GARAGE_LINE_MAX_VY = 5.0
# 回库黄线纵向最小有效速度。
RETURN_GARAGE_LINE_MIN_SPEED = 0.0
# 回库黄线参与中心计算的最大厚度, 单位像素。
RETURN_GARAGE_LINE_MAX_THICKNESS_PX = 30
# 回库黄线候选点左右水平联通黄线的最小合计长度, 单位像素。
RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50
# 主车物体识别调试绘制总开关。
MASTER_OBJECT_DEBUG_DRAW_ENABLED = False

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


def encode_master_vision_hook_sync_body(context_id, state, target, arg):
    """! @brief 编码主车视觉同步 body"""

    return bytes([_require_u8(context_id), _require_u8(state), _require_u8(target)]) + _pack_i16(arg)


def decode_master_vision_hook_sync_body(body):
    """! @brief 解码主车视觉同步 body"""

    return {
        "context_id": int(body[0]),
        "state": int(body[1]),
        "target": int(body[2]),
        "arg": _unpack_i16(body, 3),
    }


def encode_master_vision_event_report_body(context_id, event, value):
    """! @brief 编码主车视觉事件回报 body"""

    return bytes([_require_u8(context_id), _require_u8(event)]) + _pack_i16(value)


def decode_master_vision_event_report_body(body):
    """! @brief 解码主车视觉事件回报 body"""

    return {
        "context_id": int(body[0]),
        "event": int(body[1]),
        "value": _unpack_i16(body, 2),
    }


def parse_sync_packet(frame_bytes):
    """! @brief 解析 RT1021 下发的视觉上下文同步帧

    @param frame_bytes 原始固定帧
    @return 同步包字段字典, 输入无效时返回 None
    """

    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != MODE_TCP or frame["topic"] != TOPIC_MASTER_VISION_HOOK_SYNC:
        return None
    packet = decode_master_vision_hook_sync_body(frame["body"])
    return {
        "reliable_seq": int(frame["seq"]),
        "context_id": int(packet["context_id"]),
        "state": int(packet["state"]),
        "target": int(packet["target"]),
        "arg": int(packet["arg"]),
    }


def parse_ack_packet(frame_bytes):
    """! @brief 解析可靠事件确认帧

    @param frame_bytes 原始固定帧
    @return 确认包字段字典, 输入无效时返回 None
    """

    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != MODE_ACK or frame["topic"] != TOPIC_MASTER_VISION_EVENT_REPORT:
        return None
    return {"reliable_seq": int(frame["seq"])}


def is_newer_seq(seq, last_seq):
    """! @brief 按 0..255 半环规则判断序号是否更新

    @param seq 待判断序号
    @param last_seq 已记录序号, None 表示没有历史序号
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
    @param last_sent_ms 上次发送毫秒时间戳, None 表示尚未发送
    @param interval_ms 重发间隔, 单位毫秒
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


def format_ack_frame(reliable_seq):
    """! @brief 格式化可靠包确认帧

    @param reliable_seq 被确认的可靠包序号
    @return ACK 短帧
    """

    return encode_frame(MODE_ACK, TOPIC_MASTER_VISION_HOOK_SYNC, reliable_seq, b"")


def format_search_velocity_frame(vx, vy):
    """! @brief 格式化主车搜索速度数据流帧

    @param vx 主车搜索横向速度控制量
    @param vy 主车搜索纵向速度控制量
    @return 速度短帧
    """

    return encode_frame(
        MODE_UDP,
        TOPIC_LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_event_frame(reliable_seq, context_id, event, value):
    """! @brief 格式化可靠事件回报帧

    @param reliable_seq 可靠包序号
    @param context_id 视觉上下文编号
    @param event 事件编号
    @param value 事件附加值
    @return 事件短帧
    """

    return encode_frame(
        MODE_TCP,
        TOPIC_MASTER_VISION_EVENT_REPORT,
        reliable_seq,
        encode_master_vision_event_report_body(context_id, event, value),
    )


def _write_all(uart, line):
    """! @brief 向串口完整写出一帧 bytes

    @param uart 目标串口对象
    @param line 固定长度短帧
    @return 是否完整写出整帧
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
            return False
        remaining = remaining[written:]
    return True


def write_data_line(uart, line):
    """! @brief 写出速度数据流短帧

    @param uart 目标串口对象
    @param line 固定长度速度短帧
    """

    _write_all(uart, line)


def write_reliable_line(uart, line):
    """! @brief 写出可靠短帧

    @param uart 目标串口对象
    @param line 固定长度可靠短帧
    @return 是否完整写出整帧
    """

    return _write_all(uart, line)


def _find_control_frame_start(rx_buffer):
    """! @brief 查找视觉控制链路合法帧起点"""

    limit = len(rx_buffer) - FRAME_SIZE + 1
    for index in range(limit):
        frame = decode_frame(rx_buffer[index : index + FRAME_SIZE])
        if frame is None:
            continue
        mode = frame["mode"]
        topic = frame["topic"]
        if mode == MODE_TCP and topic == TOPIC_MASTER_VISION_HOOK_SYNC:
            return index
        if mode == MODE_ACK and topic == TOPIC_MASTER_VISION_EVENT_REPORT:
            return index
    return -1


def blob_rect_to_bbox(rect):
    """! @brief 将 OpenMV 的 x,y,w,h 矩形转换为边界框

    @param rect blob.rect() 返回的矩形元组
    @return left, top, right, bottom 边界框
    """

    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    """! @brief 将色块边界框转换到主车搜索使用的位置坐标"""

    normalized_top = img_height - bottom
    normalized_bottom = img_height - top
    return left, normalized_top, right, normalized_bottom


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


def object_task_id(task_name):
    """! @brief 根据任务名称返回物体编号"""

    index = 1
    for task in TASKS:
        if task[0] == task_name:
            return index
        index += 1
    return 0


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
    """! @brief 提取物体候选目标, 面积作为目标强度

    @param img 当前图像对象
    @return 候选目标列表, 元素格式为 task_name, cx, bottom, area, blob
    """

    candidates = []
    for task in TASKS:
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
        if not blobs:
            continue
        img_height = img.height()
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
            candidates.append((task_name, blob.cx(), bottom, blob_area(blob), blob))
    return candidates


def choose_best_candidate(candidates, target_x, target_y):
    """! @brief 选择最靠近 hook 目标点的候选物体

    @param candidates 候选目标列表
    @param target_x hook 目标点 x 坐标
    @param target_y hook 目标点 y 坐标
    @return 被选中的候选目标
    """

    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[2]) - float(target_y)) ** 2,
    )


def build_search_target_point(
    image_width, image_height, config_id=MASTER_SEARCH_HOOK_CONFIG_ID
):
    """! @brief 根据当前 hook 配置生成主车搜索目标点"""

    _ = image_width
    _ = image_height
    target_x = float(MASTER_SEARCH_TARGET_X_PX)
    if int(config_id) == int(MASTER_ORBIT_HOOK_CONFIG_ID):
        return float(MASTER_ORBIT_TARGET_X_PX), float(MASTER_ORBIT_TARGET_Y_PX)
    if int(config_id) in (
        int(MASTER_TRANSPORT_HOOK_CONFIG_ID),
        int(MASTER_TRANSPORT_FINISH_HOOK_CONFIG_ID),
    ):
        return target_x, float(MASTER_TRANSPORT_TARGET_Y_PX)
    return target_x, float(MASTER_SEARCH_TARGET_Y_PX)


def _build_finish_hook_ring_rois(blob, image_width, image_height):
    """! @brief 根据目标框构造收尾判定环带 roi 列表和总面积"""

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
        rois.append(
            (outer_left, int(bottom), outer_right - outer_left, outer_bottom - int(bottom))
        )
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
    """! @brief 统计单个 roi 内的黄色像素数"""

    roi_width = int(roi[2])
    roi_height = int(roi[3])
    roi_area = roi_width * roi_height
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


def build_finish_hook_yellow_ratio_percent(img, blob, image_width, image_height):
    """! @brief 计算主车收尾判定环带内的黄色像素百分比"""

    if blob is None:
        return 0.0
    rois, ring_area = _build_finish_hook_ring_rois(blob, image_width, image_height)
    if ring_area <= 0:
        return 0.0
    yellow_pixels = 0
    for roi in rois:
        yellow_pixels += _count_yellow_pixels_in_roi(img, roi)
    return float(yellow_pixels) * 100.0 / float(ring_area)


def get_marker_corners(blob):
    """! @brief 返回调试绘制使用的色块矩形角点

    @param blob 候选色块对象
    @return 色块矩形四个角点
    """

    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def get_blob_rect(blob):
    """! @brief 返回候选色块的标准矩形框"""

    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return (left, top, right - left, bottom - top)


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    """! @brief 在调试画面上绘制选中物体矩形框

    @param img 当前图像对象
    @param blob 被选中的候选色块对象
    @param pixel_x 目标中心 x 坐标
    @param pixel_y 目标标记 y 坐标
    """

    _ = pixel_x
    _ = pixel_y
    _draw_debug_rectangle(img, get_blob_rect(blob))


def draw_blob_candidates_debug(img, candidates):
    """! @brief 在调试画面上绘制当前识别到的全部物体

    @param img 当前图像对象
    @param candidates 当前帧候选目标列表
    """

    for task_name, pixel_x, _, _, blob in candidates:
        color = _debug_color_for_task_name(task_name)
        _ = pixel_x
        _draw_debug_rectangle_with_color(img, get_blob_rect(blob), color)
        _draw_debug_text_with_color(img, pixel_x + 4, blob.cy() - 6, task_name, color)


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


def _apply_min_speed(value, limit, min_speed):
    """! @brief 对非零速度量施加最小幅值和限幅

    @param value 原始速度量
    @param limit 速度幅值上限
    @param min_speed 最小速度幅值
    @return 处理后的速度量
    """

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


def _axis_p_velocity(error, deadzone, kp, limit, min_speed=MASTER_SEARCH_MIN_SPEED):
    """! @brief 生成单轴 P 控制速度

    @param error 当前轴像素误差
    @param deadzone 当前轴死区
    @param kp 当前轴 P 环增益
    @param limit 当前轴速度限幅
    @param min_speed 当前轴最小有效速度
    @return 当前轴速度控制量
    """

    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    return _apply_min_speed(
        error * float(kp),
        limit,
        min_speed,
    )


def _build_search_y_velocity(err_y, image_height):
    """! @brief 根据图像高度归一化纵向底边误差并生成速度"""

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


def build_search_velocity_from_error(err_x, err_y, image_height):
    """! @brief 根据主车搜索目标点误差生成速度控制量

    @param err_x 识别框中心相对目标点的横向误差
    @param err_y 识别框中心相对目标点的纵向误差
    @param image_height 图像高度
    @return vx, vy 速度控制量
    """

    return (
        _axis_p_velocity(
            err_x,
            MASTER_SEARCH_DEADZONE_X_PX,
            MASTER_SEARCH_KP_X,
            MASTER_SEARCH_MAX_VX,
        ),
        _build_search_y_velocity(err_y, image_height),
    )


def build_search_velocity_from_observation(observation, image_height):
    """! @brief 根据主车物体观测生成搜索速度控制量

    @param observation context_id, x, y, value 观测字段元组
    @param image_height 图像高度
    @return vx, vy 速度控制量
    """

    _, x, y, value = observation
    if float(value) <= 0.0:
        return float(MASTER_MISSING_SEARCH_VX), float(MASTER_MISSING_SEARCH_VY)
    return build_search_velocity_from_error(x, y, image_height)


def build_orbit_correction_velocity_from_observation(observation, image_height):
    """! @brief 根据主车绕行物体观测生成平移修正量

    @param observation context_id, x, y, value 观测字段元组
    @param image_height 图像高度
    @return vx, vy 平移修正量
    """

    _, _, _, value = observation
    if float(value) <= 0.0:
        return 0.0, 0.0
    _, x, y, _ = observation
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


def _pixel_to_lab(pixel):
    """! @brief 将单点像素转换为 LAB 阈值比较用三元组"""

    if pixel is None:
        return None
    if omv_image is not None:
        try:
            return omv_image.rgb_to_lab(pixel)
        except Exception:
            pass
    return pixel


def _pixel_matches_threshold(pixel, threshold):
    """! @brief 判断单点像素是否落在 LAB 阈值内"""

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
        RETURN_GARAGE_LINE_YELLOW_THRESHOLD,
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
    max_thickness = int(RETURN_GARAGE_LINE_MAX_THICKNESS_PX)
    if max_thickness > 0 and int(bottom) - int(top) > max_thickness:
        top = int(bottom) - max_thickness
    return (float(top) + float(bottom)) / 2.0


def _build_return_line_y_from_pixels(img, image_width, image_height, previous_line_y=None):
    """! @brief 在中心采样区逐像素按黄色阈值计算回库黄线中心 Y"""

    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None:
        return None
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
        has_horizontal_connected = _return_line_has_horizontal_connected_at(
            img,
            x,
            int(round(line_y)),
            image_width,
            image_height,
            RETURN_GARAGE_LINE_MIN_HORIZONTAL_CONNECTED_PX,
        )
        if has_horizontal_connected:
            return line_y
    if saw_candidate:
        return previous_line_y
    return None


def build_return_line_y_from_image(img, image_width, image_height, previous_line_y=None):
    """! @brief 按屏幕中线左右色块范围计算回库黄线中心 Y"""

    return _build_return_line_y_from_pixels(
        img, image_width, image_height, previous_line_y
    )


def build_return_line_velocity_from_y(line_y):
    """! @brief 根据回库黄线 Y 生成纵向保持速度"""

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


def is_return_line_aligned(line_y):
    """! @brief 判断回库黄线 Y 是否进入目标容差"""

    if line_y is None:
        return False
    return abs(float(line_y) - float(RETURN_GARAGE_LINE_TARGET_Y_PX)) <= float(
        RETURN_GARAGE_LINE_ALIGN_TOLERANCE_PX
    )


def _build_orbit_y_velocity(err_y, image_height):
    """! @brief 根据图像高度归一化绕行纵向误差并生成速度"""

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


class MasterVisionHook:
    """! @brief 主车物体搜索视觉 hook 状态

    @details 维护视觉上下文、稳定计数、可靠事件和重发节奏
    """

    def __init__(
        self,
        min_area=OBJECT_MIN_AREA,
        tolerance_x=OBJECT_X_TOLERANCE_PX,
        tolerance_y=OBJECT_Y_TOLERANCE_PX,
        stable_frames=OBJECT_STABLE_FRAMES,
        next_reliable_seq=1,
        now_ms=None,
        event_resend_interval_ms=RELIABLE_RESEND_INTERVAL_MS,
    ):
        """! @brief 初始化主车视觉 hook 状态

        @param min_area 目标面积下限
        @param tolerance_x 横向误差容差, 单位像素
        @param tolerance_y 纵向误差容差, 单位像素
        @param stable_frames hook 命中所需连续稳定帧数
        @param next_reliable_seq 可靠事件序号初始值
        @param now_ms 毫秒时钟函数, None 时使用默认时钟
        @param event_resend_interval_ms 可靠事件重发间隔, 单位毫秒
        """

        self.min_area = float(min_area)
        self.tolerance_x = float(tolerance_x)
        self.tolerance_y = float(tolerance_y)
        self.required_stable_frames = int(stable_frames)
        self._next_reliable_seq = int(next_reliable_seq) % SEQ_RING_SIZE
        self._now_ms = now_ms or default_now_ms
        self._event_resend_interval_ms = int(event_resend_interval_ms)
        self.context = None
        self._last_context_id = None
        self._stable_count = 0
        self._pending_event = None
        self._pending_event_last_sent_ms = None
        self._event_context_id = None
        self._finish_contact_seen = False
        self._last_return_line_y = None

    def has_context(self):
        """! @brief 判断是否已经建立视觉上下文

        @return 是否存在有效视觉上下文
        """

        return self.context is not None

    def has_pending_event(self):
        """! @brief 判断是否存在等待确认的可靠事件"""

        return self._pending_event is not None

    def current_target_config_id(self):
        """! @brief 返回当前 hook 使用的目标点配置编号"""

        if self.context is None:
            return MASTER_SEARCH_HOOK_CONFIG_ID
        return int(self.context["arg"])

    def is_finish_hook_context(self):
        """! @brief 判断当前上下文是否为收尾判定 hook"""

        if self.context is None:
            return False
        return (
            int(self.context["state"]) == int(STATE_TRANSPORT_OBJECT)
            and int(self.context["target"]) == int(TARGET_EDGE_LINE)
            and int(self.context["arg"]) == int(MASTER_TRANSPORT_FINISH_HOOK_CONFIG_ID)
        )

    def is_orbit_correction_context(self):
        """! @brief 判断当前上下文是否为绕行视觉修正 hook"""

        if self.context is None:
            return False
        return (
            int(self.context["state"]) == int(STATE_ORBITING)
            and int(self.context["target"]) == int(TARGET_OBJECT)
            and int(self.context["arg"]) == int(MASTER_ORBIT_HOOK_CONFIG_ID)
        )

    def is_return_line_context(self):
        """! @brief 判断当前上下文是否为回库黄线 hook"""

        if self.context is None:
            return False
        return (
            int(self.context["target"]) == int(TARGET_EDGE_LINE)
            and int(self.context["arg"]) == int(MASTER_RETURN_GARAGE_LINE_HOOK_CONFIG_ID)
            and (
                int(self.context["state"]) == int(STATE_RETURN_GARAGE_RETREAT)
                or int(self.context["state"]) == int(STATE_RETURN_GARAGE_LINE)
            )
        )

    def last_return_line_y(self):
        """! @brief 返回上一帧有效回库黄线 Y"""

        return self._last_return_line_y

    def remember_return_line_y(self, line_y):
        """! @brief 保存当前有效回库黄线 Y"""

        if line_y is not None:
            self._last_return_line_y = float(line_y)

    def handle_control_line(self, line):
        """! @brief 处理 RT1021 发来的同步或确认短帧

        @param line 原始控制短帧
        @return 需要回复的 ACK 帧, 无需回复时返回 None
        """

        sync_packet = parse_sync_packet(line)
        if sync_packet is not None:
            return self._handle_sync_packet(sync_packet)
        ack_packet = parse_ack_packet(line)
        if ack_packet is not None:
            self._handle_ack_packet(ack_packet)
        return None

    def _handle_sync_packet(self, packet):
        """! @brief 处理视觉上下文同步包

        @param packet 解析后的同步包字段
        @return 同步确认帧文本
        """

        context_id = int(packet["context_id"])
        if self._should_apply_context(context_id):
            self.context = {
                "context_id": context_id,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            self._last_context_id = context_id
            self._stable_count = 0
            self._event_context_id = None
            self._finish_contact_seen = False
            self._last_return_line_y = None
        return format_ack_frame(packet["reliable_seq"])

    def _should_apply_context(self, context_id):
        """! @brief 判断上下文编号是否应覆盖当前上下文

        @param context_id 待判断的上下文编号
        @return 是否应用该上下文
        """

        if self._last_context_id is None:
            return True
        if int(context_id) == int(self._last_context_id):
            return False
        return is_newer_seq(context_id, self._last_context_id)

    def _handle_ack_packet(self, packet):
        """! @brief 处理可靠事件确认包

        @param packet 解析后的确认包字段
        """

        if self._pending_event is None:
            return
        if int(packet["reliable_seq"]) == int(self._pending_event["reliable_seq"]):
            self._pending_event = None
            self._pending_event_last_sent_ms = None

    def build_observation(
        self, valid, center_x, bottom_y, area, image_width, image_height
    ):
        """! @brief 根据物体中心、底边和面积生成观测字段

        @param valid 当前帧是否存在有效目标
        @param center_x 目标中心 x 坐标
        @param bottom_y 目标底边 y 坐标
        @param area 目标面积
        @param image_width 图像宽度
        @param image_height 图像高度
        @return context_id, x, y, value 观测字段元组
        """

        context_id = 0
        if self.context is not None:
            context_id = int(self.context["context_id"])
        if int(valid) != 1:
            return context_id, 0.0, 0.0, 0.0
        target_x, target_y = build_search_target_point(
            image_width,
            image_height,
            self.current_target_config_id(),
        )
        return (
            context_id,
            float(center_x) - target_x,
            float(bottom_y) - target_y,
            float(area),
        )

    def build_return_line_observation(self, line_y):
        """! @brief 根据回库黄线 Y 生成 hook 观测"""

        context_id = 0
        if self.context is not None:
            context_id = int(self.context["context_id"])
        if line_y is None:
            return context_id, 0.0, 0.0, 0.0
        return (
            context_id,
            0.0,
            float(line_y) - float(RETURN_GARAGE_LINE_TARGET_Y_PX),
            float(line_y),
        )

    def accept_observation(self, observation, hook_value=None):
        """! @brief 累计 hook 条件并按需创建可靠事件

        @param observation 观测字段元组
        @param hook_value 当前 hook 使用的附加判定值
        """

        if self.context is None:
            return
        context_id = int(self.context["context_id"])
        observed_context_id, x, y, value = observation
        if int(observed_context_id) != context_id:
            return
        event_type = self._resolve_event_type()
        if event_type is None:
            self._stable_count = 0
            return
        if self._pending_event is not None or self._event_context_id == context_id:
            return
        if self.is_finish_hook_context():
            self._accept_finish_hook_observation(context_id, value, hook_value, event_type)
            return
        if self.is_return_line_context():
            self._accept_return_line_observation(
                context_id, x, y, value, hook_value, event_type
            )
            return
        if self._observation_matches_hook(x, y, value, hook_value):
            self._stable_count += 1
        else:
            self._stable_count = 0
            return
        if self._stable_count >= self._required_stable_frames():
            self._create_event(
                context_id,
                self._resolve_event_value(value, hook_value),
                event_type,
            )

    def _accept_return_line_observation(
        self, context_id, x, y, value, hook_value, event_type
    ):
        """! @brief 处理回库黄线配置下的对正与丢线完成事件"""

        if event_type == EVENT_RETURN_LINE_ALIGNED:
            if (
                float(value) > 0.0
                and float(value) <= float(RETURN_GARAGE_LINE_TARGET_Y_PX)
            ):
                self._stable_count += 1
            else:
                self._stable_count = 0
                return
        elif event_type == EVENT_RETURN_GARAGE_FINISHED:
            self._stable_count = 0
            return
        else:
            return
        required_stable_frames = self._required_stable_frames()
        if event_type == EVENT_RETURN_GARAGE_FINISHED:
            required_stable_frames = 5
        if self._stable_count >= required_stable_frames:
            self._create_event(
                context_id,
                self._resolve_event_value(value, None),
                event_type,
            )

    def _accept_finish_hook_observation(self, context_id, value, hook_value, event_type):
        """! @brief 按“未接触 -> 接触 -> 再次脱离”过程处理收尾 hook"""

        if float(value) <= 0.0:
            self._stable_count = 0
            return
        yellow_ratio = float(hook_value or 0.0)
        if not self._finish_contact_seen:
            if yellow_ratio > (float(FINISH_HOOK_YELLOW_RATIO_THRESHOLD) * 100.0):
                self._finish_contact_seen = True
            self._stable_count = 0
            return
        if yellow_ratio <= 0.0:
            self._stable_count += 1
        else:
            self._stable_count = 0
            return
        if self._stable_count >= self._required_stable_frames():
            self._create_event(
                context_id,
                self._resolve_event_value(value, hook_value),
                event_type,
            )

    def _observation_matches_hook(self, x, y, value, hook_value):
        """! @brief 判断单帧观测是否满足 hook 条件

        @param x 横向误差
        @param y 纵向误差
        @param value 目标强度
        @param hook_value 当前 hook 使用的附加判定值
        @return 观测是否满足当前 hook 条件
        """

        return (
            float(value) >= self.min_area
            and abs(float(x)) <= self.tolerance_x
            and abs(float(y)) <= self.tolerance_y
        )

    def _required_stable_frames(self):
        """! @brief 返回当前上下文所需的连续稳定帧数"""

        if self.is_finish_hook_context():
            return int(FINISH_HOOK_STABLE_FRAMES)
        return self.required_stable_frames

    def _resolve_event_value(self, value, hook_value):
        """! @brief 返回当前事件应携带的附加值"""

        if self.is_finish_hook_context():
            return int(float(hook_value or 0.0))
        if self.is_return_line_context():
            return int(float(value))
        if (
            self.context is not None
            and int(self.context["state"]) == STATE_SEARCH_OBJECT
            and int(self.context["target"]) == TARGET_OBJECT
            and int(self.context["arg"]) == MASTER_SEARCH_HOOK_CONFIG_ID
            and hook_value is not None
        ):
            return int(hook_value)
        return int(float(value))

    def _resolve_event_type(self):
        """! @brief 根据当前上下文解析应回报的事件类型

        @return 事件编号, 不支持的上下文返回 None
        """

        if self.context is None:
            return None
        state = int(self.context["state"])
        target = int(self.context["target"])
        arg = int(self.context["arg"])
        if (
            state == STATE_SEARCH_OBJECT
            and target == TARGET_OBJECT
            and arg == MASTER_SEARCH_HOOK_CONFIG_ID
        ):
            return EVENT_TARGET_FOUND
        if (
            state == STATE_SEARCH_OBJECT
            and target == TARGET_OBJECT
            and arg == MASTER_TRANSPORT_HOOK_CONFIG_ID
        ):
            return EVENT_ALIGNED
        if (
            state == STATE_ORBITING
            and target == TARGET_OBJECT
            and arg == MASTER_ORBIT_HOOK_CONFIG_ID
        ):
            return None
        if (
            state == STATE_TRANSPORT_OBJECT
            and target == TARGET_EDGE_LINE
            and arg == MASTER_TRANSPORT_FINISH_HOOK_CONFIG_ID
        ):
            return EVENT_ARRIVED
        if (
            state == STATE_RETURN_GARAGE_RETREAT
            and target == TARGET_EDGE_LINE
            and arg == MASTER_RETURN_GARAGE_LINE_HOOK_CONFIG_ID
        ):
            return EVENT_RETURN_LINE_ALIGNED
        if (
            state == STATE_RETURN_GARAGE_LINE
            and target == TARGET_EDGE_LINE
            and arg == MASTER_RETURN_GARAGE_LINE_HOOK_CONFIG_ID
        ):
            return EVENT_RETURN_GARAGE_FINISHED
        return None

    def _allocate_reliable_seq(self):
        """! @brief 分配新的可靠事件序号

        @return 新的可靠事件序号
        """

        reliable_seq = self._next_reliable_seq
        self._next_reliable_seq = (self._next_reliable_seq + 1) % SEQ_RING_SIZE
        return reliable_seq

    def _create_event(self, context_id, value, event):
        """! @brief 创建待确认事件

        @param context_id 视觉上下文编号
        @param value 事件附加值
        @param event 事件编号
        """

        reliable_seq = self._allocate_reliable_seq()
        self._pending_event = {
            "reliable_seq": reliable_seq,
            "context_id": int(context_id),
            "event": int(event),
            "value": int(float(value)),
        }
        self._pending_event_last_sent_ms = None
        self._event_context_id = int(context_id)

    def next_event_frame(self):
        """! @brief 返回待确认事件帧, 没有事件时返回 None

        @return 事件帧文本, 未到发送时机或无待确认事件时返回 None
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
            self._pending_event["context_id"],
            self._pending_event["event"],
            self._pending_event["value"],
        )


def init_uart():
    """! @brief 初始化主通信串口

    @return 已配置的 UART 对象
    @raises RuntimeError 主机环境缺少 UART 时抛出
    """

    if UART is None:
        raise RuntimeError("UART unavailable in host environment")
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    """! @brief 初始化 OpenART 摄像头参数

    @return image_width, image_height 图像尺寸
    @raises RuntimeError 主机环境缺少 sensor 时抛出
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


def process_uart_input(uart, rx_buffer, hook):
    """! @brief 处理 RT1021 发来的控制短帧

    @param uart 控制链路串口对象
    @param rx_buffer 上一轮遗留的未完整输入 bytes
    @param hook 主车视觉 hook 状态对象
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
        reply = hook.handle_control_line(line)
        if reply is not None:
            write_reliable_line(uart, reply)
    return rx_buffer


def build_observation_from_image(hook, img, image_width, image_height):
    """! @brief 从图像生成一帧物体观测

    @param hook 主车视觉 hook 状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    @return observation, best_blob 元组
    """

    observation, best_blob, _, _ = build_observation_and_candidates_from_image(
        hook,
        img,
        image_width,
        image_height,
    )
    return observation, best_blob


def build_observation_and_candidates_from_image(hook, img, image_width, image_height):
    """! @brief 从图像生成一帧物体观测并返回全部候选色块"""

    candidates = build_blob_candidates(img)
    if not candidates:
        return (
            hook.build_observation(0, 0, 0, 0, image_width, image_height),
            None,
            None,
            candidates,
        )
    target_x, target_y = build_search_target_point(
        image_width,
        image_height,
        hook.current_target_config_id(),
    )
    task_name, pixel_x, bottom_y, area, best_blob = choose_best_candidate(
        candidates, target_x, target_y
    )
    return (
        hook.build_observation(1, pixel_x, bottom_y, area, image_width, image_height),
        best_blob,
        task_name,
        candidates,
    )


def build_hook_event_value(hook, img, best_blob, image_width, image_height, task_name=None):
    """! @brief 根据当前 hook 生成事件附加判定值"""

    if not hook.is_finish_hook_context():
        if (
            hook.context is not None
            and int(hook.context["state"]) == STATE_SEARCH_OBJECT
            and int(hook.context["target"]) == TARGET_OBJECT
            and int(hook.context["arg"]) == MASTER_SEARCH_HOOK_CONFIG_ID
            and task_name is not None
        ):
            return object_task_id(task_name)
        return None
    return build_finish_hook_yellow_ratio_percent(
        img,
        best_blob,
        image_width,
        image_height,
    )


def _process_return_line_frame(uart, hook, img, image_width, image_height):
    """! @brief 处理回库黄线配置的一帧输出"""

    line_y = build_return_line_y_from_image(
        img,
        image_width,
        image_height,
        hook.last_return_line_y(),
    )
    hook.remember_return_line_y(line_y)
    velocity = build_return_line_velocity_from_y(line_y)
    write_data_line(uart, format_search_velocity_frame(*velocity))
    if int(hook.context["state"]) == int(STATE_RETURN_GARAGE_RETREAT):
        hook.accept_observation(hook.build_return_line_observation(line_y))
        return
    if line_y is None:
        hook.accept_observation(hook.build_return_line_observation(None), 1.0)
        return
    hook.accept_observation(hook.build_return_line_observation(line_y))


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


def _draw_debug_rectangle(img, rect):
    draw_rectangle = getattr(img, "draw_rectangle", None)
    if draw_rectangle is None:
        return
    x, y, w, h = rect
    try:
        draw_rectangle(int(x), int(y), int(w), int(h))
    except TypeError:
        draw_rectangle((int(x), int(y), int(w), int(h)))


def _draw_debug_rectangle_with_color(img, rect, color):
    draw_rectangle = getattr(img, "draw_rectangle", None)
    if draw_rectangle is None:
        return
    x, y, w, h = rect
    try:
        draw_rectangle(int(x), int(y), int(w), int(h), color=color)
    except TypeError:
        try:
            draw_rectangle((int(x), int(y), int(w), int(h)), color=color)
        except TypeError:
            draw_rectangle((int(x), int(y), int(w), int(h)))


def _draw_debug_text(img, x, y, text):
    draw_string = getattr(img, "draw_string", None)
    if draw_string is None:
        return
    try:
        draw_string(int(x), int(y), str(text), color=(255, 255, 255))
    except TypeError:
        draw_string(int(x), int(y), str(text))


def _draw_debug_text_with_color(img, x, y, text, color):
    draw_string = getattr(img, "draw_string", None)
    if draw_string is None:
        return
    try:
        draw_string(int(x), int(y), str(text), color=color)
    except TypeError:
        draw_string(int(x), int(y), str(text))


def _debug_color_for_task_name(task_name):
    for task in TASKS:
        task_name_in_config, thresholds, _, _, _, _, _ = object_task_parts(task)
        if task_name_in_config != task_name:
            continue
        first_threshold = task_thresholds(thresholds)[0]
        lab_center = (
            int((int(first_threshold[0]) + int(first_threshold[1])) / 2),
            int((int(first_threshold[2]) + int(first_threshold[3])) / 2),
            int((int(first_threshold[4]) + int(first_threshold[5])) / 2),
        )
        if omv_image is not None:
            try:
                return tuple(int(value) for value in omv_image.lab_to_rgb(lab_center))
            except Exception:
                break
    return (255, 255, 255)


def draw_master_return_line_debug(img, image_width, image_height, line_y, vx, vy):
    """! @brief 绘制主车回库黄线判定调试信息"""

    center_x = int(float(image_width) / 2.0)
    half_width = int(RETURN_GARAGE_LINE_SAMPLE_HALF_WIDTH_PX)
    max_thickness = int(RETURN_GARAGE_LINE_MAX_THICKNESS_PX)
    target_y = int(RETURN_GARAGE_LINE_TARGET_Y_PX)
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

    _draw_debug_text(img, 2, 2, "master return line debug")
    _draw_debug_text(img, 2, 14, "max h=%d x=%d..%d" % (
        max_thickness,
        center_x - half_width,
        center_x + half_width,
    ))
    _draw_debug_text(img, 2, 26, "%s found=%d" % (line_text, 1 if line_y is not None else 0))
    _draw_debug_text(img, 2, 38, "vx=%.1f vy=%.1f" % (float(vx), float(vy)))


def run_master_return_line_debug():
    """! @brief 只显示主车回库黄线判定调试画面"""

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
        draw_master_return_line_debug(img, image_width, image_height, line_y, vx, vy)


def process_search_frame(uart, hook, img, image_width, image_height):
    """! @brief 处理单帧主车搜索速度流和 hook 事件

    @param uart 主车视觉串口
    @param hook 主车视觉 hook 状态
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    if hook.has_pending_event():
        event_frame = hook.next_event_frame()
        if event_frame is not None:
            write_reliable_line(uart, event_frame)
        if not hook.is_return_line_context():
            return
    candidates = None
    if MASTER_OBJECT_DEBUG_DRAW_ENABLED and not hook.is_return_line_context():
        candidates = build_blob_candidates(img)
        draw_blob_candidates_debug(img, candidates)
    if not hook.has_context():
        return
    if hook.is_return_line_context():
        _process_return_line_frame(uart, hook, img, image_width, image_height)
        return

    if candidates is None:
        observation, best_blob, selected_task_name, candidates = (
            build_observation_and_candidates_from_image(
                hook,
                img,
                image_width,
                image_height,
            )
        )
    else:
        selected_task_name = None
        target_x, target_y = build_search_target_point(
            image_width,
            image_height,
            hook.current_target_config_id(),
        )
        if not candidates:
            observation = hook.build_observation(0, 0, 0, 0, image_width, image_height)
            best_blob = None
        else:
            selected_task_name, pixel_x, bottom_y, area, best_blob = choose_best_candidate(
                candidates, target_x, target_y
            )
            observation = hook.build_observation(
                1, pixel_x, bottom_y, area, image_width, image_height
            )
    _, x, y, _ = observation
    if MASTER_OBJECT_DEBUG_DRAW_ENABLED and best_blob is not None:
        draw_selected_marker(
            img=img,
            blob=best_blob,
            pixel_x=int(float(x) + image_width / 2.0),
            pixel_y=best_blob.cy(),
        )
    if hook.is_orbit_correction_context():
        velocity = build_orbit_correction_velocity_from_observation(
            observation,
            image_height,
        )
    else:
        velocity = build_search_velocity_from_observation(observation, image_height)
    write_data_line(uart, format_search_velocity_frame(*velocity))
    hook.accept_observation(
        observation,
        hook_value=build_hook_event_value(
            hook,
            img,
            best_blob,
            image_width,
            image_height,
            selected_task_name,
        ),
    )


def apply_lens_correction(img):
    """! @brief 执行当前帧镜头畸变校准"""

    lens_corr = getattr(img, "lens_corr", None)
    if lens_corr is None:
        return
    try:
        lens_corr(strength=2.8, zoom=1.0)
    except Exception:
        return


def run():
    """! @brief 运行主车物体搜索视觉主循环"""

    uart = init_uart()
    image_width, image_height = init_sensor()
    hook = MasterVisionHook()
    rx_buffer = b""

    while True:
        rx_buffer = process_uart_input(uart, rx_buffer, hook)
        img = sensor.snapshot()  # type: ignore
        apply_lens_correction(img)
        process_search_frame(uart, hook, img, image_width, image_height)


if __name__ == "__main__":
    run()
