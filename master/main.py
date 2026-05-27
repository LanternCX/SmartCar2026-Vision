"""! @file main.py
@brief OpenART Vision master 主车物体搜索视觉入口
@details 负责接收 RT1021 视觉上下文, 输出搜索速度, 并在 hook 条件满足时可靠回报事件
"""

import time

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
# 可靠包发送前后的保护延时，单位为秒。
RELIABLE_WRITE_DELAY_S = 0.001
# TARGET_FOUND 未确认时的重复发送间隔，单位为毫秒。
RELIABLE_RESEND_INTERVAL_MS = 100
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
MASTER_SEARCH_MIN_SPEED = 2
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
# 车端协议中的目标找到事件编号。
EVENT_TARGET_FOUND = 6
# 车端协议中的对正完成事件编号。
EVENT_ALIGNED = 7
# 车端协议中的收尾到达事件编号。
EVENT_ARRIVED = 8
# 可靠包序号的环形范围大小。
SEQ_RING_SIZE = 256
# 判断环形序号新旧关系使用的半环长度。
SEQ_HALF_RING = 128

# 红色沙包候选目标的颜色阈值，格式为 OpenART LAB 阈值。
TASKS = (("red", (0, 100, 18, 127, -23, 127)),)
# 收尾判定使用的黄色阈值，格式为 OpenART LAB 阈值。
FINISH_HOOK_YELLOW_THRESHOLD = (0, 100, -40, 10, 20, 127)


def compact_number(value):
    """! @brief 将数值编码为短包中的紧凑文本

    @param value 原始数值
    @return 短包载荷使用的紧凑数值文本
    """

    number = float(value)
    if -0.0005 < number < 0.0005:
        number = 0.0
    text = ("%.3f" % number).rstrip("0").rstrip(".")
    if not text or text == "-0":
        text = "0"
    return text


def _parse_int(text):
    """! @brief 解析严格整数字段

    @param text 数值文本
    @return 整数, 输入无效时返回 None
    """

    try:
        value = int(text)
    except ValueError:
        return None
    if str(value) != text.strip():
        return None
    return value


def _parse_u8(text):
    """! @brief 解析 0..255 范围内的整数字段

    @param text 数值文本
    @return 整数, 输入无效时返回 None
    """

    value = _parse_int(text)
    if value is None or value < 0 or value > 255:
        return None
    return value


def _split_fields(line):
    """! @brief 拆分短包字段并过滤空字段

    @param line 原始输入行
    @return 字段列表, 输入无效时返回 None
    """

    text = str(line).strip()
    if not text:
        return None
    fields = [part.strip() for part in text.split(",")]
    for field in fields:
        if field == "":
            return None
    return fields


def parse_sync_packet(line):
    """! @brief 解析 RT1021 下发的视觉上下文同步包

    @param line 原始输入行
    @return 同步包字段字典, 输入无效时返回 None
    """

    fields = _split_fields(line)
    if fields is None or len(fields) != 6 or fields[0].lower() != "s":
        return None
    reliable_seq = _parse_u8(fields[1])
    context_id = _parse_u8(fields[2])
    state = _parse_u8(fields[3])
    target = _parse_u8(fields[4])
    arg = _parse_int(fields[5])
    if (
        reliable_seq is None
        or context_id is None
        or state is None
        or target is None
        or arg is None
    ):
        return None
    return {
        "reliable_seq": reliable_seq,
        "context_id": context_id,
        "state": state,
        "target": target,
        "arg": arg,
    }


def parse_ack_packet(line):
    """! @brief 解析可靠事件确认包

    @param line 原始输入行
    @return 确认包字段字典, 输入无效时返回 None
    """

    fields = _split_fields(line)
    if fields is None or len(fields) != 2 or fields[0].lower() != "a":
        return None
    reliable_seq = _parse_u8(fields[1])
    if reliable_seq is None:
        return None
    return {"reliable_seq": reliable_seq}


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
    @return 确认帧文本
    """

    return "a,%d" % int(reliable_seq)


def format_search_velocity_frame(vx, vy):
    """! @brief 格式化主车搜索速度数据流帧

    @param vx 主车搜索横向速度控制量
    @param vy 主车搜索纵向速度控制量
    @return 速度帧文本
    """

    return "v,%s,%s" % (compact_number(vx), compact_number(vy))


def format_event_frame(reliable_seq, context_id, event, value):
    """! @brief 格式化可靠事件回报帧

    @param reliable_seq 可靠包序号
    @param context_id 视觉上下文编号
    @param event 事件编号
    @param value 事件附加值
    @return 事件帧文本
    """

    return "r,%d,%d,%d,%d" % (
        int(reliable_seq),
        int(context_id),
        int(event),
        int(value),
    )


def _write_all(uart, line):
    """! @brief 向串口完整写出一行文本

    @param uart 目标串口对象
    @param line 不含行尾的短包文本
    @return 是否完整写出整行短包
    """

    remaining = str(line) + "\r\n"
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
    """! @brief 写出 v/o 数据流短包, 不执行发送延时

    @param uart 目标串口对象
    @param line 不含行尾的数据流短包文本
    """

    _write_all(uart, line)


def sleep_reliable_delay():
    """! @brief 可靠包发送保护延时"""

    try:
        time.sleep(RELIABLE_WRITE_DELAY_S)
    except AttributeError:
        pass


def write_reliable_line(uart, line):
    """! @brief 写出 s/a/r 可靠短包, 发送前后各延时 1 ms

    @param uart 目标串口对象
    @param line 不含行尾的可靠短包文本
    @return 是否完整写出整行短包
    """

    sleep_reliable_delay()
    success = _write_all(uart, line)
    sleep_reliable_delay()
    return success


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


def build_blob_candidates(img):
    """! @brief 提取物体候选目标, 面积作为目标强度

    @param img 当前图像对象
    @return 候选目标列表, 元素格式为 task_name, cx, bottom, area, blob
    """

    candidates = []
    for task_name, threshold in TASKS:
        blobs = img.find_blobs(
            [threshold], pixels_threshold=200, area_threshold=200, merge=True
        )
        if not blobs:
            continue
        img_height = img.height()
        for blob in blobs:
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


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    """! @brief 在调试画面上绘制选中物体角点与中心

    @param img 当前图像对象
    @param blob 被选中的候选色块对象
    @param pixel_x 目标中心 x 坐标
    @param pixel_y 目标标记 y 坐标
    """

    for corner_x, corner_y in get_marker_corners(blob):
        img.draw_cross(corner_x, corner_y)
    img.draw_cross(pixel_x, pixel_y)


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

    def has_context(self):
        """! @brief 判断是否已经建立视觉上下文

        @return 是否存在有效视觉上下文
        """

        return self.context is not None

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

    def handle_control_line(self, line):
        """! @brief 处理 RT1021 发来的同步或确认短包

        @param line 原始控制短包文本
        @return 需要回复的确认帧, 无需回复时返回 None
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
    """! @brief 处理 RT1021 发来的控制短包

    @param uart 控制链路串口对象
    @param rx_buffer 上一轮遗留的未完整输入
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
        rx_buffer += data.decode()
    except Exception:
        return rx_buffer
    while True:
        idx = rx_buffer.find("\n")
        if idx == -1:
            return rx_buffer
        line = rx_buffer[:idx].rstrip("\r").strip()
        rx_buffer = rx_buffer[idx + 1 :]
        reply = hook.handle_control_line(line)
        if reply is not None:
            write_reliable_line(uart, reply)


def build_observation_from_image(hook, img, image_width, image_height):
    """! @brief 从图像生成一帧物体观测

    @param hook 主车视觉 hook 状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    @return observation, best_blob 元组
    """

    candidates = build_blob_candidates(img)
    if not candidates:
        return hook.build_observation(0, 0, 0, 0, image_width, image_height), None
    target_x, target_y = build_search_target_point(
        image_width,
        image_height,
        hook.current_target_config_id(),
    )
    _, pixel_x, bottom_y, area, best_blob = choose_best_candidate(
        candidates, target_x, target_y
    )
    return (
        hook.build_observation(1, pixel_x, bottom_y, area, image_width, image_height),
        best_blob,
    )


def build_hook_event_value(hook, img, best_blob, image_width, image_height):
    """! @brief 根据当前 hook 生成事件附加判定值"""

    if not hook.is_finish_hook_context():
        return None
    return build_finish_hook_yellow_ratio_percent(
        img,
        best_blob,
        image_width,
        image_height,
    )


def process_search_frame(uart, hook, img, image_width, image_height):
    """! @brief 处理单帧主车搜索速度流和 hook 事件

    @param uart 主车视觉串口
    @param hook 主车视觉 hook 状态
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    observation, best_blob = build_observation_from_image(
        hook, img, image_width, image_height
    )
    _, x, y, _ = observation
    if best_blob is not None:
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
        ),
    )
    event_frame = hook.next_event_frame()
    if event_frame is not None:
        write_reliable_line(uart, event_frame)


def run():
    """! @brief 运行主车物体搜索视觉主循环"""

    uart = init_uart()
    image_width, image_height = init_sensor()
    hook = MasterVisionHook()
    rx_buffer = ""

    while True:
        rx_buffer = process_uart_input(uart, rx_buffer, hook)
        img = sensor.snapshot()  # type: ignore
        try:
            img.lens_corr(strength=2.8, zoom=1.0)
        except MemoryError:
            pass
        process_search_frame(uart, hook, img, image_width, image_height)


if __name__ == "__main__":
    run()
