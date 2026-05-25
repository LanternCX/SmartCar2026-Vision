"""! @file main.py
@brief OpenART 辅车视觉入口
@details 负责跟随色标与找物体两种模式的图像处理, 输出速度短包并维护本地可靠事件
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
# 串口写入前后的保护延时，单位为秒。
WRITE_DELAY_S = 0.002
# 可靠事件默认重发间隔，单位为毫秒。
RELIABLE_RESEND_INTERVAL_MS = 100
# 可靠序号使用 0..255 环形空间。
SEQ_RING_SIZE = 256
# 判断序号新旧使用的半环长度。
SEQ_HALF_RING = 128

# 默认运行模式。
MODE_FOLLOW = "follow"
# 接收到同步后切换的找物体模式。
MODE_APPROACH_OBJECT = "approach_object"
# 接收到同步后切换的绕行修正模式。
MODE_ORBIT_OBJECT = "orbit_object"
# 辅车找物体状态编号。
STATE_APPROACH_OBJECT = 2
# 辅车绕行状态编号。
STATE_ORBIT = 3
# 物体目标编号。
TARGET_OBJECT = 1
# 找物体同步参数编号。
OBJECT_APPROACH_CONFIG_ID = 1
# 搬运对正同步参数编号。
ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID = 2
# 绕行修正同步参数编号。
ASSISTANT_ORBIT_OBJECT_CONFIG_ID = 3
# TARGET_FOUND 事件编号。
EVENT_TARGET_FOUND = 6
# ALIGNED 事件编号。
EVENT_ALIGNED = 7

# 跟随模式使用的绿色色标阈值。
FOLLOW_TASKS = (("green", (37, 8, -57, -8, -36, 6)),)
# 找物体模式使用的红色目标阈值，与主车保持一致。
OBJECT_TASKS = (("red", (0, 100, 18, 127, -23, 127)),)

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
OBJECT_ORBIT_KP_X = 0.00
# 绕行修正纵向速度 P 环增益。
OBJECT_ORBIT_KP_Y = 0.00
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


def format_vision_frame(vx, vy):
    """! @brief 将当前速度修正量编码为 v 短包文本帧

    @param vx 车体系 x 方向视觉速度修正量
    @param vy 车体系 y 方向视觉速度修正量
    @return 当前主线单行速度短包文本
    """

    return "v,%s,%s" % (compact_number(vx), compact_number(vy))


def format_ack_frame(reliable_seq):
    """! @brief 格式化本地同步确认帧

    @param reliable_seq 被确认的同步序号
    @return 确认帧文本
    """

    return "a,%d" % int(reliable_seq)


def format_event_frame(reliable_seq, event, value):
    """! @brief 格式化可靠事件回报帧

    @param reliable_seq 当前同步序号
    @param event 事件编号
    @param value 事件附加值
    @return 事件帧文本
    """

    return "r,%d,%d,%d" % (
        int(reliable_seq),
        int(event),
        int(value),
    )


def _parse_int(text):
    """! @brief 解析严格整数字段

    @param text 数值文本
    @return 整数，输入无效时返回 None
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
    @return 整数，输入无效时返回 None
    """

    value = _parse_int(text)
    if value is None or value < 0 or value > 255:
        return None
    return value


def _split_fields(line):
    """! @brief 拆分短包字段并过滤空字段

    @param line 原始输入行
    @return 字段列表，输入无效时返回 None
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
    """! @brief 解析 RT1021 下发的本地同步包

    @param line 原始输入行
    @return 同步包字段字典，输入无效时返回 None
    """

    fields = _split_fields(line)
    if fields is None or len(fields) != 5 or fields[0].lower() != "s":
        return None
    reliable_seq = _parse_u8(fields[1])
    state = _parse_u8(fields[2])
    target = _parse_u8(fields[3])
    arg = _parse_int(fields[4])
    if reliable_seq is None or state is None or target is None or arg is None:
        return None
    return {
        "reliable_seq": reliable_seq,
        "state": state,
        "target": target,
        "arg": arg,
    }


def parse_ack_packet(line):
    """! @brief 解析可靠事件确认包

    @param line 原始输入行
    @return 确认包字段字典，输入无效时返回 None
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


def build_blob_candidates(img):
    """! @brief 提取跟随模式颜色候选目标的重心、底边与尺度量信息

    @param img 当前帧图像对象
    @return 候选目标列表，元素格式为 名称, cx, cy, bottom, span, blob
    """

    candidates = []
    img_height = img.height()
    for task_name, threshold in FOLLOW_TASKS:
        blobs = img.find_blobs(
            [threshold], pixels_threshold=200, area_threshold=200, merge=True
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


def build_object_blob_candidates(img):
    """! @brief 提取找物体模式候选目标的重心、底边与面积信息

    @param img 当前帧图像对象
    @return 候选目标列表，元素格式为 名称, cx, cy, bottom, area, blob
    """

    candidates = []
    img_height = img.height()
    for task_name, threshold in OBJECT_TASKS:
        blobs = img.find_blobs(
            [threshold], pixels_threshold=200, area_threshold=200, merge=True
        )
        for blob in blobs:
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, bottom = normalize_bbox_for_protocol(
                left, top, right, bottom, img_height
            )
            candidates.append(
                (task_name, blob.cx(), blob.cy(), bottom, blob_area(blob), blob)
            )
    return candidates


def choose_best_candidate(candidates, cx_screen, img_height):
    """! @brief 选择当前帧最值得上报的目标

    @param candidates 候选目标列表
    @param cx_screen 画面横向中心像素坐标
    @param img_height 当前图像高度
    @return 更接近中线且底边更靠近地板的候选目标
    """

    return min(
        candidates,
        key=lambda item: (item[1] - cx_screen) ** 2 + (img_height - item[3]) ** 2,
    )


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    """! @brief 在调试画面上绘制当前选中目标的四角与中心

    @param img 当前图像对象
    @param blob 当前选中的色块对象
    @param pixel_x 当前目标中心 x
    @param pixel_y 当前目标中心 y
    """

    for corner_x, corner_y in get_marker_corners(blob):
        img.draw_cross(corner_x, corner_y)
    img.draw_cross(pixel_x, pixel_y)


def build_object_observation(
    valid,
    center_x,
    bottom_y,
    area,
    image_width,
    image_height,
    config_id=OBJECT_APPROACH_CONFIG_ID,
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
    config_id=OBJECT_APPROACH_CONFIG_ID,
):
    """! @brief 根据当前配置生成找物体目标点"""

    _ = image_width
    _ = image_height
    target_x = float(OBJECT_APPROACH_TARGET_X_PX)
    if int(config_id) == int(ASSISTANT_ORBIT_OBJECT_CONFIG_ID):
        return float(OBJECT_ORBIT_TARGET_X_PX), float(OBJECT_ORBIT_TARGET_Y_PX)
    if int(config_id) == int(ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID):
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
        self.mode = MODE_FOLLOW
        self.current_sync = None
        self._last_sync_seq = None
        self._stable_count = 0
        self._pending_event = None
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = None
        self._now_ms = now_ms or default_now_ms
        self._event_resend_interval_ms = int(event_resend_interval_ms)

    def handle_control_line(self, line):
        """! @brief 处理 RT1021 发来的同步或确认短包

        @param line 原始控制短包文本
        @return 需要回复的确认帧，无需回复时返回 None
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
            int(sync["state"]) == STATE_APPROACH_OBJECT
            and int(sync["target"]) == TARGET_OBJECT
            and (
                int(sync["arg"]) == OBJECT_APPROACH_CONFIG_ID
                or int(sync["arg"]) == ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID
            )
        ):
            return MODE_APPROACH_OBJECT
        if (
            int(sync["state"]) == STATE_ORBIT
            and int(sync["target"]) == TARGET_OBJECT
            and int(sync["arg"]) == ASSISTANT_ORBIT_OBJECT_CONFIG_ID
        ):
            return MODE_ORBIT_OBJECT
        return MODE_FOLLOW

    def current_object_config_id(self):
        """! @brief 返回当前找物体阶段使用的目标点配置编号"""

        if self.current_sync is None:
            return OBJECT_APPROACH_CONFIG_ID
        return int(self.current_sync["arg"])

    def _handle_ack_packet(self, packet):
        """! @brief 处理可靠事件确认包

        @param packet 解析后的确认包字段
        """

        if self._pending_event is None:
            return
        if int(packet["reliable_seq"]) == int(self._pending_event["reliable_seq"]):
            self._pending_event = None
            self._pending_event_last_sent_ms = None

    def accept_object_observation(self, observation):
        """! @brief 累计找物体稳定条件并按需创建可靠事件

        @param observation x, y, value 观测字段元组
        """

        if self.mode != MODE_APPROACH_OBJECT or self.current_sync is None:
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
        arg = int(self.current_sync["arg"])
        if (
            state == STATE_APPROACH_OBJECT
            and target == TARGET_OBJECT
            and arg == OBJECT_APPROACH_CONFIG_ID
        ):
            return EVENT_TARGET_FOUND
        if (
            state == STATE_APPROACH_OBJECT
            and target == TARGET_OBJECT
            and arg == ASSISTANT_TRANSPORT_OBJECT_CONFIG_ID
        ):
            return EVENT_ALIGNED
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


def sleep_short():
    """! @brief 执行一次统一的短延时

    @details 该函数用于串口写入前后的轻量保护，同时兼容主机侧导入场景
    """

    delay_s = globals().get("WRITE_DELAY_S", 0.002)
    try:
        time.sleep(delay_s)
    except AttributeError:
        pass


def write_line(uart, line):
    """! @brief 按协议发送单行文本

    @param uart 当前使用的串口对象
    @param line 待发送的单行 ASCII 文本，函数内部会补齐 \r\n
    """

    remaining = str(line) + "\r\n"
    sleep_short()
    while remaining:
        written = uart.write(remaining)
        if written is None:
            written = len(remaining)
        written = int(written)
        if written <= 0:
            sleep_short()
            continue
        remaining = remaining[written:]
        if remaining:
            sleep_short()
    sleep_short()


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


def process_uart_input(uart, rx_buffer, state):
    """! @brief 处理 RT1021 发来的控制短包

    @param uart 控制链路串口对象
    @param rx_buffer 上一轮遗留的未完整输入
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
        rx_buffer += data.decode()
    except Exception:
        return rx_buffer
    while True:
        idx = rx_buffer.find("\n")
        if idx == -1:
            return rx_buffer
        line = rx_buffer[:idx].rstrip("\r").strip()
        rx_buffer = rx_buffer[idx + 1 :]
        reply = state.handle_control_line(line)
        if reply is not None:
            write_line(uart, reply)


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


def process_object_frame(uart, state, img, image_width, image_height):
    """! @brief 处理单帧找物体模式速度输出与可靠事件

    @param uart 辅车视觉串口
    @param state 辅车视觉状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    candidates = build_object_blob_candidates(img)
    if not candidates:
        observation = build_object_observation(0, 0, 0, 0, image_width, image_height)
    else:
        config_id = state.current_object_config_id()
        target_x, target_y = build_object_target_point(
            image_width,
            image_height,
            config_id,
        )
        _, pixel_x, pixel_y, bottom_y, area, best_blob = choose_best_candidate(
            candidates, target_x, target_y
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
        draw_selected_marker(img=img, blob=best_blob, pixel_x=pixel_x, pixel_y=pixel_y)
    if state.mode == MODE_ORBIT_OBJECT:
        vx, vy = build_object_orbit_velocity_from_observation(observation, image_height)
    else:
        vx, vy = build_object_approach_velocity_from_observation(
            observation,
            image_height,
        )
    write_line(uart, format_vision_frame(vx, vy))
    state.accept_object_observation(observation)


def process_frame(uart, state, img, image_width, image_height):
    """! @brief 按当前模式处理单帧视觉输出

    @param uart 辅车视觉串口
    @param state 辅车视觉状态对象
    @param img 当前图像对象
    @param image_width 图像宽度
    @param image_height 图像高度
    """

    if state.mode == MODE_APPROACH_OBJECT or state.mode == MODE_ORBIT_OBJECT:
        process_object_frame(uart, state, img, image_width, image_height)
    else:
        process_follow_frame(uart, img, image_width, image_height)
    event_frame = state.next_event_frame()
    if event_frame is not None:
        write_line(uart, event_frame)


def run():
    """! @brief 持续检测目标并逐帧发送视觉短包"""

    uart = init_uart()
    image_width, image_height = init_sensor()
    state = AssistantVisionState()
    rx_buffer = ""

    while True:
        rx_buffer = process_uart_input(uart, rx_buffer, state)
        img = sensor.snapshot()  # type: ignore
        try:
            img.lens_corr(strength=2.8, zoom=1.0)
        except MemoryError:
            pass
        process_frame(uart, state, img, image_width, image_height)


if __name__ == "__main__":
    run()
