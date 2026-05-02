"""! @file main.py
@brief OpenART 辅车跟随请求生成主程序.
@details 该程序负责图像采集、目标检测、阶段判断与速度量生成,
         并直接输出 `v,<vx>,<vy>` 短包文本.
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


# 主通信串口编号, 固定使用协议约定的 `UART(2)`
UART_ID = 2
# 主通信波特率, 需与 RT1021 侧保持一致
UART_BAUDRATE = 115200
# 固定曝光时间, 单位为微秒
EXP_TIME_US = 300
# 跟随控制使用的横向死区, 单位为像素
FOLLOW_X_DEADZONE_PX = 5.0
# 跟随控制使用的纵向目标尺度量, 单位为像素
FOLLOW_TARGET_Y = 45.0
# 跟随控制使用的纵向死区, 单位为像素
FOLLOW_Y_DEADZONE_PX = 8.0
# 跟随控制使用的横向速度修正量增益
FOLLOW_CONTROL_KP_X = 0.04
# 跟随控制使用的纵向速度修正量增益
FOLLOW_CONTROL_KP_Y = -0.10
# 跟随控制误差超出死区后的最小有效速度量
FOLLOW_CONTROL_MIN_SPEED = 1.5
# 纵向速度修正量上限, 避免尺度抖动时前后动作过猛
FOLLOW_CONTROL_MAX_Y = 5
# 串口写入前后的保护延时, 单位为秒
WRITE_DELAY_S = 0.002

# 参与检测的任务集合, 元素格式为 `(名称, 阈值)`
TASKS = (("red", (24, 100, 19, 127, 38, 127)),)


def format_vision_frame(vx, vy):
    """! @brief 将当前速度修正量编码为 v 短包文本帧.

    @param vx 车体系 x 方向视觉速度修正量.
    @param vy 车体系 y 方向视觉速度修正量.
    @return 当前主线单行速度短包文本.
    """
    x_value = float(vx)
    y_value = float(vy)
    if -0.0005 < x_value < 0.0005:
        x_value = 0.0
    if -0.0005 < y_value < 0.0005:
        y_value = 0.0
    x_text = ("%.3f" % x_value).rstrip("0").rstrip(".")
    y_text = ("%.3f" % y_value).rstrip("0").rstrip(".")
    if not x_text or x_text == "-0":
        x_text = "0"
    if not y_text or y_text == "-0":
        y_text = "0"
    return "v,%s,%s" % (x_text, y_text)


def _apply_min_speed(value, min_speed, limit=None):
    """! @brief 对非零速度量施加最小幅值和可选上限.

    @param value 原始速度量.
    @param min_speed 最小速度幅值.
    @param limit 可选速度幅值上限.
    @return 处理后的速度量.
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


def _build_axis_velocity(error, kp, limit=None):
    """! @brief 根据误差和增益生成带最小幅值的单轴速度量."""

    return _apply_min_speed(
        float(error) * float(kp),
        FOLLOW_CONTROL_MIN_SPEED,
        limit,
    )


def build_follow_command(valid, err_x, err_y):
    """! @brief 在 ART 端完成辅车跟随阶段判断与速度修正量生成.

    @param valid 当前帧是否存在有效目标.
    @param err_x 目标中心相对画面中心的横向像素差值.
    @param err_y 目标尺度量相对目标尺度量的纵向差值.
    @return 包含阶段名和速度修正量的字典.
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
    """! @brief 将 OpenMV 的 `x,y,w,h` 矩形转换为边界框四元组.

    @param rect `blob.rect()` 返回的 `(x, y, w, h)` 元组.
    @return `(left, top, right, bottom)` 形式的边界框.
    """
    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    """! @brief 将边界框统一到“地板在下方”的位置判断坐标系.

    @param left 原始识别框左边界.
    @param top 原始识别框上边界.
    @param right 原始识别框右边界.
    @param bottom 原始识别框下边界.
    @param img_height 当前图像高度.
    @return `(left, top, right, bottom)` 形式的归一化边界框.

    @note 当前相机实装视角以上下颠倒为基准, 主线统一以“地板在下方”
          的正向视图定义纵向坐标. 因此这里显式对纵向边界做一次归一化,
          不依赖底层驱动对坐标语义的翻转行为.
    """
    normalized_top = img_height - bottom
    normalized_bottom = img_height - top
    return left, normalized_top, right, normalized_bottom


def compute_lateral_error(blob_cx, cx_screen):
    """! @brief 计算跟随控制使用的横向偏差.

    @param blob_cx 当前目标横向中心像素坐标.
    @param cx_screen 画面横向中心像素坐标.
    @return 当前目标中心相对画面中心的横向整数偏差.
    """
    return int(round(float(blob_cx) - float(cx_screen)))


def edge_length(p0, p1):
    """! @brief 计算两点间边长.

    @param p0 第一端点 `(x, y)`.
    @param p1 第二端点 `(x, y)`.
    @return 两点间欧氏距离.
    """
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    return (dx * dx + dy * dy) ** 0.5


def get_marker_corners(blob):
    """! @brief 返回用于距离量和显示的大角点集合.

    @param blob 当前色块对象.
    @return 优先使用最小外接旋转矩形的四个角点.
    """
    return tuple(blob.min_corners())


def compute_marker_span(corners):
    """! @brief 基于梯形上底和下底平均值计算当前目标尺度量.

    @param corners 按顺时针排序的四个角点.
    @return 当前目标的纵向距离代理尺度量.
    """
    top_width = edge_length(corners[0], corners[1])
    bottom_width = edge_length(corners[3], corners[2])
    return (top_width + bottom_width) / 2.0


def compute_vertical_error(marker_span, target_span):
    """! @brief 计算跟随控制使用的纵向偏差.

    @param marker_span 当前目标的尺度量.
    @param target_span 纵向目标尺度量.
    @return 当前目标尺度量相对目标尺度量的纵向整数偏差.
    """
    return int(round(float(marker_span) - float(target_span)))


def choose_best_candidate(candidates, cx_screen, img_height):
    """! @brief 选择当前帧最值得上报的目标.

    @param candidates 候选目标列表, 元素至少包含名称, 重心 x, 重心 y, 底边 y.
    @param cx_screen 画面横向中心像素坐标.
    @param img_height 当前图像高度.
    @return 在翻转后的正向图像中, 更接近中线且底边更靠近地板的候选目标.
    """
    return min(
        candidates,
        key=lambda item: (item[1] - cx_screen) ** 2 + (img_height - item[3]) ** 2,
    )


def sleep_short():
    """! @brief 执行一次统一的短延时.

    @details 该函数用于串口写入前后的轻量保护, 同时兼容主机侧导入场景.
    """
    delay_s = globals().get("WRITE_DELAY_S", 0.002)
    try:
        time.sleep(delay_s)
    except AttributeError:
        pass


def write_line(uart, line):
    """! @brief 按协议发送单行文本.

    @param uart 当前使用的串口对象.
    @param line 待发送的单行 ASCII 文本, 函数内部会补齐 `\r\n`.
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


def build_blob_candidates(img):
    """! @brief 提取所有颜色候选目标的重心、底边与尺度量信息.

    @param img 当前帧图像对象.
    @return 候选目标列表, 元素格式为 `(名称, cx, cy, bottom, span, blob)`.
    """
    candidates = []
    img_height = img.height()
    for task_name, threshold in TASKS:
        # 对每种颜色任务分别做一次 blob 检测,再合并成统一候选集合
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


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    """! @brief 在调试画面上绘制当前选中目标的四角与中心.

    @param img 当前图像对象.
    @param blob 当前选中的色块对象.
    @param pixel_x 当前目标中心 x.
    @param pixel_y 当前目标中心 y.
    """
    for corner_x, corner_y in get_marker_corners(blob):
        img.draw_cross(corner_x, corner_y)
    img.draw_cross(pixel_x, pixel_y)


def init_uart():
    """! @brief 初始化主通信串口.

    @return 配置完成的 UART 对象.
    @exception RuntimeError 主机侧导入且无 UART 平台支持时抛出.
    """
    if UART is None:
        raise RuntimeError("UART unavailable in host environment")
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    """! @brief 初始化 OpenART 摄像头参数.

    @return 画面横向中心像素坐标.
    @exception RuntimeError 主机侧导入且无 sensor 平台支持时抛出.
    """
    if sensor is None:
        raise RuntimeError("sensor unavailable in host environment")

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    # 传感器侧尽量输出接近正向的画面, 但协议坐标仍由软件层统一归一化
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)  # type: ignore
    sensor.set_auto_gain(False)  # type: ignore
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width() // 2


def run():
    """! @brief 持续检测目标并逐帧发送速度短包.

    @details 执行流程为: 初始化串口与摄像头 -> 持续采集图像 -> 提取颜色候选目标
             -> 选择最优目标 -> 计算像素偏差 -> 在 ART 端完成阶段判断和速度修正量生成
             -> 每抓一帧发送一帧速度短包.
    """
    # 先完成串口和摄像头初始化, 后续主循环逐帧输出速度短包
    uart = init_uart()
    cx_screen = init_sensor()

    while True:
        # 每轮抓取一帧图像作为本次检测输入
        img = sensor.snapshot()  # type: ignore

        try:
            # 镜头畸变校正有助于减小边缘区域的像素偏差
            img.lens_corr(strength=2.8, zoom=1.0)
        except MemoryError:
            # 内存不足时直接跳过校正,优先保持主循环持续运行
            pass

        # 收集当前帧内所有颜色候选目标
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
            continue

        # 选择当前帧最适合上报的单个目标
        _, pixel_x, pixel_y, _, marker_span, best_blob = choose_best_candidate(
            candidates, cx_screen, img.height()
        )
        err_x = compute_lateral_error(blob_cx=pixel_x, cx_screen=cx_screen)
        err_y = compute_vertical_error(
            marker_span=marker_span, target_span=FOLLOW_TARGET_Y
        )
        follow_command = build_follow_command(valid=1, err_x=err_x, err_y=err_y)
        # 在调试画面上标出当前被选中的目标四角和中心
        draw_selected_marker(img=img, blob=best_blob, pixel_x=pixel_x, pixel_y=pixel_y)
        write_line(
            uart,
            format_vision_frame(
                vx=follow_command["command_vx"],
                vy=follow_command["command_vy"],
            ),
        )

if __name__ == "__main__":
    run()
