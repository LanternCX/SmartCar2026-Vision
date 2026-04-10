"""! @file main.py
@brief OpenART 跟随请求生成主程序.
@details 该程序负责图像采集、目标检测、阶段判断与 follow 请求生成,
         并直接输出辅车当前已经支持的 `f=1,s=...,v=...,x=...,y=...` 文本.
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
# 当前脚本代表的物理相机 ID
CAMERA_ID = "cam_a"
# 连续视觉帧上报的最小间隔, 单位为毫秒
SEND_INTERVAL_MS = 40
# 主车当前使用的跟随中心死区, 单位为像素
FOLLOW_CENTER_DEADZONE_PX = 8.0
# 主车当前使用的横向位置式控制增益
FOLLOW_CONTROL_KP_X = -0.02
# 纵向目标面积, 单位为 blob 像素面积
FOLLOW_TARGET_AREA = 2400
# 纵向面积死区, 单位为 blob 像素面积
FOLLOW_AREA_DEADZONE = 120
# 主车当前使用的面积式纵向控制增益
FOLLOW_CONTROL_KP_AREA = 0.000
# 纵向控制量上限, 避免面积抖动时前后动作过猛
FOLLOW_CONTROL_MAX_Y = 1.2
# 默认主线直接发送 follow 请求, 不再先发不兼容的 `reset=1`
STARTUP_RESET = False
# 串口写入前后的保护延时, 单位为秒
WRITE_DELAY_S = 0.002

# 红色目标的 LAB 阈值参数
RED_THRESHOLD = (0, 100, 23, 127, -26, 127)
# 参与检测的任务集合, 元素格式为 `(名称, 阈值)`
TASKS = (("red", (0, 100, 23, 127, -26, 127)),)


def format_vision_frame(seq, valid, err_x, err_y):
    """! @brief 将当前控制结果编码为辅车速度模式请求.

    @param seq 当前发送序号.
    @param valid 当前拍 follow 控制是否有效.
    @param err_x 发给辅车的横向控制量.
    @param err_y 发给辅车的纵向控制量.
    @return 当前主线单行速度模式请求文本.

    @note 输出必须直接对齐辅车当前已经支持的 `f=1,m=1,x=...,y=...`
          速度模式语义, 无效拍与保持拍统一输出 `x=0,y=0`.
    """
    if not int(valid):
        return "f=1,m=1,x=0,y=0"

    x_value = float(err_x)
    y_value = float(err_y)
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
    return "f=1,m=1,x=%s,y=%s" % (x_text, y_text)


def build_follow_command(valid, err_x, blob_area):
    """! @brief 在 ART 端完成跟随阶段判断与控制量生成.

    @param valid 当前帧是否存在有效目标.
    @param err_x 目标中心相对画面中心的横向像素差值.
    @param blob_area 当前目标的像素面积.
    @return 包含阶段名和 follow 控制量的字典.

    @note 这里直接复用辅车当前速度模式入口: `x/y` 表示速度量,
          而不是继续输出位置式控制量.
    """
    if int(valid) != 1:
        return {
            "phase": "MARKER_MISSING",
            "valid": 0,
            "follow_x": 0.0,
            "follow_y": 0.0,
        }

    err_x = float(err_x)
    blob_area = float(blob_area)
    deadzone_px = float(FOLLOW_CENTER_DEADZONE_PX)
    area_error = float(FOLLOW_TARGET_AREA) - blob_area
    area_deadzone = float(FOLLOW_AREA_DEADZONE)
    if abs(err_x) <= deadzone_px and abs(area_error) <= area_deadzone:
        return {
            "phase": "CENTER_HOLD",
            "valid": 0,
            "follow_x": 0.0,
            "follow_y": 0.0,
        }

    if abs(err_x) > deadzone_px:
        return {
            "phase": "ALIGN_X",
            "valid": 1,
            "follow_x": -err_x * float(FOLLOW_CONTROL_KP_X),
            "follow_y": 0.0,
        }

    follow_y = area_error * float(FOLLOW_CONTROL_KP_AREA)
    max_y = float(FOLLOW_CONTROL_MAX_Y)
    if follow_y > max_y:
        follow_y = max_y
    if follow_y < -max_y:
        follow_y = -max_y

    return {
        "phase": "ALIGN_Y",
        "valid": 1,
        "follow_x": 0.0,
        "follow_y": follow_y,
    }


def blob_rect_to_bbox(rect):
    """! @brief 将 OpenMV 的 `x,y,w,h` 矩形转换为边界框四元组.

    @param rect `blob.rect()` 返回的 `(x, y, w, h)` 元组.
    @return `(left, top, right, bottom)` 形式的边界框.
    """
    left, top, width, height = rect
    return left, top, left + width, top + height


def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    """! @brief 将边界框统一到“地板在下方”的协议坐标系.

    @param left 原始识别框左边界.
    @param top 原始识别框上边界.
    @param right 原始识别框右边界.
    @param bottom 原始识别框下边界.
    @param img_height 当前图像高度.
    @return `(left, top, right, bottom)` 形式的协议坐标边界框.

    @note 当前相机实装视角以上下颠倒为基准, 协议层统一以“地板在下方”
          的正向视图定义纵向坐标. 因此这里显式对纵向边界做一次归一化,
          不依赖底层驱动对坐标语义的翻转行为.
    """
    normalized_top = img_height - bottom
    normalized_bottom = img_height - top
    return left, normalized_top, right, normalized_bottom


def compute_lateral_error(blob_cx, cx_screen):
    """! @brief 计算主线协议要求的横向偏差.

    @param blob_cx 当前目标横向中心像素坐标.
    @param cx_screen 画面横向中心像素坐标.
    @return 当前目标中心相对画面中心的横向整数偏差.
    """
    return int(round(float(blob_cx) - float(cx_screen)))


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


def should_send(now_ms, last_send_ms, interval_ms):
    """! @brief 判断当前时刻是否允许发送新一帧观测.

    @param now_ms 当前毫秒计数.
    @param last_send_ms 上一次发送时刻, 若为空表示尚未发送过.
    @param interval_ms 允许发送的最小间隔.
    @return 若达到发送周期则返回 `True`, 否则返回 `False`.
    """
    if last_send_ms is None:
        return True
    try:
        elapsed_ms = time.ticks_diff(now_ms, last_send_ms)
    except AttributeError:
        elapsed_ms = now_ms - last_send_ms
    return elapsed_ms >= interval_ms


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


def send_startup_reset(uart):
    """! @brief 启动时按协议发送一次运行态复位.

    @param uart 当前使用的串口对象.
    """
    write_line(uart, "reset=1")


def build_blob_candidates(img):
    """! @brief 提取所有颜色候选目标的重心与底边信息.

    @param img 当前帧图像对象.
    @return 候选目标列表, 元素格式为 `(名称, cx, cy, bottom, area, blob)`.
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
            candidates.append(
                (task_name, blob.cx(), blob.cy(), bottom, blob.area(), blob)
            )
    return candidates


def build_frame_detections(candidates, img_height):
    """! @brief 将当前帧候选集合转换成协议检测列表."""
    detections = []
    for category, _cx, _cy, _bottom, blob in candidates:
        left, top, right, bottom = blob_rect_to_bbox(blob.rect())
        left, top, right, bottom = normalize_bbox_for_protocol(
            left, top, right, bottom, img_height
        )
        detections.append(
            {
                "category": str(category),
                "left": int(round(left)),
                "top": int(round(top)),
                "right": int(round(right)),
                "bottom": int(round(bottom)),
            }
        )
    return detections


def poll_uart_lines(uart, rx_buffer):
    """! @brief 读取串口并拆出当前已完整接收的文本行."""
    try:
        available = uart.any()
    except AttributeError:
        available = 0
    if available:
        payload = uart.read(available)
        if payload is not None:
            rx_buffer += payload.decode()

    lines = []
    while True:
        line_end = rx_buffer.find("\n")
        if line_end == -1:
            break
        line = rx_buffer[:line_end].rstrip("\r").strip()
        rx_buffer = rx_buffer[line_end + 1 :]
        if line:
            lines.append(line)
    return rx_buffer, lines


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
    """! @brief 持续检测目标并直接发送速度模式请求.

    @details 执行流程为: 初始化串口与摄像头 -> 按需发送一次 `reset=1` ->
             持续采集图像 -> 提取颜色候选目标 -> 选择最优目标 -> 计算像素偏差
             -> 在 ART 端完成阶段判断和速度量生成 -> 按节流周期发送速度模式请求.
    @note 输出直接对齐辅车当前速度模式入口, 不再额外扩展第二套视觉发包字段.
    """
    # 先完成串口和摄像头初始化,后续主循环直接输出速度模式请求
    uart = init_uart()
    cx_screen = init_sensor()
    # 记录上一帧成功发送的时刻,用于发送节流
    last_send_ms = None
    # 当前视觉上报序号,每次真正发包时递增
    report_seq = 0

    if STARTUP_RESET:
        # 启动阶段按协议发送一次 reset,让 RT1021 进入干净运行态
        send_startup_reset(uart)

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
        now_ms = time.ticks_ms()
        if not should_send(now_ms, last_send_ms, SEND_INTERVAL_MS):
            continue

        report_seq += 1
        if not candidates:
            follow_command = build_follow_command(valid=0, err_x=0, blob_area=0)
            write_line(
                uart,
                format_vision_frame(
                    seq=report_seq,
                    valid=follow_command["valid"],
                    err_x=follow_command["follow_x"],
                    err_y=follow_command["follow_y"],
                ),
            )
            last_send_ms = now_ms
            continue

        # 选择当前帧最适合上报的单个目标
        target_label, pixel_x, pixel_y, _, blob_area, best_blob = choose_best_candidate(
            candidates, cx_screen, img.height()
        )
        left, top, right, bottom = blob_rect_to_bbox(best_blob.rect())
        left, top, right, bottom = normalize_bbox_for_protocol(
            left, top, right, bottom, img.height()
        )
        err_x = compute_lateral_error(blob_cx=pixel_x, cx_screen=cx_screen)
        follow_command = build_follow_command(valid=1, err_x=err_x, blob_area=blob_area)
        # 在调试画面上标出当前被选中的目标
        img.draw_rectangle(best_blob.rect())
        img.draw_cross(pixel_x, pixel_y)
        write_line(
            uart,
            format_vision_frame(
                seq=report_seq,
                valid=follow_command["valid"],
                err_x=follow_command["follow_x"],
                err_y=follow_command["follow_y"],
            ),
        )
        last_send_ms = now_ms


if __name__ == "__main__":
    run()
