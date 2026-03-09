"""! @file main.py
@brief OpenART 纯视觉观测上报主程序.
@details 该程序只负责图像采集、目标选择与 `x,y` 观测上报,
         不再承担本地动作状态机与离散控制命令编排.
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
EXP_TIME_US = 500
# 连续视觉帧上报的最小间隔, 单位为毫秒
SEND_INTERVAL_MS = 40
# 启动后是否先发送一次 `reset=1`
STARTUP_RESET = True
# 串口写入前后的保护延时, 单位为秒
WRITE_DELAY_S = 0.002

# 红色目标的 LAB 阈值参数
RED_THRESHOLD = (0, 100, 23, 127, -26, 127)
# 绿色目标的 LAB 阈值参数
GREEN_THRESHOLD = (0, 100, -128, -14, -128, 127)
# 参与检测的任务集合, 元素格式为 `(名称, 阈值)`
TASKS = (("Red", RED_THRESHOLD), ("Green", GREEN_THRESHOLD))


def format_vision_frame(pixel_x, pixel_y):
    """! @brief 将像素坐标编码为协议要求的纯 `x,y` 文本帧.

    @param pixel_x 目标横向像素坐标.
    @param pixel_y 目标纵向像素坐标.
    @return 形如 `x=<num>,y=<num>` 的单行协议文本.
    """
    return "x=%s,y=%s" % (int(round(pixel_x)), int(round(pixel_y)))


def choose_best_candidate(candidates, cx_screen, img_height):
    """! @brief 选择当前帧最值得上报的目标.

    @param candidates 候选目标列表, 元素至少包含名称, 像素 x, 像素 y.
    @param cx_screen 画面横向中心像素坐标.
    @param img_height 当前图像高度.
    @return 与画面中线更接近且更靠近底边的候选目标.
    """
    return min(
        candidates,
        key=lambda item: (item[1] - cx_screen) ** 2 + (img_height - item[2]) ** 2,
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
    sleep_short()
    uart.write(line + "\r\n")
    sleep_short()


def send_startup_reset(uart):
    """! @brief 启动时按协议发送一次运行态复位.

    @param uart 当前使用的串口对象.
    """
    write_line(uart, "reset=1")


def build_blob_candidates(img):
    """! @brief 提取所有颜色候选目标的像素中心点.

    @param img 当前帧图像对象.
    @return 候选目标列表, 元素格式为 `(名称, cx, cy, blob)`.
    """
    candidates = []
    for task_name, threshold in TASKS:
        # 对每种颜色任务分别做一次 blob 检测,再合并成统一候选集合
        blobs = img.find_blobs(
            [threshold], pixels_threshold=200, area_threshold=200, merge=True
        )
        for blob in blobs:
            candidates.append((task_name, blob.cx(), blob.cy(), blob))
    return candidates


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
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)  # type: ignore
    sensor.set_auto_gain(False)  # type: ignore
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width() // 2


def run():
    """! @brief 持续检测目标并向 RT1021 上报最新像素观测.

    @details 执行流程为: 初始化串口与摄像头 -> 按需发送一次 `reset=1` ->
             持续采集图像 -> 提取颜色候选目标 -> 选择最优目标 -> 按节流周期
             发送 `x,y` 观测帧. 无目标时不发送伪帧, 由 RT1021 侧按超时机制
             判定目标丢失.
    @note 本函数不再发送 `dx/dy/d_angle`, `rear`, `angle` 等控制命令,
          OpenArt 仅承担感知与观测上报职责.
    """
    # 先完成串口和摄像头初始化,后续主循环只处理观测上报
    uart = init_uart()
    cx_screen = init_sensor()
    # 记录上一帧成功发送的时刻,用于发送节流
    last_send_ms = None

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
        if not candidates:
            # 无目标时不发送任何伪帧,由 RT1021 侧自行按超时判丢失
            continue

        # 选择当前帧最适合上报的单个目标
        _, pixel_x, pixel_y, best_blob = choose_best_candidate(
            candidates, cx_screen, img.height()
        )
        # 在调试画面上标出当前被选中的目标
        img.draw_rectangle(best_blob.rect())
        img.draw_cross(pixel_x, pixel_y)

        now_ms = time.ticks_ms()
        if should_send(now_ms, last_send_ms, SEND_INTERVAL_MS):
            # 只发送严格的 x,y 二元观测帧,避免落回旧遥控协议解析分支
            write_line(uart, format_vision_frame(pixel_x, pixel_y))
            last_send_ms = now_ms


if __name__ == "__main__":
    run()
