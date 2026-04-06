"""! @file main.py
@brief OpenART 纯视觉观测上报主程序.
@details 该程序只负责图像采集、目标选择与完整识别框观测上报,
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
# 纵向期望抓取位置, 单位为像素差值
TARGET_ERR_Y = 60
# 启动后是否先发送一次 `reset=1`
STARTUP_RESET = True
# 串口写入前后的保护延时, 单位为秒
WRITE_DELAY_S = 0.002

# 红色目标的 LAB 阈值参数
RED_THRESHOLD = (0, 100, 23, 127, -26, 127)
# 绿色目标的 LAB 阈值参数
GREEN_THRESHOLD = (0, 100, -128, -14, -128, 127)
# 参与检测的任务集合, 元素格式为 `(名称, 阈值)`
TASKS = (("red", RED_THRESHOLD), ("green", GREEN_THRESHOLD))


def format_vision_frame(seq, valid, err_x, err_y):
    """! @brief 将当前目标编码为最小视觉协议帧.

    @param seq 当前发送序号.
    @param valid 当前帧是否存在有效目标.
    @param err_x 当前目标横向偏差.
    @param err_y 当前目标纵向偏差.
    @return 当前主线视觉单行协议文本.

    @note 正式发送只保留版本位、序号以及像素差值, 其中 `x/y` 仍分别表示
          横向中心偏差与底边相对期望抓取位置的纵向偏差.
    """
    if not int(valid):
        return "v=0,s=%s" % int(seq)
    return "v=1,s=%s,x=%s,y=%s" % (
        int(seq),
        int(round(err_x)),
        int(round(err_y)),
    )


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


def compute_protocol_errors(blob_cx, normalized_bottom, cx_screen, img_height):
    """! @brief 计算主线协议要求的横纵向偏差.

    @param blob_cx 当前目标横向中心像素坐标.
    @param normalized_bottom 当前目标在协议坐标系下的底边像素坐标.
    @param cx_screen 画面横向中心像素坐标.
    @param img_height 当前图像高度.
    @return `(err_x, err_y)` 形式的整数偏差, 其中 `err_y=0` 表示目标已到达期望抓取距离,
            目标底边超过期望抓取位置时为正, 未达到时为负.
    """
    err_x = int(round(float(blob_cx) - float(cx_screen)))
    err_y = int(
        round(float(normalized_bottom) - float(img_height) + float(TARGET_ERR_Y))
    )
    return err_x, err_y


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
    sleep_short()
    uart.write(line + "\r\n")
    sleep_short()


def send_startup_reset(uart):
    """! @brief 启动时按协议发送一次运行态复位.

    @param uart 当前使用的串口对象.
    """
    write_line(uart, "reset=1")


def build_blob_candidates(img):
    """! @brief 提取所有颜色候选目标的重心与底边信息.

    @param img 当前帧图像对象.
    @return 候选目标列表, 元素格式为 `(名称, cx, cy, bottom, blob)`.
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
            candidates.append((task_name, blob.cx(), blob.cy(), bottom, blob))
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
    # 传感器侧尽量输出接近正向的画面, 但协议坐标仍由软件层统一归一化
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)  # type: ignore
    sensor.set_auto_gain(False)  # type: ignore
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width() // 2


def run():
    """! @brief 持续检测目标并向 RT1021 上报最新识别框观测.

    @details 执行流程为: 初始化串口与摄像头 -> 按需发送一次 `reset=1` ->
             持续采集图像 -> 提取颜色候选目标 -> 选择最优目标 -> 按节流周期
             发送最小观测帧. 无目标时发送 `v=0,s=<seq>`, 有目标时发送
             `v=1,s=<seq>,x=<x>,y=<y>`.
    @note 本函数不再发送 `dx/dy/d_angle`, `rear`, `angle` 等控制命令,
          OpenArt 仅承担感知与观测上报职责.
    """
    # 先完成串口和摄像头初始化,后续主循环只处理观测上报
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
            write_line(
                uart,
                format_vision_frame(
                    seq=report_seq,
                    valid=0,
                    err_x=0,
                    err_y=0,
                ),
            )
            last_send_ms = now_ms
            continue

        # 选择当前帧最适合上报的单个目标
        target_label, pixel_x, pixel_y, _, best_blob = choose_best_candidate(
            candidates, cx_screen, img.height()
        )
        left, top, right, bottom = blob_rect_to_bbox(best_blob.rect())
        left, top, right, bottom = normalize_bbox_for_protocol(
            left, top, right, bottom, img.height()
        )
        err_x, err_y = compute_protocol_errors(
            blob_cx=pixel_x,
            normalized_bottom=bottom,
            cx_screen=cx_screen,
            img_height=img.height(),
        )
        # 在调试画面上标出当前被选中的目标
        img.draw_rectangle(best_blob.rect())
        img.draw_cross(pixel_x, pixel_y)
        write_line(
            uart,
            format_vision_frame(
                seq=report_seq,
                valid=1,
                err_x=err_x,
                err_y=err_y,
            ),
        )
        last_send_ms = now_ms


if __name__ == "__main__":
    run()
