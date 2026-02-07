# OpenART Color Tracking (Direct Control)
# 色块追踪 + 底盘直接控制
# 硬件连接: UART 2 (Tx接底盘Rx, Rx接底盘Tx)

import sensor
import image
import time
import math
from machine import UART

# -----------------------------------------------------------
# 1. 参数配置
# -----------------------------------------------------------
# [控制参数]
Kp_x = 0.0005   # 左右平移 P参数
Kp_y = 0.0005   # 前后移动 P参数
Kp_angle = 0.5  # 转向 P参数 (像素 -> 角度)
DEADZONE_XY = 10  # 像素死区 (中心多少像素内不移动)

# [视觉参数]
# 曝光时间: RGB模式下如果画面太暗，请增大此值
EXP_TIME_US = 500

# [颜色阈值]
# 请务必使用 OpenMV IDE -> Tools -> Threshold Editor 获取准确值
# 格式: (L Min, L Max, A Min, A Max, B Min, B Max)
# 示例: 一个通用的绿色
target_threshold = (0, 100, -128, -13, -128, 127)

# -----------------------------------------------------------
# 2. 串口与硬件初始化
# -----------------------------------------------------------
uart = UART(2, baudrate=115200)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)    # 颜色识别必须用 RGB565
sensor.set_framesize(sensor.QVGA)      # 320x240
sensor.set_vflip(True)      # 垂直翻转
sensor.set_hmirror(True)    # 水平翻转 (配合垂直翻转实现180度旋转)
sensor.skip_frames(time=2000)

# [曝光与白平衡设置]
sensor.set_auto_gain(False)
# 颜色识别必须关闭白平衡，否则阈值会随环境光变动
sensor.set_auto_whitebal(False)
sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)

# -----------------------------------------------------------
# 3. 辅助函数
# -----------------------------------------------------------

def adjust_output(val):
    """
    电机死区补偿：
    当理论输出值绝对值较小时，强制实际输出值绝对值至少为 1cm (0.01m)
    """
    if abs(val) > 1e-5 and abs(val) < 0.01:
        if val > 0:
            return 0.01
        else:
            return -0.01
    return val

# -----------------------------------------------------------
# 4. 主循环
# -----------------------------------------------------------

cx_screen = sensor.width() // 2
# cy_screen 不再使用，因为Y轴改为对齐底边

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()

    # [鱼眼去畸变]
    # RGB565模式下比较耗内存，如果报错请减小 strength 或 zoom
    # 如果不需要去畸变，注释掉下面这行可提高FPS
    try:
        img.lens_corr(strength=2.8, zoom=1.0)
    except MemoryError:
        pass

    # [寻找色块]
    # pixels_threshold: 忽略噪点
    # merge=True: 合并重叠色块
    blobs = img.find_blobs([target_threshold], pixels_threshold=200, area_threshold=200, merge=True)

    if blobs:
        # 寻找最大的色块作为追踪目标
        target_blob = max(blobs, key=lambda b: b.pixels())

        # 1. 计算误差 (基于旋转180度后的图像)
        # X轴: 目标必须在图像水平中心
        dx_raw = target_blob.cx() - cx_screen

        # Y轴Error: 目标底边(y+h) 应该趋向于 图像底边(sensor.height())
        # 距离反转后的底边越远，输出的 dy 越大
        blob_bottom = target_blob.y()
        dy_raw = target_blob.y()

        # 2. 应用死区 logic
        t_dx = 0.0
        if abs(dx_raw) > DEADZONE_XY:
            t_dx = float(dx_raw)

        t_dy = 0.0
        # 如果 Y 轴误差在死区内 (说明目标底边已与图像底边重合)，则 t_dy=0
        # 此时只进行 X 轴对齐 (t_dx 可能不为0)
        if abs(dy_raw) > DEADZONE_XY:
            t_dy = float(dy_raw)

        # 3. 如果都在死区内，发送停止/复位
        if t_dx == 0 and t_dy == 0:
            uart.write("reset=1\n")
        else:
            # 4. 分段控制逻辑
            if t_dy != 0:
                # A. 纵向未到位
                if t_dx != 0:
                    # 1. 如果 X 轴有偏差，优先原地旋转对正 (d_angle)
                    # 使得 dx = 0，此时不进行纵向移动
                    cmd_angle = t_dx * Kp_angle
                    cmd_str = "d_angle=%.4f\n" % (cmd_angle)
                    uart.write(cmd_str)
                else:
                    # 2. X 轴已对正 (t_dx == 0)，再执行 Y 轴移动 (dy)
                    y_cmd_float = t_dy * Kp_y
                    cmd_dy = adjust_output(y_cmd_float)
                    cmd_str = "dy=%.4f\n" % (cmd_dy)
                    uart.write(cmd_str)
            else:
                # B. 纵向已到位 (t_dy=0) -> 使用 平移(dx) 精确对齐
                # 锁定距离后，进行横向平移对准
                x_cmd_float = t_dx * Kp_x
                cmd_dx = adjust_output(x_cmd_float)

                cmd_str = "dx=%.4f\n" % (cmd_dx)
                uart.write(cmd_str)

        # [调试绘图] 在IDE中可以看到框选效果
        img.draw_rectangle(target_blob.rect())
        img.draw_cross(target_blob.cx(), target_blob.cy())

    else:
        # 安全停机
        uart.write("reset=1\n")
