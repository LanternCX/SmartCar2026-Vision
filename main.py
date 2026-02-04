# OpenMV/OpenART AprilTag Localization
# 硬件连接: UART 2 (Strictly enforced)

import sensor
import image
import time
import math
from machine import UART

# -----------------------------------------------------------
# 1. 串口初始化 (严格保留你的 UART 2 设置)
# -----------------------------------------------------------
uart = UART(2, baudrate=115200)
uart.write("SYSTEM_START\r\n")

# -----------------------------------------------------------
# 2. 传感器初始化 (保留你的配置)
# -----------------------------------------------------------
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # 320x240
sensor.skip_frames(time=2000)

# 曝光设置 (保留你的 800us)
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False, exposure_us=800)
sensor.set_auto_whitebal(False)

# -----------------------------------------------------------
# 3. 变量准备
# -----------------------------------------------------------
# 注意：根据你的固件版本，如果报错请改回 image.TAG36H11Tag
tag_families = image.TAG36H11
cx_screen = sensor.width() // 2
cy_screen = sensor.height() // 3
clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()

    # -----------------------------------------------------------
    # 核心修改：sigma=0
    # -----------------------------------------------------------
    # 之前的 3.7 FPS 全是因为 sigma=1.0。
    # 改为 0 后，不做高斯模糊，速度将直接恢复正常。
    tags = img.find_apriltags(families=tag_families, decimation=2, sigma=0)

    fps = clock.fps()

    if tags:
        tag = tags[0]

        dx = tag.cx() - cx_screen
        dy = tag.cy() - cy_screen

        angle = math.degrees(tag.rotation())
        if angle > 180:
            angle -= 360

        # 串口发送 (保留你的格式)
        packet = f"id:{tag.id()},dx:{dx},dy:{dy},angle:{angle:.1f},fps:{fps:.1f}\r\n"
        uart.write(packet)

    else:
        # 丢失目标
        uart.write(f"$NONE,{fps:.1f}\r\n")
