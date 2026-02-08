# OpenART Color Tracking + State Machine Control
# 色块追踪 + 状态机搬运逻辑
# 硬件连接: UART 2 (Tx接底盘Rx, Rx接底盘Tx)

import sensor
import image
import time
import math
from machine import UART

# OpenMV 上的 enum/Enum 在部分固件中会触发 recursion_space 错误。
# 这里使用最简单的整数状态常量，避免任何 Enum/.name 行为。


# -----------------------------------------------------------
# 1. 状态机与任务定义
# -----------------------------------------------------------

class SMState:
    IDLE = 0        # 闲置/搜索
    ALIGN_ANGLE = 1 # 角度对正 (旋转)
    ALIGN_DIST = 2  # 距离对正 (前后)
    ALIGN_DX = 3    # 横向对正 (平移)
    ORBITING = 4    # 绕行转向
    PUSHING = 5     # 执行推操作
    RETURNING = 6   # 执行返回操作
    DONE = 7        # 完成

class TaskConfig:
    def __init__(self, name, threshold, push_angle_deg):
        self.name = name
        self.threshold = threshold
        self.push_angle = push_angle_deg

# [颜色阈值] - 请务必根据实际环境调整
RED_TASK = TaskConfig("Red", (0, 100, 23, 127, -26, 127), -90)    # -90 deg (Left)
GREEN_TASK = TaskConfig("Green", (0, 100, -128, -14, -128, 127), 90) # 90 deg (Right)
TASKS = [RED_TASK, GREEN_TASK]

# -----------------------------------------------------------
# 2. 参数配置
# -----------------------------------------------------------

Kp_x = 0.0005   # 左右平移 P参数
Kp_y = 0.0005   # 前后移动 P参数
Kp_angle = 0.3  # 转向 P参数 (像素 -> 角度)
DEADZONE_XY = 15  # 像素死区 (中心多少像素内不移动)

PUSH_DISTANCE_M = 0.2     # 推行距离 (m)
# 预估推行时间 (用于超时控制，根据底盘实际移动速度调整)
PUSH_DURATION_MS = 2500   # ms
EXP_TIME_US = 500

Kp_push_correct = 0.0001 # 推进过程中的修正系数 (米/像素)

# -----------------------------------------------------------
# 3. 硬件初始化
# -----------------------------------------------------------

uart = UART(2, baudrate=115200)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) # 320x240
sensor.set_vflip(True)      # 倒装修正
sensor.set_hmirror(True)    # 倒装修正
sensor.skip_frames(time=2000) # type: ignore
sensor.set_auto_gain(False) # type: ignore
sensor.set_auto_whitebal(False)
sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)

cx_screen = sensor.width() // 2

# -----------------------------------------------------------
# 3. 辅助函数
# -----------------------------------------------------------

def normalize_angle(angle):
    """将角度标准化到 -180 到 180 度"""
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

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

class RobotController:
    def __init__(self, uart_obj):
        self.uart = uart_obj
        self.x = 0
        self.y = 0
        self.angle = 0
        self.push_start_x = 0
        self.push_start_y = 0
        self.push_target_x = 0
        self.push_target_y = 0

    def _write_line(self, cmd_str):
        # 低层写串口：只负责发送 + 小延时，避免总线冲突
        time.sleep(0.002)
        self.uart.write(cmd_str + "\r\n")
        time.sleep(0.002) # 短暂延时确保发送完成

    def send_cmd(self, cmd_str):
        # 普通发送（不做 lock 同步）
        self._write_line(cmd_str)

    def send_cmd_sync(self, cmd_str, timeout_ms=5000, wait_after=False):
        # 同步发送：保证按 lock 顺序执行（用于离散动作/阶段切换）
        self.wait_until_idle(timeout_ms=timeout_ms)
        self._write_line(cmd_str)
        if wait_after:
            self.wait_until_idle(timeout_ms=timeout_ms)

    def print_msg(self, msg):
        # 打印不参与 lock 同步，避免与 wait_until_idle 互相递归
        self._write_line("print=" + str(msg))


    def _read_line(self, timeout_ms):
        buf = b""
        start = time.ticks_ms()
        res = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self.uart.any():
                chunk = self.uart.read()
                if chunk:
                    buf += chunk
                    if b"\n" in buf:
                        line, _, _ = buf.partition(b"\n")
                        res = line.rstrip(b"\r")
                        break
            time.sleep(0.002)
        time.sleep(0.002) # 短暂延时确保读取完成
        return res

    def update_pose(self):
        # 清空缓冲区
        while self.uart.any():
            self.uart.read()

        # 发送新协议查询指令 ?pos
        self._write_line("?pos")
        line = self._read_line(30)
        if line.startswith(b"?pos="):
            parts = line[5:].split(b',')
        elif line.startswith(b"pos="):
            parts = line[4:].split(b',')
        else:
            return
        if len(parts) >= 3:
            try:
                self.x = float(parts[0])
                self.y = float(parts[1])
                self.angle = float(parts[2])
            except ValueError:
                pass

    def is_locked(self):
        """查询底盘是否处于锁定(运动)状态"""
        # 清空缓冲区
        while self.uart.any():
            self.uart.read()

        # 注意：查询不能走 send_cmd_sync，否则会递归
        self._write_line("?lock")
        line = self._read_line(50)
        # self.print_msg("LockResp:" + str(line))
        if line.startswith(b"?lock="):
            payload = line[6:]
        elif line.startswith(b"lock="):
            payload = line[5:]
        else:
            return False
        try:
            return int(payload) == 1
        except ValueError:
            return False

    def wait_until_idle(self, timeout_ms=3000):
        """阻塞等待直到底盘解锁(运动完成)"""
        # 给一点时间让之前的指令生效进入Lock状态
        time.sleep(0.05)

        start = time.ticks_ms()
        # self.print_msg("Waiting idle...")
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if not self.is_locked():
                # self.print_msg("Wait idle done")
                return True
            time.sleep(0.05)
        # 超时提示也不能用 print_msg -> send_cmd_sync 链路
        self._write_line("print=Wait idle timeout")
        return False

    def init_push_target(self, distance_m):
        """初始化推行任务，计算世界坐标终点"""
        # 推行是离散动作：发出前确保底盘空闲（lock=0）
        self.send_cmd_sync("dy=%.4f" % distance_m, timeout_ms=5000, wait_after=False)

    def update_push_correction(self, angle_deg, correction_m):
        """
        在推行过程中修正终点目标
        angle_deg: 当前推行的主方向
        correction_m: 横向修正量 (+为向左修正)
        """
        # 计算修正向量 (垂直于运动方向)
        # 运动方向向量: (sin(a), cos(a))
        # 垂直向量(左): (-cos(a), sin(a))  <-- 逆时针90度
        # 验证: Angle=0(Y+), Left=(-1, 0)(X-). Right?
        # Protocol: X+ is Right. So Left is X-.
        # Left Vector for Angle=0 (0,1) should be (-1, 0).
        # Formula: -cos(0)=-1, sin(0)=0. Correct.

        rad = math.radians(angle_deg)
        lat_dx = correction_m * (-math.cos(rad))
        lat_dy = correction_m * math.sin(rad)

        # 更新目标点
        self.push_target_x += lat_dx
        self.push_target_y += lat_dy

        # 发送更新后的绝对位置指令 (角度保持不变)
        self.send_cmd_sync("angle=%.1f,x=%.4f,y=%.4f" % (angle_deg, self.push_target_x, self.push_target_y), timeout_ms=5000, wait_after=False)

    def return_to_start(self, angle_deg):
        """返回到推行起点"""
        # 直接去起点的绝对坐标
        self.send_cmd_sync("angle=%.1f,x=%.4f,y=%.4f" % (angle_deg, self.push_start_x, self.push_start_y), timeout_ms=5000, wait_after=False)

    def turn_relative(self, d_angle):
        target_angle = self.angle + d_angle
        self.send_cmd_sync("angle=%.4f" % target_angle, timeout_ms=5000, wait_after=False)

    def stop(self):
        # 使用速度0来停止（相对安全）
        # stop 需要尽快生效，不能等待 lock
        self.send_cmd("vx=0,vy=0,omega=0")

    def reset(self):
        self.send_cmd_sync("reset=1", timeout_ms=5000, wait_after=False)
        self.x = 0
        self.y = 0
        self.angle = 0

# -----------------------------------------------------------
# 5. 主逻辑
# -----------------------------------------------------------

robot = RobotController(uart)
state = SMState.IDLE
current_task = TASKS[0]
state_start_time = 0
push_last_correction_ms = 0

robot.reset()
robot.print_msg("SystemStart")
time.sleep(1)

while True:
    img = sensor.snapshot()

    try:
        img.lens_corr(strength=2.8, zoom=1.0)
    except MemoryError:
        pass

    # [关键] 每帧更新姿态 -> 改为按需更新，节省通信资源
    # robot.update_pose()

    # [显示当前状态]
    state_str = "UNK"
    if state == SMState.IDLE: state_str = "IDLE"
    elif state == SMState.ALIGN_ANGLE: state_str = "ALIGN_ANGLE"
    elif state == SMState.ALIGN_DIST: state_str = "ALIGN_DIST"
    elif state == SMState.ALIGN_DX: state_str = "ALIGN_DX"
    elif state == SMState.ORBITING: state_str = "ORBITING"
    elif state == SMState.PUSHING: state_str = "PUSHING"
    elif state == SMState.RETURNING: state_str = "RETURNING"
    elif state == SMState.DONE: state_str = "DONE"

    img.draw_string(10, 10, state_str, color=(255, 0, 0), scale=2)
    # 显示当前的绝对坐标，便于调试
    pose_str = "X:%.2f Y:%.2f A:%.1f" % (robot.x, robot.y, robot.angle)
    img.draw_string(10, 30, pose_str, color=(255, 0, 0), scale=2)

    if state == SMState.IDLE:
        found = False
        for task in TASKS:
            blobs = img.find_blobs([task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
            if blobs:
                current_task = task
                state = SMState.ALIGN_ANGLE
                robot.print_msg("State:ALIGN_ANGLE_Task:" + task.name)
                found = True
                break

        if not found:
            # 停止
            # robot.stop()
            pass

    elif state == SMState.ALIGN_ANGLE:
        # [状态1] 角度对正: 旋转车身使目标处于画面中心 (dx -> 0)
        blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if blobs:
            # [User Request] 选择距离屏幕中线(cx)和底边(height)最近的目标
            target_blob = min(blobs, key=lambda b: (b.cx() - cx_screen)**2 + (b.cy() - img.height())**2)
            dx_raw = target_blob.cx() - cx_screen

            # 角度死区
            if abs(dx_raw) > DEADZONE_XY:
                # 旋转对正
                cmd_angle = float(dx_raw) * Kp_angle
                robot.send_cmd_sync("d_angle=%.4f" % cmd_angle, timeout_ms=5000, wait_after=False)
            else:
                # 角度满足，进入距离对正
                # robot.stop()
                state = SMState.ALIGN_DIST
                robot.print_msg("State:ALIGN_DIST")

            img.draw_rectangle(target_blob.rect())
            img.draw_cross(target_blob.cx(), target_blob.cy())
        else:
            state = SMState.IDLE
            robot.print_msg("State:IDLE_Lost")

    elif state == SMState.ALIGN_DIST:
        # [状态2] 距离对正: 前后移动使目标达到合适距离 (dy -> target)
        blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if blobs:
            # [User Request] 选择距离屏幕中线(cx)和底边(height)最近的目标
            target_blob = min(blobs, key=lambda b: (b.cx() - cx_screen)**2 + (img.height() - b.cy())**2)
            dx_raw = target_blob.cx() - cx_screen

            # 如果旋转误差甚至过大，回退到 ALIGN_ANGLE 状态
            if abs(dx_raw) > DEADZONE_XY:
                state = SMState.ALIGN_ANGLE
                robot.print_msg("State:ALIGN_ANGLE_Re")
            else:
                # [Fix] 使用索引访问 [1] (y) 避免方法/属性调用歧义
                dy_raw = target_blob[1]

                if abs(dy_raw) > DEADZONE_XY:
                    cmd_dy = float(dy_raw) * Kp_y
                    cmd_dy = adjust_output(cmd_dy)
                    robot.send_cmd_sync("dy=%.4f" % cmd_dy, timeout_ms=5000, wait_after=False)
                else:
                    # 距离满足，进入横向对齐准备 (或Orbit检查)
                    # robot.stop()
                    state = SMState.ALIGN_DX
                    robot.print_msg("State:ALIGN_DX")

            img.draw_rectangle(target_blob.rect())
            img.draw_cross(target_blob.cx(), target_blob.cy())
        else:
            state = SMState.IDLE
            robot.print_msg("State:IDLE_Lost")

    elif state == SMState.ALIGN_DX:
        # [状态3] 横向对正: 平移车身再次精细对准中心，准备判断推行逻辑
        blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if blobs:
            # [User Request] 选择距离屏幕中线(cx)和底边(height)最近的目标
            target_blob = min(blobs, key=lambda b: (b.cx() - cx_screen)**2 + (img.height() - b.cy())**2)
            dx_raw = target_blob.cx() - cx_screen

            # 这里的对正使用平移 (dx) 而不是旋转
            if abs(dx_raw) > 10:
                # 平移对正
                cmd_dx = float(dx_raw) * Kp_x
                cmd_dx = adjust_output(cmd_dx)

                # 限制速度
                cmd_dx = max(min(cmd_dx, 0.1), -0.1)

                robot.send_cmd_sync("dx=%.4f" % cmd_dx, timeout_ms=5000, wait_after=False)
                img.draw_string(10, 50, "AlignDX:%.3f" % cmd_dx, color=(0, 255, 0))
            else:
                # 横向对正完成，检查全局角度是否满足推行要求
                # robot.stop()
                time.sleep(0.1) # 等待停稳
                robot.update_pose()
                angle_diff = normalize_angle(current_task.push_angle - robot.angle)

                img.draw_string(10, 50, "AngDiff:%.1f" % angle_diff, color=(0, 255, 0))

                if abs(angle_diff) < 10:
                    # 角度达标，进入推行
                    state = SMState.PUSHING
                    robot.print_msg("State:PUSHING")
                    state_start_time = time.ticks_ms()
                    push_last_correction_ms = 0
                    robot.init_push_target(PUSH_DISTANCE_M)
                else:
                    # 角度不达标，进入绕行调整
                    state = SMState.ORBITING
                    robot.print_msg("State:ORBITING")

            img.draw_rectangle(target_blob.rect())
            img.draw_cross(target_blob.cx(), target_blob.cy())
        else:
            state = SMState.IDLE
            robot.print_msg("State:IDLE_Lost")

    elif state == SMState.ORBITING:
        # [绕行] 负责执行旋转动作
        # 旋转后，物体在画面中位置会变，因此转完后必须回到 ALIGN_DX 重新对正

        # [User Request] 使用绝对角度控制，配合 rear=1 模式
        target_angle = current_task.push_angle
        robot.send_cmd_sync("rear=1,angle=%.1f" % target_angle, timeout_ms=5000, wait_after=True)
        robot.print_msg("OrbitAbs:%.1f" % target_angle)

        # 转完后回到 DX 对正状态
        state = SMState.ALIGN_DX
        robot.print_msg("State:ALIGN_DX")

    elif state == SMState.PUSHING:
        # [推行] 过程不中断：持续做与 ALIGN_DX 类似的横向对正(发送 dx)，但不切状态
        now_ms = time.ticks_ms()

        # 周期性纠偏，避免每帧疯狂发串口
        if push_last_correction_ms == 0 or time.ticks_diff(now_ms, push_last_correction_ms) > 120:
            push_last_correction_ms = now_ms
            blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
            if blobs:
                target_blob = min(blobs, key=lambda b: (b.cx() - cx_screen)**2 + (img.height() - b.cy())**2)
                dx_raw = target_blob.cx() - cx_screen

                if abs(dx_raw) > 10:
                    cmd_dx = float(dx_raw) * Kp_x
                    cmd_dx = adjust_output(cmd_dx)
                    # 推行中纠偏幅度更小，避免把前进打断得太厉害
                    cmd_dx = max(min(cmd_dx, 0.05), -0.05)
                    robot.send_cmd("dx=%.4f" % cmd_dx)

                img.draw_rectangle(target_blob.rect())
                img.draw_cross(target_blob.cx(), target_blob.cy())

        # 推行结束判定：用 lock=0 而不是超时
        if time.ticks_diff(now_ms, state_start_time) > 2000:
            if not robot.is_locked():
                state = SMState.RETURNING
                robot.print_msg("State:RETURNING")
                state_start_time = now_ms

    elif state == SMState.RETURNING:
        # [User Request] 向后转 180 度
        robot.send_cmd_sync("d_angle=180", timeout_ms=5000, wait_after=True)
        robot.print_msg("Return:180")

        state = SMState.DONE
        robot.print_msg("State:DONE")

    elif state == SMState.DONE:
        # robot.stop()
        time.sleep(2)
        state = SMState.IDLE
        robot.print_msg("State:IDLE")
