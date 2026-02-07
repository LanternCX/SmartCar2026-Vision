# OpenART Color Tracking + State Machine Control
# 色块追踪 + 状态机搬运逻辑
# 硬件连接: UART 2 (Tx接底盘Rx, Rx接底盘Tx)

import sensor
import image
import time
import math
from machine import UART

# 尝试导入 Enum (OpenMV兼容)
try:
    from enum import Enum, auto # type: ignore
except ImportError:
    # 简易 Mock 实现
    _auto_counter = 0
    def auto():
        global _auto_counter
        val = _auto_counter
        _auto_counter += 1
        return val
    
    class Enum: # type: ignore
        pass


# -----------------------------------------------------------
# 1. 状态机与任务定义
# -----------------------------------------------------------

class SMState(Enum):
    IDLE = auto()       # 闲置/搜索
    TRACKING = auto()   # 视觉伺服追踪
    PUSHING = auto()    # 执行推操作
    RETURNING = auto()  # 执行返回操作
    DONE = auto()       # 完成

class TaskConfig:
    def __init__(self, name, threshold, push_angle_deg):
        self.name = name
        self.threshold = threshold
        self.push_angle = push_angle_deg

# [颜色阈值] - 请务必根据实际环境调整
RED_TASK = TaskConfig("Red", (0, 100, 23, 127, -26, 127), 0)    # 0 deg (Forward)
GREEN_TASK = TaskConfig("Green", (0, 100, -128, -14, -128, 127), 90) # 90 deg (Right)
TASKS = [RED_TASK, GREEN_TASK]

# -----------------------------------------------------------
# 2. 参数配置
# -----------------------------------------------------------

Kp_x = 0.0005   # 左右平移 P参数
Kp_y = 0.0005   # 前后移动 P参数
Kp_angle = 0.5  # 转向 P参数 (像素 -> 角度)
DEADZONE_XY = 15  # 像素死区 (中心多少像素内不移动)

PUSH_DISTANCE_M = 0.3     # 推行距离 (m)
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
# 4. 辅助控制器
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

    def send_cmd(self, cmd_str):
        self.uart.write(cmd_str + "\n")
    
    def print_msg(self, msg):
        self.send_cmd("print=" + str(msg))
    
    def update_pose(self):
        # 清空缓冲区
        while self.uart.any():
            self.uart.read()
            
        # 发送新协议查询指令 ?pos
        self.uart.write("?pos\n")
        
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < 30: # 30ms超时
            if self.uart.any():
                try:
                    line = self.uart.read().decode().strip()
                    # 期望格式: ?pos=x,y,yaw
                    if line.startswith("?pos="):
                        parts = line[5:].split(',')
                        if len(parts) >= 3:
                            self.x = float(parts[0])
                            self.y = float(parts[1])
                            self.angle = float(parts[2])
                        break
                except:
                    pass
            time.sleep(0.002)

    def init_push_target(self, angle_deg, distance_m):
        """初始化推行任务，计算世界坐标终点"""
        self.update_pose()
        self.push_start_x = self.x
        self.push_start_y = self.y
        rad = math.radians(angle_deg)
        # 理论终点
        self.push_target_x = self.x + distance_m * math.sin(rad)
        self.push_target_y = self.y + distance_m * math.cos(rad)
        
        # 发送初始绝对位置指令
        self.send_cmd("angle=%.1f,x=%.4f,y=%.4f" % (angle_deg, self.push_target_x, self.push_target_y))

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
        self.send_cmd("angle=%.1f,x=%.4f,y=%.4f" % (angle_deg, self.push_target_x, self.push_target_y))

    def return_to_start(self, angle_deg):
        """返回到推行起点"""
        # 直接去起点的绝对坐标
        self.send_cmd("angle=%.1f,x=%.4f,y=%.4f" % (angle_deg, self.push_start_x, self.push_start_y))

    def stop(self):
        # 使用速度0来停止（相对安全）
        self.send_cmd("vx=0,vy=0,omega=0")

    def reset(self):
        self.send_cmd("reset=1")
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
    state_name = getattr(state, "name", None)
    if state_name:
        state_str = state_name
    else:
        if state == SMState.IDLE: state_str = "IDLE"
        elif state == SMState.TRACKING: state_str = "TRACKING"
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
                print("Found task:", task.name)
                current_task = task
                state = SMState.TRACKING
                robot.print_msg("State:TRACKING_Task:" + task.name)
                found = True
                break
        
        if not found:
            # 停止
            robot.stop()

    elif state == SMState.TRACKING:
        blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if blobs:
            target_blob = max(blobs, key=lambda b: b.pixels())

            dx_raw = target_blob.cx() - cx_screen
            dy_raw = target_blob.y()
            
            # 死区
            t_dx = 0.0
            if abs(dx_raw) > DEADZONE_XY: t_dx = float(dx_raw)
            t_dy = 0.0
            if abs(dy_raw) > DEADZONE_XY: t_dy = float(dy_raw)

            if t_dx == 0 and t_dy == 0:
                print("Aligned. Pushing...")
                robot.stop()
                time.sleep(0.5)
                state = SMState.PUSHING
                robot.print_msg("State:PUSHING")
                state_start_time = time.ticks_ms()
                # 初始化推行 (记录起点，计算并发送绝对终点)
                robot.init_push_target(current_task.push_angle, PUSH_DISTANCE_M)
            else:
                # 追踪控制 (保持原始逻辑)
                if t_dy != 0:
                    if t_dx != 0:
                        cmd_angle = t_dx * Kp_angle
                        robot.send_cmd("d_angle=%.4f" % (cmd_angle))
                    else:
                        y_cmd_float = t_dy * Kp_y
                        cmd_dy = adjust_output(y_cmd_float)
                        robot.send_cmd("dy=%.4f" % (cmd_dy))
                else:
                    x_cmd_float = t_dx * Kp_x
                    cmd_dx = adjust_output(x_cmd_float)
                    robot.send_cmd("dx=%.4f" % (cmd_dx))

            img.draw_rectangle(target_blob.rect())
            img.draw_cross(target_blob.cx(), target_blob.cy())
            
        else:
            print("Lost target.")
            state = SMState.IDLE
            robot.print_msg("State:IDLE_LostTarget")
            robot.stop()

    elif state == SMState.PUSHING:
        # [PUSHING 闭环逻辑]
        # 尝试寻找色块进行纠偏
        blobs = img.find_blobs([current_task.threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if blobs:
            target_blob = max(blobs, key=lambda b: b.pixels())
            dx_raw = target_blob.cx() - cx_screen
            
            # 如果偏差过大 (例如超过 20 像素)，进行修正
            # [Fix] 逻辑修正：与 TRACKING 保持一致
            # dx_raw < 0 (物体在左)，需要车往左移 -> Correction > 0 (根据 update_push_correction 定义)
            if abs(dx_raw) > 20: 
                # 注意这里的负号: dx_raw 为负时(左)，我们需要 positive correction (左移)
                correction = -float(dx_raw) * Kp_push_correct

                # 限制单次修正幅度，防止震荡
                if correction > 0.05: correction = 0.05
                if correction < -0.05: correction = -0.05
                
                robot.update_push_correction(current_task.push_angle, correction)
                img.draw_string(10, 50, "Correct:%.3f" % correction, color=(0, 255, 0))

            img.draw_rectangle(target_blob.rect())
            img.draw_cross(target_blob.cx(), target_blob.cy())

        # 超时检查
        if time.ticks_diff(time.ticks_ms(), state_start_time) > PUSH_DURATION_MS:
            print("Push Done. Returning...")
            state = SMState.RETURNING
            robot.print_msg("State:RETURNING")
            state_start_time = time.ticks_ms()
            # 返回起点 (绝对坐标)
            robot.return_to_start(current_task.push_angle)

    elif state == SMState.RETURNING:
        if time.ticks_diff(time.ticks_ms(), state_start_time) > PUSH_DURATION_MS:
            print("Return Done.")
            state = SMState.DONE
            robot.print_msg("State:DONE")
            
    elif state == SMState.DONE:
        robot.stop()
        time.sleep(2)
        state = SMState.IDLE
        robot.print_msg("State:IDLE")
