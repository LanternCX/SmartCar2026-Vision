# 2025 智能车蚂蚁搬家组 - OpenART 视觉系统代码

## 项目概述

本项目为智能车竞赛蚂蚁搬家组的视觉上位机代码，运行于 OpenART 模块（MicroPython 环境）。系统架构设计时考虑到搬运车底盘代码的高复用性，将主要的**业务状态机逻辑**和**视觉识别算法**均部署在 OpenART 端，底盘仅负责执行具体的运动指令。

## 开发环境

开发环境延续了搬运车模的 VS Code + MicroPython 方案。

1.  **IDE**: Visual Studio Code
2.  **插件**: Pylance, Python
3.  **智能提示 (Stubs)**: 使用了第三方的 MicroPython stubs 以获得代码补全和类型检查支持。

## 核心功能

系统功能主要划分为两大部分：**视觉识别 (Vision)** 和 **状态机与通信 (FSM & Communication)**。

### 1. 视觉识别架构 (Vision Architecture)

目前主要基于普通的色块识别（Color Blob Detection）实现，鉴于目标物体特征（沙包）形态不规则，未采用 `findRect` 等矩形检测算法。

为了应对未来可能引入的更复杂的识别需求（如 Yolo），可以借鉴我机械臂项目，设计一个 **Detector 路由**。所有的检测器均继承自统一的抽象基类，保证接口一致性。

```python
# 检测器抽象基类，基于此架构可以方便地扩展多种识别算法并进行路由管理
class BaseDetector(ABC):
    """
    所有检测器的基类
    """
    def __init__(self, tag):
        self.tag = tag

    @abstractmethod
    def detect(self, frame):
        """
        在图像中检测目标
        返回值: boxes, frame
        boxes: [((x1, y1), (x2, y2)), ...]
        frame: 可视化后的图像
        """
        pass
```

### 2. 状态机控制 (State Machine)

为了解耦视觉决策与底盘运动，我们将状态机逻辑运行在 OpenART 上。状态机定义了搬运任务的完整生命周期：

```python
IDLE = 0        # 闲置/搜索状态
ALIGN_ANGLE = 1 # 角度对正 (旋转调整)
ALIGN_DIST = 2  # 距离对正 (前后调整)
ALIGN_DX = 3    # 横向对正 (左右平移)
ORBITING = 4    # 绕行转向 (复杂轨迹)
PUSHING = 5     # 执行推操作
RETURNING = 6   # 执行返回操作
DONE = 7        # 任务完成
```

### 3. 通信协议与封装 (Communication Protocol)

本项目对底层串口协议进行了二次封装，实现了更可靠的指令发送机制。底盘通信波特率较高，且存在半双工总线竞争风险，因此通信层的实现细节至关重要。

#### 3.1 底层写入保护

为了避免串口总线占用导致的冲突，在 `_write_line` 方法中强制加入了微小的延时。由于 UART `write` 是异步操作，而物理串口发送是同步串行的，连续快速调用可能会导致后一条消息覆盖前一条消息或导致总线拥塞。

```python
def _write_line(self, cmd_str):
    time.sleep(0.002)
    self.uart.write(cmd_str + "\r\n")
    time.sleep(0.002) # 短暂延时确保发送完成
```

#### 3.2 发送策略：同步 vs 异步

针对不同的控制场景，封装了两种发送模式：

**A. 普通发送 (Async/No-lock)**

适用于**视觉闭环控制**过程。此类指令发送频率高（如连续的速度修正指令），即使单条指令丢失，后续的闭环回路也会迅速补充新的修正指令。此模式不检查底盘的 `lock` 状态，类似于 UDP 协议。

```python
def send_cmd(self, cmd_str):
    # 普通发送（不做 lock 同步）
    self._write_line(cmd_str)
```

**B. 同步发送 (Sync/Lock-check)**

适用于**状态切换**或**关键离散动作**（如“移动到指定位置”、“重置状态”）。此类指令要求必须被执行，且执行期间不能被其他指令打断。发送前会轮询底盘的 `lock` 状态，直到底盘空闲（Idle）才发送指令。

```python
def send_cmd_sync(self, cmd_str, timeout_ms=5000, wait_after=False):
    # 同步发送：保证按 lock 顺序执行（用于离散动作/阶段切换）
    # 阻塞等待底盘解锁
    self.wait_until_idle(timeout_ms=timeout_ms)
    self._write_line(cmd_str)
    
    # 可选：发送后继续等待直到动作执行完毕（再次解锁）
    if wait_after:
        self.wait_until_idle(timeout_ms=timeout_ms)
```

> **最佳实践**：在没有明确的高频闭环需求时，建议默认使用 `send_cmd_sync`，这能显著减少因指令覆盖或状态冲突导致的 Bug。

## 待优化项 (Future Work)

1.  **消息队列 (Message Queue)**：目前虽然通过 `sleep` 和 `lock` 检查解决了大部分问题，但当大量指令涌入时，仍可能存在顺序执行无保证的问题。计划引入简单的软件消息队列来管理指令发送顺序。
2.  **协议深度封装**：目前的封装主要在传输层，计划参考 ESP32 或 Arduino 的库封装思路，将所有协议命令封装为 Python 对象/方法，对上层完全屏蔽字符串拼接细节。
