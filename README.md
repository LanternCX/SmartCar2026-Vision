# SmartCar2025 - OpenMV Vision

这是一个为智能车竞赛配置的 OpenMV 视觉项目，当前按“物理相机 ID + 可重叠类别”的双摄协议与 RT1021 主控协同。

## 项目治理与开发流程

- 统一代理规则入口: `AGENTS.md`
- 项目技能目录: `.agents/skills`
- 开发必须遵循 TDD（RED -> GREEN -> REFACTOR）
- 本仓库保留**单文件架构**: 运行时代码主入口固定为 `main.py`
- 远端规则更新机制文档位于: `docs/problem_statement/`
- 通信协议说明文档位于: `docs/Protocol.md`，用于说明 OpenArt 与 RT1021 之间的串口通信协议、视觉上报格式及兼容控制命令
# 2025 智能车蚂蚁搬家组 - OpenART 视觉系统代码

本项目为智能车竞赛蚂蚁搬家组的视觉上位机代码，运行于 OpenART 模块（MicroPython 环境）。系统架构设计时考虑到搬运车底盘代码的高复用性，将主要的**业务状态机逻辑**和**视觉识别算法**均部署在 OpenART 端，底盘仅负责执行具体的运动指令。

## 开发环境

开发环境延续了搬运车模的 VS Code + MicroPython 方案。

1.  **IDE**: Visual Studio Code
2.  **插件**: Pylance, Python
3.  **智能提示 (Stubs)**: 使用了第三方的 MicroPython stubs 以获得代码补全和类型检查支持。

## 当前协议要点

- `camera_id` 只表示物理相机身份，例如 `cam_a`、`cam_b`
- `category` 只表示检测类别，例如 `cargo`、`follower`、`obstacle`
- 同一 `category` 可以在不同 `camera_id` 上重复出现
- OpenART 不再主动连续发包，而是仅在收到 `?frame=<camera_id>` 时回传该物理相机当前缓存帧
- 单次查询响应允许返回 `0..N` 条检测，最后必须追加 `frame_end=1`
- 未被点名的相机必须严格静默，即使它也看到了相同类别目标

最小示例：

```text
?frame=cam_a
camera_id=cam_a,frame_id=12,category=cargo,left=100,top=20,right=140,bottom=90
camera_id=cam_a,frame_id=12,category=follower,left=150,top=25,right=190,bottom=95
camera_id=cam_a,frame_id=12,frame_end=1

?frame=cam_b
camera_id=cam_b,frame_id=33,category=cargo,left=120,top=18,right=170,bottom=110
camera_id=cam_b,frame_id=33,category=obstacle,left=20,top=30,right=80,bottom=140
camera_id=cam_b,frame_id=33,frame_end=1
```

## 核心功能

系统功能当前主要划分为两部分：**视觉识别 (Vision)** 和 **查询/响应通信 (Query/Response Communication)**。

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

### 2. 通信协议与封装 (Communication Protocol)

本项目当前按查询/响应方式与 RT1021 通信。视觉端负责维护本物理相机最新缓存帧，并在查询到来时回传多检测结果。

#### 3.1 底层写入保护

为了避免串口总线占用导致的冲突，在 `_write_line` 方法中强制加入了微小的延时。由于 UART `write` 是异步操作，而物理串口发送是同步串行的，连续快速调用可能会导致后一条消息覆盖前一条消息或导致总线拥塞。

```python
def _write_line(self, cmd_str):
    time.sleep(0.002)
    self.uart.write(cmd_str + "\r\n")
    time.sleep(0.002) # 短暂延时确保发送完成
```

#### 2.2 查询/响应策略

- 运行时主链路是 `?frame=<camera_id>` 查询
- 只有被点名的物理相机允许回包
- 每一帧回包都必须以 `frame_end=1` 结束
- 主车负责跨相机仲裁，视觉端只提供观测，不承担任务状态机

联调建议：

1. 先在主控端发送 `?frame=cam_a`，确认只收到 `cam_a` 回包
2. 再发送 `?frame=cam_b`，确认只收到 `cam_b` 回包
3. 人工制造类别重叠场景，确认两颗相机都可以合法返回 `cargo` 或 `follower`
4. 在 RT1021 侧通过 `?vision` 观察主车如何完成跨相机仲裁

## 待优化项 (Future Work)

1.  **多类别检测稳定性**：继续优化重叠类别场景下的检测稳定性，降低帧间抖动。
2.  **查询响应健壮性**：补充更多 UART 粘包、空帧和异常查询场景测试。
3.  **类别扩展能力**：在不改变 `camera_id` 物理语义的前提下扩展更多 `category`。
