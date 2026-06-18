# SmartCar2026 - OpenART Vision

这是智能车竞赛 OpenART 视觉项目, 运行于 OpenART MicroPython 环境。

## 仓库定位

本仓库作为 `../SmartCar2026-TransportCar` 的附属视觉仓库维护。

本仓库维护三类内容:

1. OpenART 视觉运行时代码。
2. 视觉协议行为测试与回归测试。
3. 面向本仓库使用者的必要说明。

项目规则、题面材料、协作规格、协作计划和长期开发文档以 `../SmartCar2026-TransportCar` 为准。

## 开发边界

- 运行时代码按角色维护在 `assistant/` 与 `master/` 目录。
- 每个角色目录独立维护面向 OpenART 设备部署的构建脚本。
- 本仓库不维护独立 `.agents/skills` 体系。
- 本仓库不维护题面规则文档副本。
- 文档、注释和规则入口通过 review 检查。
- 测试覆盖运行时行为、协议行为和回归场景。

## 入口接口风格

- `master/main_v2.py` 与 `assistant/main_v2.py` 的接口风格保持一致。
- 当前轮次真正处理的对象应显式传入接口, 例如 `img`。
- 由当前处理对象可直接得到的派生信息, 例如图像宽高, 不再作为参数层层下传。
- 当前运行态只维护一份的共享状态, 统一放在入口级 `state` 对象中读取。

## 角色职责

当前视觉链路统一使用固定长度短帧:

- 每帧固定 15 字节, 由 `head/mode/topic/seq/body[10]/crc8` 组成。
- `head=0xA5`, `crc8` 覆盖 `mode/topic/seq/body[10]`, 用于串口字节流重新对齐。
- 本地视觉速度使用 `MODE_UDP + TOPIC_LOCAL_VISION_VELOCITY`。
- 本地视觉同步与事件回报使用 `MODE_TCP`。
- 可靠确认使用 `MODE_ACK`, `topic` 与被确认的可靠帧保持一致。

### OpenART Vision master

`master/main.py` 运行在主车 OpenART 上, 负责物体搜索视觉链路:

- 接收 RT1021 下发的主车视觉同步帧, body 字段为 `context_id/state/target/arg`。
- 使用主车视觉同步 topic 的 ACK 帧确认可靠同步包。
- `master/main_v2.py` 按阶段切换识别方式：第一次找目标使用 YOLO 与色块混合，搬运前最后对正只使用 YOLO，绕行修正与搬运结束判定只使用色块，回库黄线任务跳过物体 YOLO。
- 主车纯 YOLO 阶段按可调的隔帧间隔触发模型推理，间隔帧沿用上一帧目标结果，避免每帧都发停车控制。
- 基于候选目标识别框中心点计算搜索 P 环。
- 输出主车搜索速度短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 速度短帧独立于 task 上下文, 每帧直接根据当前识别开关选择的结果输出。
- 在 task 条件满足时输出主车视觉事件回报帧, body 字段为 `context_id/event/value`。
- `arg=1` 表示主车物体搜索 task 配置, 稳定满足条件后回报 `TARGET_FOUND=6`。
- `arg=2` 表示主车搬运入口对正 task 配置, 稳定满足条件后回报 `ALIGNED=7`。
- `arg=5` 表示主车回库黄线 task 配置, 后退段回报 `RETURN_LINE_ALIGNED=10`, 平移段在有效跟随区域连续 5 帧算不出黄线时回报 `RETURN_GARAGE_FINISHED=12`。
- 主车搜索目标点按 task 配置编号切换：`arg=1` 使用寻找阶段目标点，默认 `x=160, y=210`；`arg=2` 使用搬运入口对正目标点，默认 `x=160, y=240`。
- task 判定使用物体中心相对目标点的横向误差。
- task 判定使用物体底边相对目标点的纵向误差。
- task 判定使用候选目标面积。
- 主车搜索控制使用 `v` 数据流, 不依赖周期 `o` 观测包。

### OpenART Vision assistant

`assistant/main.py` 运行在辅车 OpenART 上, 负责辅车跟随主车色标和辅车找目标物体:

- 跟随模式识别主车色标。
- 找物体模式可通过代码开关选择色块阈值或 YOLO 模型识别目标物体。
- `assistant/main_v2.py` 按阶段切换识别方式：第一次找目标使用 YOLO 与色块混合，搬运前最后对正只使用 YOLO，绕行修正与正式搬运只使用色块，跟随与回库黄线任务跳过物体 YOLO。
- 辅车纯 YOLO 阶段按可调的隔帧间隔触发模型推理，间隔帧沿用上一帧目标结果，避免每帧都发停车控制。
- 在 OpenART 端完成角色内阶段判断。
- 输出辅车视觉速度修正短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 接收辅车 RT1021 下发的本地任务同步帧, body 字段为 `state/target/arg`。
- 使用本地任务同步 topic 的 ACK 帧确认可靠同步包。
- 找物体同步 `arg=1` 稳定满足条件后输出辅车视觉事件回报帧 `event=6`。
- 搬运入口同步 `arg=2` 稳定满足条件后输出辅车视觉事件回报帧 `event=7`。


## 辅车找物体规则

- 找物体模式可通过代码开关选择色块阈值或 YOLO 模型，与主车目标搜索保持一致。
- `assistant/main_v2.py` 在混合识别阶段先基于 YOLO ROI 和标定阈值尝试短期跟踪, 跟踪失手时短时保留预测目标, 连续失败或达到上限后回退到 YOLO；纯 YOLO 阶段不允许回退到色块，纯色块阶段不触发 YOLO。
- 纯 YOLO 阶段的隔帧间隔可分别通过 `MASTER_YOLO_ONLY_INTERVAL_FRAMES` 与 `ASSISTANT_YOLO_ONLY_INTERVAL_FRAMES` 调整。
- 找物体目标点按同步配置编号切换，可通过对应目标点参数调整。
- 找物体同步 `arg=1` 使用寻找阶段目标点，默认 `x=160, y=210`。
- 搬运入口同步 `arg=2` 使用推行前对正目标点，默认 `x=160, y=240`。
- 横向控制目标为目标中心对齐当前目标点。
- 纵向控制目标为目标底边对齐当前目标点。
- 无有效目标时使用配置的搜索速度。
- 目标满足面积下限并连续进入横向、纵向容差窗口后, 按当前同步配置回报 `TARGET_FOUND` 或 `ALIGNED`。
- 纯预测帧只用于补连续控制, 不触发可靠事件。
- 未确认的可靠事件按低频节奏重复发送。

## 主车视觉发送规则

- `master/main.py` 的 `v` 数据流包直接写出, 不执行发送前后延时。
- `master/main.py` 的 `v` 数据流包不等待 task 上下文建立。
- `master/main.py` 的可靠帧在当前入口层直接写出, 不额外插入发送保护延时。
- 未确认的 `r` 事件按低频节奏重复发送, 不随每帧图像重复写出。
- 主通信串口: `UART(2)`。
- RT1021 接收串口: `UART6`。
- 默认波特率: `115200`。

## 角色部署

每个角色目录各自维护独立构建脚本。脚本会读取角色目录外部 `calibration/` 下的共享标定文件，更新本角色源码入口，并上传到板端目录。

```bash
./assistant/build.sh
./assistant/build_v2.sh
./master/build.sh
./master/build_v2.sh
```

默认板端目录为 `/Volumes/NO NAME`。需要指定板端目录时使用 `TARGET_DIR`:

```bash
TARGET_DIR=/path/to/device ./assistant/build.sh
TARGET_DIR=/path/to/device ./assistant/build_v2.sh
TARGET_DIR=/path/to/device ./master/build.sh
TARGET_DIR=/path/to/device ./master/build_v2.sh
```

需要同步 YOLO 权重时传入 `yolo` 参数:

```bash
./assistant/build.sh yolo
./assistant/build_v2.sh yolo
./master/build.sh yolo
./master/build_v2.sh yolo
```

## 物体识别开关

- `master/main.py` 中的 `OBJECT_DETECTION_USE_YOLO`
- `assistant/main.py` 中的 `OBJECT_DETECTION_USE_YOLO`

设为 `False` 时使用当前色块阈值识别，设为 `True` 时使用 `yolo.tflite` 模型识别。

## ChromaForge 色彩标定接入

`calibration/chromaforge_export_adapter.py` 负责把同目录的 `chromaforge-rules.json` 转为 OpenART 入口可用的识别配置。规则文件每次运行都会校验。新格式下, 每个物体会携带自己的识别参数, 包括合并间距、面积下限、最大边长和颜色簇命中要求; 老格式顶层参数仍可作为回退值读取。

手动生成单个入口:

```bash
uv run python calibration/chromaforge_export_adapter.py \
  --source master/main.py \
  --output /tmp/master-main.py \
  --task-constant-name TASKS
```

v2 入口生成时使用 `--task-constant-name OBJECT_TASKS`。

只需要配置片段时省略 `--source` 和 `--output`。

## 验证命令

```bash
uv run --with pytest python -m pytest tests/unit tests/contract -q
```
