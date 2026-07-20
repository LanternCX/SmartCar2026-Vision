# SmartCar2026 - OpenART Vision

这是智能车竞赛 OpenART 视觉项目, 运行于 OpenART MicroPython 环境。

## 仓库定位

本仓库作为 `../SmartCar2026-TransportCar` 的附属视觉仓库维护。

本仓库维护三类内容:

1. OpenART 视觉运行时代码。
2. 视觉协议行为测试与回归测试。
3. 面向本仓库使用者的必要说明。

项目规则、题面材料、协作规格、协作计划、长期开发文档和长期协作记忆以 `../SmartCar2026-TransportCar` 为准。

## 开发边界

- 运行时代码按角色维护在 `assistant/` 与 `master/` 目录。
- 每个角色目录独立维护面向 OpenART 设备部署的构建脚本。
- 本仓库不维护独立 `.agents/skills` 体系。
- 本仓库不维护题面规则文档副本。
- 文档、注释和规则入口通过 review 检查。
- 测试覆盖运行时行为、协议行为和回归场景。

## 入口接口风格

- `master/run.py` 与 `assistant/run.py` 的接口风格保持一致; 各角色 `main.py` 只负责导入 `run` 模块并启动正式运行函数。
- 当前轮次真正处理的对象应显式传入接口, 例如 `img`。
- 由当前处理对象可直接得到的派生信息, 例如图像宽高, 不作为参数层层下传。
- 当前运行态只维护一份的共享状态, 统一放在入口级 `state` 对象中读取。

## 角色职责

当前视觉链路统一使用固定长度短帧:

- 每帧固定 15 字节, 由 `head/mode/topic/seq/body[10]/crc8` 组成。
- `head=0xA5`, `crc8` 覆盖 `mode/topic/seq/body[10]`, 用于串口字节流重新对齐。
- 本地视觉速度使用 `MODE_UDP + TOPIC_LOCAL_VISION_VELOCITY`。
- 本地视觉同步与事件回报使用 `MODE_TCP`。
- 可靠确认使用 `MODE_ACK`, `topic` 与被确认的可靠帧保持一致。

### OpenART Vision master

`master/run.py` 运行在主车 OpenART 上, 负责物体搜索视觉链路:

- 接收 RT1021 下发的主车视觉同步帧, body 字段为 `context_id/state/target/arg`。
- 使用主车视觉同步 topic 的 ACK 帧确认可靠同步包。
- 主车物体阶段由本地开关选择纯 YOLO 或纯色块模式, 每张图像只运行当前模式对应的检测器。
- 主车搜索候选类别连续两帧保持一致后才锁定, 类别变化或空帧会重新计数。
- YOLO 通过最近一次目标位置持续关联棕熊与白熊; 白熊可暂时使用棕熊候选, 同一目标收敛为白熊后只升级、不降级。
- 基于候选目标识别框中心点计算搜索 P 环。
- 输出主车搜索速度短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 主车绕行修正阶段输出独立 `vx/vy` 平移修正, 底盘侧负责绕行动作解算。
- 主车寻找阶段无有效目标时使用配置的搜索速度, 搬运前对正阶段无有效目标时输出零速度。
- 速度短帧独立于 task 上下文, 每帧直接根据当前识别开关选择的结果输出。
- 在 task 条件满足时输出主车视觉事件回报帧, body 字段为 `context_id/event/value`。
- `arg` 低字节为主车视觉 task 配置编号, `0x0100` 标记最后一次搬运, `0x0200` 标记第一次搬运。
- 配置编号 `1` 表示主车物体搜索 task, 稳定满足条件后回报 `TARGET_FOUND=6`。
- 配置编号 `2` 表示主车搬运入口对正 task, 稳定满足条件后回报 `ALIGNED=7`。
- 主车搜索目标点按 task 配置编号切换：配置编号 `1` 使用寻找阶段目标点，默认 `x=160, y=210`；配置编号 `2` 使用搬运入口对正目标点，默认 `x=160, y=240`。
- task 判定使用物体中心相对目标点的横向误差。
- task 判定使用物体底边相对目标点的纵向误差。
- task 判定使用候选目标面积。
- 主车搜索控制使用 `v` 数据流, 不依赖周期 `o` 观测包。

### OpenART Vision assistant

`assistant/run.py` 运行在辅车 OpenART 上, 负责辅车跟随主车色标和辅车找目标物体:

- 跟随模式识别主车色标。
- 找物体模式可通过代码开关选择色块阈值或 YOLO 模型识别目标物体。
- 辅车物体阶段由本地开关选择纯 YOLO 或纯色块模式, 每张图像只运行当前模式对应的检测器。
- 色标跟随使用独立的色块算法, 不进入物体识别模式路由。
- 在 OpenART 端完成角色内阶段判断。
- 输出辅车视觉速度修正短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 辅车绕行修正阶段输出独立 `vx/vy` 平移修正, 底盘侧负责绕行动作解算。
- 接收辅车 RT1021 下发的本地任务同步帧, body 字段为 `state/target/arg`。
- 使用本地任务同步 topic 的 ACK 帧确认可靠同步包。
- 找物体同步 `arg=1` 稳定满足条件后输出辅车视觉事件回报帧 `event=6`。
- 搬运入口同步 `arg=2` 稳定满足条件后输出辅车视觉事件回报帧 `event=7`。


## 辅车找物体规则

- 找物体模式可通过代码开关选择色块阈值或 YOLO 模型，与主车目标搜索保持一致。
- YOLO 模式逐图执行模型, 色块模式逐图使用本地标定阈值。
- YOLO 通过最近一次目标位置持续关联棕熊与白熊; 白熊可暂时使用棕熊候选, 同一目标收敛为白熊后只升级、不降级。
- 单帧无有效候选时按目标丢失处理, 不复用上一张图像的候选或检测框。
- 找物体目标点按同步配置编号切换，可通过对应目标点参数调整。
- 找物体同步 `arg=1` 使用寻找阶段目标点，默认 `x=160, y=210`。
- 搬运入口同步 `arg=2` 使用推行前对正目标点，默认 `x=160, y=240`。
- 横向控制目标为目标中心对齐当前目标点。
- 纵向控制目标为目标底边对齐当前目标点。
- 寻找阶段无有效目标时使用配置的搜索速度, 搬运前对正阶段无有效目标时输出零速度。
- 目标满足面积下限并连续进入横向、纵向容差窗口后, 按当前同步配置回报 `TARGET_FOUND` 或 `ALIGNED`。
- 未确认的可靠事件按低频节奏重复发送。

## 主车视觉发送规则

- `master/run.py` 的 `v` 数据流包直接写出, 不执行发送前后延时。
- `master/run.py` 的 `v` 数据流包不等待 task 上下文建立。
- `master/run.py` 的可靠帧在当前入口层直接写出, 不额外插入发送保护延时。
- 未确认的 `r` 事件按低频节奏重复发送, 不随每帧图像重复写出。
- 主通信串口: `UART(12)`。
- RT1021 接收串口: `UART6`。
- 默认波特率: `115200`。

## 上电状态指示

- 白灯在视觉启动和正常运行期间保持关闭。
- RGB 绿灯常亮表示初始化和首帧准备; YOLO 模型在相机初始化前加载且不执行上电推理, 色块模式在相机初始化后执行检测预热。每次成功写出启动 READY 后翻转一次。
- 本地启动握手完成后蓝灯慢闪, 表示正在等待首个正式任务; 正式逐帧运行时每帧翻转一次。
- 启动或运行发生致命错误时红灯常亮, 白灯关闭。
- 致命错误的完整异常栈覆盖写入 `/sd/vision_error.log`, 保留最近一次错误。
- 调试模式完成首帧准备后直接进入逐帧蓝灯指示, 不执行启动握手。

## 调试模式

- 调试模式上电后直接持续显示物体识别结果, 不初始化或处理通信。
- 识别来源由 `OBJECT_DETECTION_USE_YOLO` 决定, YOLO 与色块模式使用同一套候选绘制行为。
- 调试画面显示候选框、选中目标和目标点, 不发送速度、事件或 ACK。

## 角色部署

每个角色目录各自维护独立构建脚本。脚本会把本角色的 `main.py` 与 `run.py` 上传到板端目录。

```bash
./assistant/build.sh
./master/build.sh
```

默认板端目录为 `/Volumes/NO NAME`。需要指定板端目录时使用 `TARGET_DIR`:

```bash
TARGET_DIR=/path/to/device ./assistant/build.sh
TARGET_DIR=/path/to/device ./master/build.sh
```

需要同步 YOLO 权重时传入 `yolo` 参数:

```bash
./assistant/build.sh yolo
./master/build.sh yolo
```

## 物体识别开关

- `master/run.py` 中的 `OBJECT_DETECTION_USE_YOLO`
- `assistant/run.py` 中的 `OBJECT_DETECTION_USE_YOLO`

设为 `False` 时使用当前色块阈值识别，设为 `True` 时使用 `yolo.tflite` 模型识别。

主车决赛目标选择策略由 `master/run.py` 中的 `FINAL_ROUND_SELECTION_MODE` 选择：

- `_ObjectSelectionMode.CENTER`：选择离搜索目标点最近的候选。
- `_ObjectSelectionMode.EDGE`：按候选所在外侧和上一次目标边分组选择。
- `_ObjectSelectionMode.NEAREST_BOTTOM`：选择离图像底边最近的候选。

默认使用 `_ObjectSelectionMode.EDGE`。

主车决赛红色目标策略由 `master/run.py` 中的 `RED_SELECTION_MODE` 选择：

- `_RedSelectionMode.ALL`：每次搜索都允许红色参与决赛目标选择。
- `_RedSelectionMode.LAST`：非最后一次排除红色, 最后一次允许所有颜色参与中心距离选择。
- `_RedSelectionMode.NEVER`：每次搜索都排除红色。
- `_RedSelectionMode.FIRST`：第一次只保留红色并使用中心距离选择, 后续排除红色。

默认使用 `_RedSelectionMode.FIRST`。第一次和最后一次由 RT1021 根据已完成数量和配置总数标记, 识别侧不写死搬运次数。

## ChromaForge 色彩标定接入

`calibration/chromaforge_export_adapter.py` 负责把同目录的 `chromaforge-rules.json` 转为 OpenART 入口可用的识别配置。规则文件每次运行都会校验。新格式下, 每个物体会携带自己的识别参数, 包括合并间距、面积下限、最大边长和颜色簇命中要求; 老格式顶层参数仍可作为回退值读取。

手动生成单个入口:

```bash
uv run python calibration/chromaforge_export_adapter.py \
  --source master/run.py \
  --output /tmp/master-run.py \
  --task-constant-name OBJECT_TASKS
```

只需要配置片段时省略 `--source` 和 `--output`。

## 验证命令

```bash
PYTHONPATH=. uv run --with pytest-xdist pytest -n auto tests/unit tests/contract -q
```
