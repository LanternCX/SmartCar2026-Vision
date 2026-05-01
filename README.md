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

- 运行时代码按角色维护在 `assistant/main.py` 与 `master/main.py`。
- 每个角色目录独立维护面向 OpenART 设备部署的 `build.sh`。
- 本仓库不维护独立 `.agents/skills` 体系。
- 本仓库不维护题面规则文档副本。
- 文档、注释和规则入口通过 review 检查。
- 测试覆盖运行时行为、协议行为和回归场景。

## 角色职责

### OpenART Vision master

`master/main.py` 运行在主车 OpenART 上, 负责物体搜索视觉链路:

- 接收 RT1021 下发的 hook 上下文 `s,<reliable_seq>,<context_id>,<state>,<target>,<arg>`。
- 使用 `a,<reliable_seq>` 确认可靠同步包。
- 持续输出物体观测 `o,<context_id>,<x>,<y>,<value>`。
- 在 hook 条件满足时输出可靠事件 `r,<reliable_seq>,<context_id>,6,<value>`。
- `arg=1` 表示主车物体搜索 hook 配置。
- `x` 使用物体中心相对画面中线的横向误差。
- `y` 使用物体中心相对画面下三分之二点的纵向误差。
- `value` 使用候选目标面积。

### OpenART Vision assistant

`assistant/main.py` 运行在辅车 OpenART 上, 负责辅车跟随主车色标:

- 识别主车色标。
- 在 OpenART 端完成角色内阶段判断。
- 输出辅车视觉速度修正数据流 `v,<vx>,<vy>`。
- `v` 数据流包不携带 `omega` 和阶段元信息。

## 主车视觉发送规则

- `master/main.py` 的 `o` 数据流包直接写出, 不执行发送前后延时。
- `master/main.py` 的 `s/a/r` 可靠包按可靠发送规则写出, 发送前后各执行一次 1 ms 延时。
- 未确认的 `r` 事件按低频节奏重复发送, 不随每帧图像重复写出。
- 主通信串口: `UART(2)`。
- RT1021 接收串口: `UART6`。
- 默认波特率: `115200`。

## 角色部署

每个角色目录提供独立构建脚本, 脚本会把本目录的 `main.py` 复制为设备根目录的 `main.py`。

```bash
./assistant/build.sh
./master/build.sh
```

默认设备挂载目录为 `/Volumes/NO NAME`。需要指定目标目录时使用 `TARGET_DIR`:

```bash
TARGET_DIR=/path/to/device ./assistant/build.sh
TARGET_DIR=/path/to/device ./master/build.sh
```

## 验证命令

```bash
python3 -m pytest tests/unit tests/contract -q
```
