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

当前视觉链路统一使用固定长度短帧:

- 每帧固定 13 字节, 由 `head/mode/topic/seq/body[8]/crc8` 组成。
- `head=0xA5`, `crc8` 覆盖 `mode/topic/seq/body[8]`, 用于串口字节流重新对齐。
- 本地视觉速度使用 `MODE_UDP + TOPIC_LOCAL_VISION_VELOCITY`。
- 本地视觉同步与事件回报使用 `MODE_TCP`。
- 可靠确认使用 `MODE_ACK`, `topic` 与被确认的可靠帧保持一致。

### OpenART Vision master

`master/main.py` 运行在主车 OpenART 上, 负责物体搜索视觉链路:

- 接收 RT1021 下发的主车视觉同步帧, body 字段为 `context_id/state/target/arg`。
- 使用主车视觉同步 topic 的 ACK 帧确认可靠同步包。
- 基于候选目标识别框中心点计算搜索 P 环。
- 输出主车搜索速度短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 速度短帧独立于 hook 上下文, 每帧直接根据色块识别结果输出。
- 在 hook 条件满足时输出主车视觉事件回报帧, body 字段为 `context_id/event/value`。
- `arg=1` 表示主车物体搜索 hook 配置, 稳定满足条件后回报 `TARGET_FOUND=6`。
- `arg=2` 表示主车搬运入口对正 hook 配置, 稳定满足条件后回报 `ALIGNED=7`。
- `arg=5` 表示主车回库黄线 hook 配置, 后退段回报 `RETURN_LINE_ALIGNED=10`, 平移段在有效跟随区域连续 5 帧算不出黄线时回报 `RETURN_GARAGE_FINISHED=12`。
- 主车搜索目标点按 hook 配置编号切换：`arg=1` 使用寻找阶段目标点，默认 `x=160, y=210`；`arg=2` 使用搬运入口对正目标点，默认 `x=160, y=240`。
- hook 判定使用物体中心相对目标点的横向误差。
- hook 判定使用物体底边相对目标点的纵向误差。
- hook 判定使用候选目标面积。
- 主车搜索控制使用 `v` 数据流, 不依赖周期 `o` 观测包。

### OpenART Vision assistant

`assistant/main.py` 运行在辅车 OpenART 上, 负责辅车跟随主车色标和辅车找目标物体:

- 跟随模式识别主车色标。
- 找物体模式识别红色目标物体。
- 在 OpenART 端完成角色内阶段判断。
- 输出辅车视觉速度修正短帧, body 字段为 `vx/vy/omega/has_omega`, 其中 `omega=0`、`has_omega=0`。
- 接收辅车 RT1021 下发的本地任务同步帧, body 字段为 `state/target/arg`。
- 使用本地任务同步 topic 的 ACK 帧确认可靠同步包。
- 找物体同步 `arg=1` 稳定满足条件后输出辅车视觉事件回报帧 `event=6`。
- 搬运入口同步 `arg=2` 稳定满足条件后输出辅车视觉事件回报帧 `event=7`。


## 辅车找物体规则

- 找物体模式使用红色目标阈值，与主车目标搜索保持一致。
- 找物体目标点按同步配置编号切换，可通过对应目标点参数调整。
- 找物体同步 `arg=1` 使用寻找阶段目标点，默认 `x=160, y=210`。
- 搬运入口同步 `arg=2` 使用推行前对正目标点，默认 `x=160, y=240`。
- 横向控制目标为目标中心对齐当前目标点。
- 纵向控制目标为目标底边对齐当前目标点。
- 无有效目标时使用配置的搜索速度。
- 目标满足面积下限并连续进入横向、纵向容差窗口后, 按当前同步配置回报 `TARGET_FOUND` 或 `ALIGNED`。
- 未确认的可靠事件按低频节奏重复发送。

## 主车视觉发送规则

- `master/main.py` 的 `v` 数据流包直接写出, 不执行发送前后延时。
- `master/main.py` 的 `v` 数据流包不等待 hook 上下文建立。
- `master/main.py` 的可靠帧在当前入口层直接写出, 不额外插入发送保护延时。
- 未确认的 `r` 事件按低频节奏重复发送, 不随每帧图像重复写出。
- 主通信串口: `UART(2)`。
- RT1021 接收串口: `UART6`。
- 默认波特率: `115200`。

## 角色部署

根目录构建脚本会读取 `tools/chromaforge-rules.json`, 校验后生成主车和辅车两个角色入口。

```bash
bash build.sh
```

默认输出目录为 `build/`。需要指定输出目录时使用 `BUILD_DIR`:

```bash
BUILD_DIR=/path/to/build bash build.sh
```

每个角色目录也提供独立部署脚本, 脚本会把本目录的 `main.py` 复制为设备根目录的 `main.py`。

```bash
./assistant/build.sh
./master/build.sh
```

默认设备挂载目录为 `/Volumes/NO NAME`。需要指定目标目录时使用 `TARGET_DIR`:

```bash
TARGET_DIR=/path/to/device ./assistant/build.sh
TARGET_DIR=/path/to/device ./master/build.sh
```

## ChromaForge 色彩标定接入

`tools/chromaforge_export_adapter.py` 负责把同目录的 `chromaforge-rules.json` 转为 OpenART 入口可用的识别配置。规则文件每次运行都会校验。

手动生成单个入口:

```bash
uv run python tools/chromaforge_export_adapter.py \
  --source master/main.py \
  --output /tmp/master-main.py \
  --task-constant-name TASKS
```

只需要配置片段时省略 `--source` 和 `--output`。

## 验证命令

```bash
uv run --with pytest python -m pytest tests/unit tests/contract -q
```
