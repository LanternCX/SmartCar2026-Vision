# SmartCar2026 - OpenART Vision

这是一个为智能车竞赛配置的 OpenART 视觉项目，运行于 OpenART（MicroPython）环境。

## 仓库定位

本仓库作为 `../SmartCar2026-TransportCar` 的附属视觉仓库维护。

本仓库只维护三类内容：

1. OpenART 视觉运行时代码。
2. 视觉协议行为测试与回归测试。
3. 面向本仓库使用者的必要说明。

项目规则、题面材料、协作规格、协作计划和长期开发文档以 `../SmartCar2026-TransportCar` 为准。

## 开发边界

- 运行时代码按角色维护在 `assistant/main.py` 与 `master/main.py`。
- 每个角色目录独立维护面向 OpenART 设备部署的 `build.sh`。
- 不在本仓库维护独立 `.agents/skills` 体系。
- 不在本仓库维护题面规则文档副本。
- 文档、注释和规则入口通过 review 检查，不写硬约束测试。
- 测试只覆盖运行时行为、协议行为和回归场景。

## 项目说明

本项目是 2026 智能车蚂蚁搬家组的 OpenART 视觉侧代码，职责收敛为三件事：

1. 图像采集与目标识别。
2. 在 OpenART 端完成角色内阶段判断。
3. 生成并持续发送视觉速度修正短包。

视觉主链路由 OpenART 直接发送 `v,<vx>,<vy>`。

## 开发环境

开发环境使用 VS Code + MicroPython 方案。

1. **IDE**: Visual Studio Code
2. **插件**: Pylance, Python
3. **智能提示 (Stubs)**: 使用第三方 MicroPython stubs 获得代码补全和类型检查支持

## 当前主线

### 1. 视觉识别

主线基于色块识别（Color Blob Detection）完成目标检测与目标选择。

- `assistant/main.py` 面向辅车跟随主车色标。
- `master/main.py` 面向主车搜索搬运目标。
- 检测目标使用 `red` 色块。
- 横向控制输入来自目标中心相对画面中心的横向偏差。
- 纵向控制输入来自目标尺度量相对目标尺度量的误差。

### 2. 速度短包发送

视觉端在主循环中逐帧发送速度数据流短包，协议格式为：

```text
v,<vx>,<vy>
```

- 发送方向: OpenART -> RT1021。
- 主通信串口: `UART(2)`。
- RT1021 接收串口: `UART6`。
- 默认波特率: `115200`。
- 发送方式: 每抓一帧就发送当前结果，不等待历史结果处理完成。
- `vx` 表示车体系 x 方向视觉速度修正量。
- `vy` 表示车体系 y 方向视觉速度修正量。
- `UART6` 视觉链路不发送 `omega`。
- OpenART 直接承担角色内阶段判断和控制量生成。
- 阶段名固定为 `MARKER_MISSING`、`CENTER_HOLD`、`ALIGN_X`、`ALIGN_Y`、`ALIGN_XY`。
- `ALIGN_X` 阶段只输出横向速度修正量，`ALIGN_Y` 阶段只输出纵向速度修正量。
- 无目标或保持阶段都发送 `v,0,0`。

### 3. 责任边界

- OpenART: 负责看见目标、选中目标、判断角色内阶段并生成 `vx` / `vy` 速度修正量。
- RT1021: 负责消费并执行 OpenART 生成的视觉速度修正短包。

## 角色部署

每个角色目录提供独立构建脚本，脚本会把本目录的 `main.py` 复制为设备根目录的 `main.py`。

```bash
./assistant/build.sh
./master/build.sh
```

默认设备挂载目录为 `/Volumes/NO NAME`。需要指定目标目录时使用 `TARGET_DIR`：

```bash
TARGET_DIR=/path/to/device ./assistant/build.sh
TARGET_DIR=/path/to/device ./master/build.sh
```

## 验证命令

```bash
python3 -m pytest tests/unit tests/contract -q
```
