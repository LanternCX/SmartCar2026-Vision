# SmartCar2026 - OpenART Vision

这是一个为智能车竞赛配置的 OpenART 视觉项目，运行于 OpenART（MicroPython）环境。

## 项目治理与开发流程

- 统一代理规则入口: `AGENTS.md`
- 项目技能目录: `.agents/skills`
- 开发必须遵循 TDD（RED -> GREEN -> REFACTOR）
- 本仓库保留**单文件架构**: 运行时代码主入口固定为 `main.py`
- 远端规则更新机制文档位于: `docs/problem_statement/`
- 通信协议说明文档位于: `docs/Protocol.md`，用于说明 OpenART 与 RT1021 之间的视觉主链路

## 项目说明

本项目是 2026 智能车蚂蚁搬家组的 OpenART 视觉侧代码，职责收敛为三件事：

1. 图像采集与目标识别
2. 在 ART 端完成跟随阶段判断
3. 生成并持续发送视觉速度修正短包

视觉主链路由 OpenART 直接发送 `v,<vx>,<vy>`。

## 开发环境

开发环境使用 VS Code + MicroPython 方案。

1. **IDE**: Visual Studio Code
2. **插件**: Pylance, Python
3. **智能提示 (Stubs)**: 使用第三方 MicroPython stubs 获得代码补全和类型检查支持

## 当前主线

### 1. 视觉识别

当前主线基于色块识别（Color Blob Detection）完成目标检测与目标选择。

- 当前跟随目标只保留 `red` 色块
- 横向控制输入来自目标中心相对画面中心的横向偏差
- 纵向控制输入来自目标尺度量相对目标尺度量的误差

### 2. 速度短包发送

视觉端在主循环中逐帧发送速度数据流短包，协议格式为：

```text
v,<vx>,<vy>
```

- 发送方向: OpenART -> RT1021
- 主通信串口: `UART(2)`
- RT1021 接收串口: `UART6`
- 默认波特率: `115200`
- 发送方式: 每抓一帧就发送当前结果，不等待历史结果处理完成
- `vx` 表示车体系 x 方向视觉速度修正量
- `vy` 表示车体系 y 方向视觉速度修正量
- `UART6` 视觉链路不发送 `omega`
- OpenART 直接承担跟随阶段判断和控制量生成
- 阶段名固定为 `MARKER_MISSING`、`CENTER_HOLD`、`ALIGN_X`、`ALIGN_Y`、`ALIGN_XY`
- `ALIGN_X` 阶段只输出横向速度修正量，`ALIGN_Y` 阶段只输出纵向速度修正量
- 无目标或保持阶段都发送 `v,0,0`

### 3. 责任边界

- OpenART: 负责看见目标、选中目标、判断当前跟随阶段并生成 `vx` / `vy` 速度修正量
- RT1021: 负责消费并执行 ART 生成的视觉速度修正短包

## 协议入口

协议细节、字段约定和合法载荷说明见 `docs/Protocol.md`。
