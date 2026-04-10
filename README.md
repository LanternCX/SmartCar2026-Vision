# SmartCar2025 - OpenART Vision

这是一个为智能车竞赛配置的 OpenART 视觉项目，运行于 OpenART（MicroPython）环境。

## 项目治理与开发流程

- 统一代理规则入口: `AGENTS.md`
- 项目技能目录: `.agents/skills`
- 开发必须遵循 TDD（RED -> GREEN -> REFACTOR）
- 本仓库保留**单文件架构**: 运行时代码主入口固定为 `main.py`
- 远端规则更新机制文档位于: `docs/problem_statement/`
- 通信协议说明文档位于: `docs/Protocol.md`，用于说明 OpenArt 与 RT1021 之间的当前视觉主链路

## 项目说明

本项目是 2025 智能车蚂蚁搬家组的 OpenART 视觉侧代码，职责收敛为三件事：

1. 图像采集与目标识别
2. 在 ART 端完成跟随阶段判断
3. 直接生成并持续发送辅车已支持的 follow 请求

也就是说，当前主链路口径就是由 OpenArt 直接发送辅车 follow 请求。

视觉端不额外扩第二套视觉协议，而是直接对齐辅车当前 follow 请求格式。

## 开发环境

开发环境延续 VS Code + MicroPython 方案。

1. **IDE**: Visual Studio Code
2. **插件**: Pylance, Python
3. **智能提示 (Stubs)**: 使用第三方 MicroPython stubs 获得代码补全和类型检查支持

## 当前主线

### 1. 视觉识别

当前主线基于色块识别（Color Blob Detection）完成目标检测与目标选择。

- 当前跟随目标只保留 `red` 色块
- `x` 继续按目标中心相对画面中心的横向偏差生成
- `y` 改为按目标面积相对目标面积的误差生成

### 2. 速度模式请求发送

视觉端在主循环中持续发送速度模式请求，当前主线协议格式为：

```text
f=1,m=1,x=<x>,y=<y>
```

- 发送方向: OpenArt -> RT1021
- 主通信串口: `UART(2)`
- 默认波特率: `115200`
- 发送方式: 随视觉循环持续发送最新结果，不等待历史结果处理完成
- `f=1,m=1` 表示这是一条速度模式入口请求
- `x` 表示发给辅车的横向速度
- `y` 表示发给辅车的纵向速度
- OpenArt 直接承担跟随阶段判断和控制量生成
- 当前阶段名固定为 `MARKER_MISSING`、`CENTER_HOLD`、`ALIGN_X`、`ALIGN_Y`
- `ALIGN_X` 阶段只输出横向控制量, `ALIGN_Y` 阶段只输出纵向控制量
- 当前主线直接对齐辅车已经支持的速度模式入口

### 3. 责任边界

- OpenArt: 负责看见目标、选中目标、判断当前跟随阶段并生成 follow 控制量
- RT1021: 负责直接消费并执行 ART 已经生成好的 follow 请求

## 协议入口

协议细节、字段约定和合法载荷说明见 `docs/Protocol.md`。
