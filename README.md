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

本项目是 2025 智能车蚂蚁搬家组的 OpenART 视觉侧代码，职责收敛为两件事：

1. 图像采集与目标识别
2. 将最新识别结果持续直接回传给 RT1021

视觉端不再承担任务状态机、底盘控制决策或动作编排；这些都由 RT1021 侧负责。

## 开发环境

开发环境延续 VS Code + MicroPython 方案。

1. **IDE**: Visual Studio Code
2. **插件**: Pylance, Python
3. **智能提示 (Stubs)**: 使用第三方 MicroPython stubs 获得代码补全和类型检查支持

## 当前主线

### 1. 视觉识别

当前主线基于色块识别（Color Blob Detection）完成目标检测与目标选择。

### 2. 持续直接回传

视觉端在主循环中持续发送最新目标框，当前主线协议格式为：

```text
left=<num>,top=<num>,right=<num>,bottom=<num>
```

- 发送方向: OpenArt -> RT1021
- 主通信串口: `UART(2)`
- 默认波特率: `115200`
- 发送方式: 随视觉循环持续发送最新结果，不等待历史结果处理完成

### 3. 责任边界

- OpenArt: 负责看见目标、选中目标、持续上报最新框
- RT1021: 负责消费视觉结果，并完成后续控制与动作组织

## 协议入口

协议细节、字段约定和合法载荷说明见 `docs/Protocol.md`。
