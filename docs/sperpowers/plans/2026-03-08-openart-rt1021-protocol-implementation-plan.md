# OpenArt 与 RT1021 协议文档 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重写 `Protocol.md`,将当前 RT1021 遥控协议与 OpenArt/RT1021 视觉通信协议拆分清楚,供 vision 端重构直接参考。

**Architecture:** 以 `SmartCar2026-Vision/main.py` 与 `../SmartCar2026-TransportCar` 的现有实现为事实来源,先整理共享串口约定,再分别编写遥控协议与视觉专用协议,最后补充兼容与迁移说明。文档只记录已落地行为,不预埋未实现字段。

**Tech Stack:** Markdown, OpenArt MicroPython, RT1021 MicroPython, UART 文本协议

---

### Task 1: 盘点当前协议事实来源

**Files:**
- Modify: `Protocol.md`
- Reference: `main.py`
- Reference: `../SmartCar2026-TransportCar/src/hardware/uart_bus.py`
- Reference: `../SmartCar2026-TransportCar/src/services/transport_car.py`
- Reference: `../SmartCar2026-TransportCar/src/services/vision_protocol.py`
- Reference: `../SmartCar2026-TransportCar/src/services/command_router.py`

**Step 1: 确认共享传输约定**

- 记录 OpenArt 与 RT1021 当前使用的串口角色、波特率与行分帧约定。

**Step 2: 确认遥控协议事实**

- 逐项核对 `vx/vy/omega/x/y/dx/dy/angle/d_angle/rear/reset/print` 的输入格式和锁语义。

**Step 3: 确认视觉协议事实**

- 核对视觉协议的判定条件: 来源必须为 `UART6`,并且消息只包含 `x` 和 `y` 两个键。

**Step 4: 确认查询回包规则**

- 核对 `?pos`、`?lock` 以及 `?vision` 等诊断查询的返回格式与回写串口规则。

**Step 5: 如用户要求再提交 commit**

- 当前阶段只落文档,默认不创建 git commit。

### Task 2: 重写共享约定与遥控协议模块

**Files:**
- Modify: `Protocol.md`

**Step 1: 写“共享传输约定”**

- 说明 ASCII 文本、`\r\n` 行分帧、查询格式 `?token`、回包同串口返回。

**Step 2: 写“车模控制协议”命令表**

- 将控制命令按速度、位置、模式、系统四类整理成一张表。

**Step 3: 写“车模控制协议”查询表**

- 区分简洁查询返回（如 `?pos`、`?lock`）和结构化诊断查询返回（如 `?vision`）。

**Step 4: 写典型交互示例**

- 提供常见控制命令和查询示例,便于人工调试与脚本对接。

**Step 5: 如用户要求再提交 commit**

- 当前阶段只落文档,默认不创建 git commit。

### Task 3: 重写视觉通信模块与兼容说明

**Files:**
- Modify: `Protocol.md`

**Step 1: 写视觉协议边界**

- 明确 `x,y` 为图像坐标观测,不是世界坐标命令。

**Step 2: 写视觉消费规则**

- 说明仅缓存最新帧、超时丢失、无逐帧 ACK、锁定时会清空视觉状态。

**Step 3: 写兼容与迁移说明**

- 说明旧 OpenArt 仍可走遥控协议,但新的 vision 重构目标应收敛为“持续发 `x,y` + 按需查询”。

**Step 4: 运行仓库验证命令**

Run: `python3 -m pytest tests/unit tests/contract -q`
Expected: 测试继续通过,因为本次仅修改文档。

**Step 5: 如用户要求再提交 commit**

- 当前阶段只落文档,默认不创建 git commit。
