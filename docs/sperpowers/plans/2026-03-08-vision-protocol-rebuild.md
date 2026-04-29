# Vision Protocol Rebuild Implementation Plan

## 执行状态

- 状态: Archive


> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OpenArt 端重构为只负责检测、选目标并持续上报 `x,y` 像素观测。

**Architecture:** `main.py` 保持单文件入口,运行时主循环只做图像采集、候选目标提取、最优目标选择与节流发送。RT1021 负责状态机、控制与动作编排,OpenArt 不再下发离散控制命令。

**Tech Stack:** MicroPython/OpenMV (`sensor`, `machine.UART`), Python `pytest`

---

### Task 1: 纯函数测试与实现

**Files:**
- Modify: `main.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

1. 先为视觉帧格式化、候选目标选择、发送节流写失败测试。
2. 运行 `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q` 确认失败原因正确。
3. 在 `main.py` 中写最小实现。
4. 重新运行同一测试并确认通过。

### Task 2: 旧链路移除契约

**Files:**
- Modify: `main.py`
- Test: `tests/contract/test_main_vision_protocol_contract.py`

1. 写失败测试,约束 `main.py` 不再包含旧状态机与同步控制链路。
2. 运行 `python3 -m pytest tests/contract/test_main_vision_protocol_contract.py -q` 确认失败。
3. 重写 `main.py` 为纯视觉上报结构。
4. 重新运行契约测试并确认通过。

### Task 3: 全量回归

**Files:**
- Modify: `main.py`
- Modify: `tests/unit/test_vision_protocol_rebuild.py`
- Modify: `tests/contract/test_main_vision_protocol_contract.py`

1. 运行 `python3 -m pytest tests/unit tests/contract -q`。
2. 若失败,仅修复与纯视觉协议重构相关的问题。
3. 全绿后再做最小清理,保持单文件架构与中文注释要求。
