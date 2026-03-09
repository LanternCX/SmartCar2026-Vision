# Vision BBox Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 OpenArt 端从发送 `x,y` 单点观测切换为发送 `left,top,right,bottom` 完整识别框。

**Architecture:** 保持 `main.py` 单文件结构和现有目标选择流程不变，只把串口上报从中心点改为基于 `blob.rect()` 派生的完整框。测试先失败再实现，确保视觉端协议与主控端新协议一致。

**Tech Stack:** Python 3.10+, pytest, OpenART MicroPython single-file runtime

---

### Task 1: 写入设计与计划文档

**Files:**
- Create: `docs/plans/2026-03-09-vision-bbox-reporting-design.md`
- Create: `docs/plans/2026-03-09-vision-bbox-reporting.md`

**Step 1: 写入文档**

- 记录完整框协议切换和最小改动策略

**Step 2: 校验文档存在**

Run: `python3 - <<'PY'
from pathlib import Path
for path in [
    Path('docs/plans/2026-03-09-vision-bbox-reporting-design.md'),
    Path('docs/plans/2026-03-09-vision-bbox-reporting.md'),
]:
    assert path.is_file(), path
print('ok')
PY`
Expected: `ok`

### Task 2: 先写完整框失败测试

**Files:**
- Modify: `tests/unit/test_vision_protocol_rebuild.py`
- Modify: `tests/contract/test_main_vision_protocol_contract.py`

**Step 1: Write the failing test**

```python
def test_format_vision_frame_encodes_bbox() -> None:
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(100, 20, 140, 90)
    assert frame == "left=100,top=20,right=140,bottom=90"
```

```python
def test_main_formats_bbox_observation_frames() -> None:
    text = MAIN_PATH.read_text(encoding="utf-8")
    assert 'left=%s,top=%s,right=%s,bottom=%s' in text
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py tests/contract/test_main_vision_protocol_contract.py -q`
Expected: FAIL because `main.py` still formats `x,y`

**Step 3: Write minimal implementation**

- 更新格式化函数签名与 contract 断言

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py tests/contract/test_main_vision_protocol_contract.py -q`
Expected: PASS

### Task 3: 接入运行时发送路径

**Files:**
- Modify: `main.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

**Step 1: Write the failing test**

```python
def test_run_uses_blob_rect_for_uart_frame() -> None:
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: FAIL because `run()` still sends center-point `x,y`

**Step 3: Write minimal implementation**

- 从 `best_blob.rect()` 提取 `left,top,width,height`
- 转成 `left,top,right,bottom` 后发送

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: PASS

### Task 4: 主机侧回归验证

**Files:**
- Modify: `README.md`（如示例仍残留旧 `x,y`）

**Step 1: Run targeted tests**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py tests/contract/test_main_vision_protocol_contract.py -q`
Expected: PASS

**Step 2: Run full host tests**

Run: `python3 -m pytest tests/unit tests/contract -q`
Expected: PASS
