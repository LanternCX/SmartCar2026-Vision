# 视觉直接回传实现计划

## 执行状态

- 状态: Archive


> **供代理执行时使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务执行本计划，步骤统一使用复选框 `- [ ]` 跟踪。

**Goal:** 先锁死“持续直接回传最新目标框，且仅允许启动阶段可选一次 `reset=1`”这一实现边界，再按需修正文档、测试与运行时代码；当前执行结果可能无需修改 `main.py`。

**Architecture:** 保持单文件 `main.py`，先用主机侧测试与文档检查锁死协议边界，再按实际差异做最小修改。运行时主线始终是持续上报观测；唯一允许的非观测动作仅为启动阶段可选一次 `reset=1`。

**Tech Stack:** Python 3、pytest、OpenART MicroPython 单文件运行时

---

### Task 1: 锁死契约测试里的禁止边界

**Files:**
- Modify: `tests/contract/test_main_vision_protocol_contract.py`
- Verify: `main.py`

- [ ] **Step 1: 写失败测试，禁止旧轮询、旧查询、旧控制关键字**

```python
def test_main_rejects_legacy_query_and_control_tokens() -> None:
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden = [
        "send_cmd_sync(",
        "wait_until_idle(",
        "?lock",
        "?pos",
        "STATE?",
        "PING",
    ]
    for token in forbidden:
        assert token not in text
```

- [ ] **Step 2: 写失败测试，禁止旧状态机和旧控制字段**

```python
def test_main_rejects_legacy_state_machine_tokens() -> None:
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden = ["ALIGN_ANGLE", "ALIGN_DIST", "ALIGN_DX", '"dx=', '"dy=']
    for token in forbidden:
        assert token not in text
```

- [ ] **Step 3: 写失败测试，禁止查询函数和同步等待函数继续存在**

```python
def test_main_does_not_define_query_or_sync_helpers() -> None:
    text = MAIN_PATH.read_text(encoding="utf-8")
    forbidden_defs = ["def wait_until_idle", "def query_", "def send_cmd_sync"]
    for token in forbidden_defs:
        assert token not in text
```

- [ ] **Step 4: 运行契约测试，确认新增断言先失败**

Run: `python3 -m pytest tests/contract/test_main_vision_protocol_contract.py -q`
Expected: FAIL，且失败原因对应旧边界尚未完全锁死

### Task 2: 锁死单元测试里的协议输出边界

**Files:**
- Modify: `tests/unit/test_vision_protocol_rebuild.py`
- Verify: `main.py`

- [ ] **Step 1: 写失败测试，确认观测格式只输出完整识别框**

```python
def test_format_vision_frame_outputs_only_bbox_fields() -> None:
    (format_vision_frame,) = load_functions("format_vision_frame")
    frame = format_vision_frame(1, 2, 3, 4)
    assert frame == "left=1,top=2,right=3,bottom=4"
```

- [ ] **Step 2: 写失败测试，确认坐标归一化在视觉端内部完成**

```python
def test_protocol_frame_uses_normalized_bbox_coordinates() -> None:
    ...
    assert frame == "left=100,top=150,right=140,bottom=220"
```

- [ ] **Step 3: 写失败测试，确认发送节流边界不早发不漏发**

```python
def test_should_send_throttle_boundary() -> None:
    (should_send,) = load_functions("should_send")
    assert should_send(now_ms=130, last_send_ms=100, interval_ms=30) is True
    assert should_send(now_ms=129, last_send_ms=100, interval_ms=30) is False
```

- [ ] **Step 4: 运行单元测试，确认新增断言先失败**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: FAIL，且失败原因对应当前边界尚未完全满足

### Task 3: 锁死单元测试里的运行时行为边界

**Files:**
- Modify: `tests/unit/test_vision_protocol_rebuild.py`
- Verify: `main.py`

- [ ] **Step 1: 写失败测试，确认启动复位只允许一条 `reset=1`**

```python
def test_send_startup_reset_emits_exactly_one_reset_line() -> None:
    ...
    send_startup_reset(namespace_uart)
    assert namespace_uart.writes == ["reset=1\r\n"]
```

- [ ] **Step 2: 写失败测试，确认目标选择分值更小者优先**

```python
def test_choose_best_candidate_prefers_smaller_score() -> None:
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [("Red", 40, 0, 200), ("Green", 158, 0, 230)]
    assert choose_best_candidate(candidates, 160, 240) == ("Green", 158, 0, 230)
```

- [ ] **Step 3: 写失败测试，确认同分时取原顺序第一个候选**

```python
def test_choose_best_candidate_keeps_first_candidate_when_scores_tie() -> None:
    (choose_best_candidate,) = load_functions("choose_best_candidate")
    candidates = [("Red", 150, 0, 230), ("Green", 170, 0, 230)]
    assert choose_best_candidate(candidates, 160, 240) == ("Red", 150, 0, 230)
```

- [ ] **Step 4: 写失败测试，确认无目标时保持静默**

```python
def test_run_stays_silent_when_no_candidates() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "if not candidates:\n            continue" in source
```

- [ ] **Step 5: 运行单元测试，确认新增断言先失败**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: FAIL，且失败原因对应选择规则或启动边界尚未完全满足

### Task 4: 按需修正纯函数边界

**Files:**
- Modify if needed: `main.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

- [ ] **Step 1: 修正 `format_vision_frame()`，只输出完整识别框字段**

```python
def format_vision_frame(left, top, right, bottom):
    return "left=%s,top=%s,right=%s,bottom=%s" % (...)
```

- [ ] **Step 2: 修正 `normalize_bbox_for_protocol()`，保证视觉端内部完成归一化**

```python
def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    normalized_top = img_height - bottom
    normalized_bottom = img_height - top
    return left, normalized_top, right, normalized_bottom
```

- [ ] **Step 3: 修正 `should_send()`，满足节流边界**

```python
def should_send(now_ms, last_send_ms, interval_ms):
    ...
    return elapsed_ms >= interval_ms
```

- [ ] **Step 4: 运行单元测试，确认纯函数相关测试转绿**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: PASS 或只剩运行循环相关失败

### Task 5: 按需修正候选选择边界

**Files:**
- Modify if needed: `main.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

- [ ] **Step 1: 修正 `build_blob_candidates()`，确保排序使用归一化后的 `bottom`**

```python
_, _, _, bottom = normalize_bbox_for_protocol(left, top, right, bottom, img_height)
candidates.append((task_name, blob.cx(), blob.cy(), bottom, blob))
```

- [ ] **Step 2: 修正 `choose_best_candidate()`，写死分值规则**

```python
def choose_best_candidate(candidates, cx_screen, img_height):
    return min(
        candidates,
        key=lambda item: (item[1] - cx_screen) ** 2 + (img_height - item[3]) ** 2,
    )
```

- [ ] **Step 3: 运行单元测试，确认候选选择测试转绿**

Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`
Expected: PASS 或只剩主循环/契约相关失败

### Task 6: 按需修正主循环边界

**Files:**
- Modify if needed: `main.py`
- Test: `tests/contract/test_main_vision_protocol_contract.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

- [ ] **Step 1: 删除 `main.py` 中残留的同步等待和查询 helper 定义**

```python
# 不保留任何 wait_until_idle / query_* / send_cmd_sync helper
```

- [ ] **Step 2: 让 `run()` 只保留启动可选一次复位和持续观测上报**

```python
if STARTUP_RESET:
    send_startup_reset(uart)

while True:
    img = sensor.snapshot()
    candidates = build_blob_candidates(img)
    if not candidates:
        continue
    ...
```

- [ ] **Step 3: 确认无目标时不发送任何伪帧或占位帧**

```python
if not candidates:
    continue
```

- [ ] **Step 4: 运行契约和单元测试，确认代码边界转绿**

Run: `python3 -m pytest tests/contract/test_main_vision_protocol_contract.py tests/unit/test_vision_protocol_rebuild.py -q`
Expected: PASS

### Task 7: 收口仓库文档口径

**Files:**
- Modify: `docs/Protocol.md`
- Reference: `docs/plans/2026-04-02-vision-direct-report-design.md`

- [ ] **Step 1: 更新 `docs/Protocol.md`，明确当前主机制是持续直接回传，且唯一允许的非观测动作是启动阶段可选一次 `reset=1`**

```md
- OpenART 持续直接回传观测
- 启动阶段可选一次 reset=1
- 无目标时由 RT1021 自行按超时判定丢失
```

- [ ] **Step 2: 删除或改写所有“兼容旧链路”“轮询查询”“视觉侧控制端”表述**

```md
- 不保留历史兼容说明作为当前主线
```

- [ ] **Step 3: 用文本检查和人工复查确认文档口径收口准确**

Run: 使用文本搜索检查 `docs/Protocol.md` 中是否仍保留旧轮询、旧查询、旧控制主线表述
Expected: 文本检查无旧主线残留，且人工复查确认“持续直接回传”为主线、“启动阶段可选一次 reset=1”为唯一非观测例外

### Task 8: 做最终验证

**Files:**
- Verify: `main.py`
- Verify: `tests/unit/test_vision_protocol_rebuild.py`
- Verify: `tests/contract/test_main_vision_protocol_contract.py`
- Verify: `docs/Protocol.md`

- [ ] **Step 1: 运行单元测试**

Run: `python3 -m pytest tests/unit -q`
Expected: PASS

- [ ] **Step 2: 运行契约测试**

Run: `python3 -m pytest tests/contract -q`
Expected: PASS

- [ ] **Step 3: 运行主机侧全量测试**

Run: `python3 -m pytest tests/unit tests/contract -q`
Expected: PASS

- [ ] **Step 4: 检查工作区状态**

Run: `git status --short`
Expected: 只包含本轮预期改动

- [ ] **Step 5: 等待用户确认是否需要提交实现改动**

```bash
# 不自动提交实现改动，先向用户汇报验证结果
```
