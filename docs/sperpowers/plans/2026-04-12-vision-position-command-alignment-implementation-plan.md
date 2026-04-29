# OpenART Position Command Alignment Implementation Plan

## 执行状态

- 状态: Archive


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `main.py` 的逐帧串口输出从旧速度请求收口为 `dx=<dx>,dy=<dy>,lock=0` 位置式命令,并同步测试与文档口径。

**Architecture:** 保留 `main.py` 单文件结构、现有目标检测和四段式阶段判断,只替换输出格式、控制量语义和逐帧发送逻辑。先改测试锁定新边界,再改运行时代码,最后统一收口 `README.md` 与 `docs/Protocol.md`。

**Tech Stack:** Python, MicroPython, OpenART, pytest

---

## 文件与职责

- `main.py`: 视觉运行时唯一入口,负责候选提取、阶段判断、位置量计算与逐帧串口输出。
- `tests/unit/test_vision_protocol_rebuild.py`: 约束纯函数和运行时参数语义,锁定位置式输出行为。
- `tests/contract/test_main_vision_protocol_contract.py`: 约束 `main.py`、`README.md` 与 `docs/Protocol.md` 的正式协议口径。
- `README.md`: 仓库主入口说明,描述当前视觉主线职责与正式输出格式。
- `docs/Protocol.md`: 视觉与底盘之间的正式文本协议说明。
- `docs/sperpowers/specs/2026-04-12-vision-position-command-alignment-spec.md`: 当前实现计划对应的已批准规格,执行时逐项对照。

> 说明: 本计划不包含 commit 步骤。若执行阶段需要提交,先向用户确认提交消息。

### Task 1: 重写红灯测试边界

**Files:**
- Read: `docs/sperpowers/specs/2026-04-12-vision-position-command-alignment-spec.md`
- Modify: `tests/unit/test_vision_protocol_rebuild.py`
- Modify: `tests/contract/test_main_vision_protocol_contract.py`

- [ ] **Step 1: 先对照规格列出必须翻转的断言项**

  逐项确认以下边界都要由测试锁定: 输出文本改为 `dx=<dx>,dy=<dy>,lock=0`; `format_vision_frame()` 不再携带序号参数; `build_follow_command()` 返回位置量字段而不是速度量字段; `ALIGN_X` 只输出横向量; `ALIGN_Y` 只输出纵向量; 无目标和保持区都输出零量; 仓库文字材料不再保留旧速度协议。

- [ ] **Step 2: 先改单元测试的格式化断言**

  在 `tests/unit/test_vision_protocol_rebuild.py` 中把 `format_vision_frame()` 的断言改成位置命令文本,并把函数签名期望收口为最小参数集合,不再接受旧序号参数和旧速度字段文案。

- [ ] **Step 3: 再改单元测试的阶段输出断言**

  把 `build_follow_command()` 的断言改成位置量语义,统一使用 `command_dx` / `command_dy` 作为返回字段名,并覆盖以下场景: `ALIGN_X` 只给 `command_dx`; `ALIGN_Y` 只给 `command_dy`; 无目标零量; 保持区零量; 横向和纵向各自限幅。

- [ ] **Step 4: 改契约测试的正式口径断言**

  在 `tests/contract/test_main_vision_protocol_contract.py` 中把必需 token 改成 `dx=<dx>,dy=<dy>,lock=0`,并把旧速度协议与旧运行时痕迹列为禁止项,至少包含: `f=1,m=1`, `x`/`y` 速度语义文字, 启动复位文本, 逐帧节流痕迹。

- [ ] **Step 5: 运行红灯测试确认失败方向正确**

  Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py tests/contract/test_main_vision_protocol_contract.py -q`

  Expected: FAIL, 且失败点集中在 `main.py`、`README.md`、`docs/Protocol.md` 仍保持旧速度输出口径。

### Task 2: 收口运行时代码到位置式逐帧输出

**Files:**
- Modify: `main.py`
- Test: `tests/unit/test_vision_protocol_rebuild.py`

- [ ] **Step 1: 调整顶部参数与注释语义**

  把 `main.py` 顶部与输出相关的参数、注释和文档字符串收口到“位置量”语义。保留目标选择、像素死区和面积目标参数,但把横向与纵向控制参数明确成位置量 P 系数,并为两个方向分别保留单帧上限。

- [ ] **Step 2: 删除不再需要的旧输出路径辅助项**

  清理旧速度协议相关的序号、启动复位和逐帧节流路径,包括不再需要的常量、辅助函数和主循环状态变量,确保主循环进入后每抓一帧就发送一帧。

- [ ] **Step 3: 收口最小文本格式化函数**

  调整 `format_vision_frame()` 的职责,让它只负责把 `command_dx` / `command_dy` 编码成 `dx=<dx>,dy=<dy>,lock=0` 的紧凑文本,统一零值输出为 `0`。

- [ ] **Step 4: 收口阶段判断输出字段**

  保留 `MARKER_MISSING`、`CENTER_HOLD`、`ALIGN_X`、`ALIGN_Y` 四段式判断,但让 `build_follow_command()` 只输出位置量字段 `command_dx` / `command_dy`: `ALIGN_X` 只给横向量, `ALIGN_Y` 只给纵向量, 其余阶段给零量。

- [ ] **Step 5: 收口主循环的三条发送路径**

  调整 `run()` 中无目标、有目标、保持区三类路径的串口写入逻辑,统一走新的位置文本格式,并确保无目标帧继续发送零量而不是停发。

- [ ] **Step 6: 跑单元测试验证主逻辑转绿**

  Run: `python3 -m pytest tests/unit/test_vision_protocol_rebuild.py -q`

  Expected: PASS.

### Task 3: 同步正式文档口径

**Files:**
- Modify: `README.md`
- Modify: `docs/Protocol.md`
- Test: `tests/contract/test_main_vision_protocol_contract.py`

- [ ] **Step 1: 收口 README 主线描述**

  把 `README.md` 中关于主线职责、输出格式和字段语义的表述改成“视觉端逐帧发送位置式命令”, 明确正式格式是 `dx=<dx>,dy=<dy>,lock=0`, 不再保留旧速度请求说明。

- [ ] **Step 2: 收口协议正文**

  把 `docs/Protocol.md` 中的链路用途、字段定义、合法示例、无目标规则和责任边界统一改成位置式命令口径, 并明确无目标发零量、每帧发送、不保留额外控制字段。

- [ ] **Step 3: 做一轮文档口径自查**

  检查两份文档里不再出现以下旧口径: `f=1,m=1`, “速度模式入口”, “`x` 表示横向速度”, “`y` 表示纵向速度”, 启动复位作为正式主线, 以及任何要求等待前一帧完成的描述。

- [ ] **Step 4: 跑契约测试验证文档与代码对齐**

  Run: `python3 -m pytest tests/contract/test_main_vision_protocol_contract.py -q`

  Expected: PASS.

### Task 4: 全量验证与交接

**Files:**
- Read: `docs/sperpowers/specs/2026-04-12-vision-position-command-alignment-spec.md`
- Verify: `main.py`
- Verify: `tests/unit/test_vision_protocol_rebuild.py`
- Verify: `tests/contract/test_main_vision_protocol_contract.py`
- Verify: `README.md`
- Verify: `docs/Protocol.md`

- [ ] **Step 1: 运行全量主机侧测试**

  Run: `python3 -m pytest tests/unit tests/contract -q`

  Expected: PASS.

- [ ] **Step 2: 对照规格逐项人工复核**

  对照 `docs/sperpowers/specs/2026-04-12-vision-position-command-alignment-spec.md` 复核以下五项: 每帧发送; 无目标发零量; 四段式阶段判断仍保留; 旧速度协议已清空; 文档和测试已经同步收口。

- [ ] **Step 3: 明确仍需板端确认的现场项**

  只保留与现场联调直接相关的检查项: `dx/dy` 正负方向是否符合车体系; 横向和纵向单帧上限是否平稳; 面积误差映射到 `dy` 的默认手感是否需要后续调参。

- [ ] **Step 4: 向用户汇报并等待执行方式选择**

  汇报计划已覆盖的实现范围、主机侧验证命令和仍需板端确认的事项, 然后等待用户决定是按子代理逐任务执行, 还是在当前会话内联执行。
