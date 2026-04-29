# 视觉端完整识别框上报设计文档

## 执行状态

- 状态: Archive


## 背景

当前 OpenArt 端 `main.py` 仍向 RT1021 发送 `x,y` 单点观测，而主控端协议已经切换为 `left,top,right,bottom` 完整识别框。若视觉端不跟进升级，车端将直接忽略旧载荷，视觉状态机无法继续工作。

## 目标

1. 让 OpenArt 按新协议发送 `left,top,right,bottom`
2. 保持现有目标选择逻辑不变，只调整上报格式
3. 保持单文件架构，不拆分 `main.py`

## 设计决策

- 候选目标仍使用现有 blob 检测结果与“更靠近中线、且更靠近底边”的排序逻辑
- 串口上报从中心点切换为 `best_blob.rect()` 对应的边界框
- 调试画面继续保留 `draw_rectangle()` 与 `draw_cross()`，便于人工观察选择结果
- 测试同步从 `x,y` 断言切换到 `left,top,right,bottom` 断言

## 风险与缓解

1. 风险: `blob.rect()` 的坐标顺序与主控协议不一致
   - 缓解: 在格式化函数中显式将 `(x, y, w, h)` 转为 `left,top,right,bottom`
2. 风险: 改动误伤现有目标选择策略
   - 缓解: 保持 `choose_best_candidate()` 逻辑不动，只改上报载荷

## 验收标准

- `format_vision_frame()` 返回 `left,top,right,bottom`
- `run()` 在发送时使用 `best_blob.rect()` 生成完整框载荷
- `tests/unit` 与 `tests/contract` 全部通过
