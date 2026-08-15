# YOLO3 Nano 模型训练

本目录保存模型训练、评估和导出工具。数据集不包含在仓库中，部署模型统一使用 [`../yolo.tflite`](../yolo.tflite)。

## 核心入口

- [`config.cfg`](config.cfg)：训练、模型结构和推理后处理参数
- [`train.py`](train.py)：训练、INT8 量化、TFLite 导出和后处理入口
- [`model_resnet.py`](model_resnet.py)：MobileNetV2 骨干、检测头和损失函数
- [`utils.py`](utils.py)：配置读取、数据增强和学习率调度
- [`voc_convertor.py`](voc_convertor.py)：将 VOC 数据集转换为训练索引
- [`kmeans.py`](kmeans.py)：根据数据集重新聚类 anchors
- [`evaluate.py`](evaluate.py)：TFLite 模型 mAP 评估
- [`tflite_add_post_processing.py`](tflite_add_post_processing.py)：调用对应平台的动态库，为模型加入 OpenART 检测后处理节点

训练顺序为 `voc_convertor.py` → `kmeans.py` → `train.py`，训练结果使用 `evaluate.py` 评估。

## 影响模型表现的关键参数

| 参数 | 位置 | 作用 |
| --- | --- | --- |
| `width`、`height` | `config.cfg` | 输入分辨率，直接影响小目标识别、推理速度和内存占用 |
| `num_heads`、`divider` | `config.cfg` | 检测头数量和输出步长，决定多尺度能力与计算量 |
| `yolo3_anchors.txt` | 当前目录 | 目标框先验尺寸，更换数据集后需要重新聚类 |
| `iou_type`、`iou_threshold` | `config.cfg` | 定位损失类型和忽略阈值 |
| `obj_scale`、`noobj_scale` | `config.cfg` | 平衡目标与背景样本的损失 |
| `batch_size`、`total_epochs` | `config.cfg` | 影响训练稳定性、显存占用和训练时长 |
| 学习率 | `train.py` | 使用余弦调度，`lr_start=1e-2`、`lr_min=1e-6` |
| 数据增强 | `utils.py` | 包括缩放、翻转、色相、饱和度和亮度扰动 |
| INT8 校准样本 | `train.py` | 使用 100 个训练样本，样本覆盖度影响量化后的识别效果 |

`nms_iou_threshold`、`nms_score_threshold` 和 `max_detections` 只改变部署时的过滤结果，不改变模型已经学到的能力。

## 部署模型核对

对 [`../yolo.tflite`](../yolo.tflite) 的 FlatBuffer 解析结果如下：

| 项目 | 部署模型 | 训练目录 | 结论 |
| --- | --- | --- | --- |
| 输入 | `1×80×80×3`，INT8 | `96×96` | 不一致 |
| 检测头 | 1 个 | `num_heads=1` | 一致 |
| 输出步长 | 原始输出为 `5×5`，对应步长 16 | `divider=16` | 一致 |
| 类别数 | 5 | `num_classes=5` | 一致 |
| anchors | `(6,6)`、`(11,10)`、`(19,17)` | `(8,7)`、`(14,13)`、`(26,22)` | 不一致 |
| NMS IoU 阈值 | 0.45 | 0.45 | 一致 |
| 置信度阈值 | 0.45 | 0.45 | 一致 |
| 最大检测数 | 5 | 5 | 一致 |
| 后处理 | `YOLO_Detection_PostProcess` | 训练结束时调用动态库添加 | 一致 |

部署模型只保存类别数量，不保存类别名称。运行代码与训练配置使用相同的顺序：`tennis`、`red`、`blue`、`brown`、`white`。

`config.cfg` 和 `yolo3_anchors.txt` 不能完整复现部署模型。复训现有部署模型时，需要先确认使用 80×80 输入和部署模型中的 anchors；训练新模型时，以新模型采用的配置为准。
