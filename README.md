# SmartCar2025 - OpenMV Vision

这是一个为智能车竞赛配置的 OpenMV 视觉项目，提供完整的开发环境和代码补全支持。

## 项目结构

```
SmartCar2025-Vision/
├── .venv/                  # Python 虚拟环境
├── .vscode/                # VS Code 配置
│   ├── settings.json       # 编辑器设置
│   └── launch.json         # 调试配置
├── stubs/                  # OpenMV API 类型提示
│   ├── sensor.py           # 传感器模块
│   ├── image.py            # 图像处理模块
│   ├── pyb.py              # 硬件控制模块
│   ├── time.py             # 时间模块
│   └── machine.py          # 机器模块
├── examples/               # 示例程序
│   ├── color_tracking.py   # 颜色识别
│   ├── line_following.py   # 巡线示例
│   └── apriltag_detection.py  # AprilTag检测
├── main.py                 # 主程序入口
├── .gitignore              # Git 忽略文件
└── README.md               # 本文件
```

## 环境配置

### 已安装的组件

1. **Python 虚拟环境** (.venv)
   - 独立的 Python 环境，避免包冲突
   - 位置: `SmartCar2025-Vision/.venv`

2. **OpenMV 包**
   - 提供基础的 OpenMV 库支持
   - 已自动安装到虚拟环境

3. **OpenMV Stubs**
   - 完整的 OpenMV API 类型提示
   - 支持 VS Code 智能补全
   - 包含主要模块: sensor, image, pyb, time, machine

4. **VS Code 配置**
   - 自动识别虚拟环境
   - 配置了 Python 路径和类型提示
   - 启用了代码格式化

## 使用方法

### 1. 激活虚拟环境

在 VS Code 中打开终端 (Ctrl + `)，虚拟环境会自动激活。

或者手动激活:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 在 VS Code 中编码

- 打开任意 `.py` 文件
- 输入 `import sensor` 或 `import image` 会自动获得代码补全
- 所有 OpenMV API 都有完整的类型提示和文档

示例代码补全:
```python
import sensor
import image

sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # 会有自动补全提示
sensor.set_framesize(sensor.QVGA)

img = sensor.snapshot()
img.find_blobs(...)  # 会显示参数提示
```

### 3. 转移到 OpenMV IDE 调试

1. 在 VS Code 中完成代码编写
2. 复制代码到 OpenMV IDE
3. 连接 OpenMV 摄像头
4. 在 OpenMV IDE 中运行和调试

**或者使用文件同步:**

可以直接保存文件到 OpenMV 的存储中，然后在 OpenMV IDE 中打开：
- 连接 OpenMV 摄像头
- 在 OpenMV IDE 中打开此项目目录中的文件
- 直接运行

## 示例程序说明

### main.py - Hello World
基础的 OpenMV 示例，演示:
- 传感器初始化
- 图像捕获
- 基本绘图功能
- FPS 计算

### examples/color_tracking.py - 颜色识别
智能车常用的颜色跟踪:
- 识别红、绿、蓝色块
- 计算色块中心位置
- 显示色块边界

**使用提示:** 需要根据实际光照环境调整颜色阈值

### examples/line_following.py - 巡线
线性回归巡线算法:
- 二值化图像
- ROI 区域设置
- 线性回归计算
- 输出偏移量和角度

**适用场景:** 智能车循线比赛

### examples/apriltag_detection.py - AprilTag检测
标签识别与定位:
- 检测 AprilTag 标签
- 获取标签 ID 和位置
- 计算旋转角度
- 支持多标签同时检测

**适用场景:** 智能车定位、导航

## 调试技巧

### 在 VS Code 中模拟测试
虽然 VS Code 不能直接运行 OpenMV 硬件代码，但可以:
1. 测试算法逻辑
2. 使用 OpenCV 进行图像处理验证
3. 调试数据处理部分

### 在 OpenMV IDE 中真机调试
1. 使用 `print()` 输出调试信息
2. 查看串口输出
3. 使用帧缓冲查看器实时查看图像
4. 调整参数观察效果

## 常用 API 速查

### 传感器初始化
```python
sensor.reset()                      # 重置传感器
sensor.set_pixformat(sensor.RGB565) # 设置像素格式
sensor.set_framesize(sensor.QVGA)   # 设置分辨率
sensor.skip_frames(time=2000)       # 跳过初始帧
```

### 图像捕获与处理
```python
img = sensor.snapshot()             # 捕获图像
img.binary([threshold])             # 二值化
img.find_blobs([threshold])         # 查找色块
img.get_regression([threshold])     # 线性回归
img.find_apriltags()                # 查找 AprilTag
```

### 绘图函数
```python
img.draw_rectangle(x, y, w, h)      # 绘制矩形
img.draw_circle(x, y, r)            # 绘制圆
img.draw_line(x0, y0, x1, y1)       # 绘制直线
img.draw_string(x, y, text)         # 绘制文本
```

## 性能优化建议

1. **选择合适的分辨率**
   - QQVGA (160x120) - 最快
   - QVGA (320x240) - 平衡
   - VGA (640x480) - 最清晰但慢

2. **使用 ROI (感兴趣区域)**
   - 只处理图像的特定区域
   - 显著提升处理速度

3. **降低颜色深度**
   - 使用 GRAYSCALE 而非 RGB565
   - 对于巡线等应用足够

4. **优化算法**
   - 减少不必要的图像处理
   - 使用合适的阈值参数

## 故障排查

### 代码补全不工作
1. 确认虚拟环境已激活
2. 检查 `.vscode/settings.json` 中的路径
3. 重启 VS Code
4. 运行命令: `Python: Select Interpreter`

### OpenMV IDE 找不到设备
1. 检查 USB 连接
2. 安装 OpenMV 驱动程序
3. 尝试不同的 USB 端口

### 图像识别效果不佳
1. 调整光照环境
2. 重新标定颜色阈值
3. 调整 ROI 区域
4. 使用 OpenMV IDE 的阈值编辑器

## 相关资源

- [OpenMV 官方文档](https://docs.openmv.io/)
- [OpenMV 中文论坛](https://singtown.com/openmv/)
- [智能车竞赛官网](https://www.smartcar.club/)

## 开发建议

1. **先在 VS Code 中编写代码** - 享受完整的代码补全和类型提示
2. **在 OpenMV IDE 中调试** - 实时查看图像和调试输出
3. **版本控制** - 使用 Git 管理代码版本
4. **记录参数** - 不同环境下的阈值参数要记录保存

## 许可证

本项目供学习和竞赛使用。

---

**Happy Coding! 🚗📷**
