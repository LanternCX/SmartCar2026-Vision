"""辅车色标跟随距离标定 Demo

@file assistant/marker.py
@brief 在 OpenMV IDE 中观察正式色标跟随使用的边长和距离误差
"""

import sensor
import time


# 色标 LAB 阈值, 与辅车正式跟随配置一致
FOLLOW_THRESHOLD = (50, 100, 41, 127, -60, 127)
# 色块最小像素数
PIXELS_THRESHOLD = 200
# 色块最小外接矩形面积, 单位为 px²
AREA_THRESHOLD = 200
# 距离环目标色标边长, 单位为 px
TARGET_SPAN_PX = 100.0
# 图像曝光时间, 单位为 us
EXP_TIME_US = 500


def edge_length(point_a, point_b):
    """计算两个色标角点之间的像素距离"""

    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return (dx * dx + dy * dy) ** 0.5


def marker_span_px(blob):
    """按正式跟随逻辑计算色标最大边长"""

    corners = blob.min_corners()
    return max(
        edge_length(corners[0], corners[1]),
        edge_length(corners[1], corners[2]),
    )


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(0, time=2000)
sensor.set_auto_gain(False)  # pyright: ignore[reportCallIssue]
sensor.set_auto_whitebal(False)
sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot().replace(vflip=True, hmirror=True, transpose=False)
    img.lens_corr(strength=2.8, zoom=1.0)
    center_x = float(img.width()) / 2.0
    image_height = float(img.height())
    selected = None
    selected_score = None

    for blob in img.find_blobs(
        [FOLLOW_THRESHOLD],
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=True,
    ):
        left, top, width, height = blob.rect()
        bottom = float(top + height)
        score = (float(blob.cx()) - center_x) ** 2 + (bottom - image_height) ** 2
        if selected_score is None or score < selected_score:
            selected = blob
            selected_score = score

    if selected is None:
        img.draw_string(2, 2, "marker: missing", color=(255, 255, 255))
        print("missing fps=%.1f" % clock.fps())
        continue

    span_px = marker_span_px(selected)
    err_span_px = span_px - TARGET_SPAN_PX
    err_x_px = float(selected.cx()) - center_x

    img.draw_rectangle(selected.rect(), color=(255, 0, 0), thickness=2)
    img.draw_cross(selected.cx(), selected.cy(), color=(255, 0, 0))
    img.draw_cross(int(center_x), int(image_height / 2.0), color=(0, 255, 0))
    img.draw_string(2, 2, "span:%.1f px" % span_px, color=(255, 255, 255))
    img.draw_string(2, 14, "target:%.1f px" % TARGET_SPAN_PX, color=(255, 255, 255))
    img.draw_string(2, 26, "error:%+.1f px" % err_span_px, color=(255, 255, 255))

    print(
        "span_px=%.1f target_span_px=%.1f err_span_px=%+.1f err_x_px=%+.1f fps=%.1f"
        % (span_px, TARGET_SPAN_PX, err_span_px, err_x_px, clock.fps())
    )
