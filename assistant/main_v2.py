"""辅车 OpenART 视觉入口 v2."""

import gc
import image
import sensor
import tf
import time
from machine import UART


UART_ID = 2
UART_BAUDRATE = 115200
EXP_TIME_US = 500
RELIABLE_RESEND_INTERVAL_MS = 100


class Mode:
    UDP = 0x01
    TCP = 0x02
    ACK = 0x03


class Topic:
    LOCAL_VISION_VELOCITY = 0x01
    ASSISTANT_VISION_TASK_SYNC = 0x11
    ASSISTANT_VISION_EVENT_REPORT = 0x13


class RunMode:
    FOLLOW = "follow"
    APPROACH_OBJECT = "approach_object"
    ORBIT_OBJECT = "orbit_object"
    RETURN_LINE = "return_line"


class State:
    APPROACH_OBJECT = 2
    ORBIT = 3
    TRANSPORT_OBJECT = 4
    RETURN_FOLLOW = 6


class Target:
    NONE = 0
    OBJECT = 1


class Task:
    SEARCH = 1
    TRANSPORT = 2
    ORBIT = 3
    RETURN_GARAGE_LINE = 5


class Event:
    TARGET_FOUND = 6
    ALIGNED = 7
    RETURN_GARAGE_FINISHED = 12


SEQ_RING_SIZE = 256
SEQ_HALF_RING = 128
FRAME_BODY_SIZE = 8
FRAME_HEAD = 0xA5
FRAME_SIZE = 13
YOLO_MODEL_PATH = "/sd/yolo.tflite"
YOLO_IMAGE_COPY_SCALE = 0.75
YOLO_MIN_SCORE = 0.90
YOLO_LABELS = ("tennis", "red", "blue", "brown", "white")
VISION_REFERENCE_FPS = 18.0

OBJECT_BLOB_MERGE_MARGIN = 0
OBJECT_BLOB_PIXELS_THRESHOLD = 200
OBJECT_BLOB_AREA_THRESHOLD = 200
FOLLOW_TASKS = (("marker", (37, 57, 64, 95, -64, 20)),)
OBJECT_TASKS = (
    ('red', ((16, 51, 21, 84, -11, 52),), 3, 30, 70, 90, True),
)
RETURN_LINE_YELLOW_THRESHOLD = (58, 87, -32, -12, 64, 84)

FOLLOW_X_DEADZONE_PX = 5.0
FOLLOW_TARGET_Y = 45.0
FOLLOW_Y_DEADZONE_PX = 8.0
FOLLOW_CONTROL_KP_X = 0.04
FOLLOW_CONTROL_KP_Y = -0.10
FOLLOW_CONTROL_MIN_SPEED = 0
FOLLOW_CONTROL_MAX_Y = 5

OBJECT_MISSING_SEARCH_VX = 0.0
OBJECT_MISSING_SEARCH_VY = 0.0
OBJECT_APPROACH_KP_X = 0.05
OBJECT_APPROACH_KP_Y = -0.2
OBJECT_APPROACH_MIN_SPEED = 2
OBJECT_APPROACH_DEADZONE_X_PX = 15.0
OBJECT_APPROACH_DEADZONE_Y_PX = 8.0
OBJECT_APPROACH_MAX_VX = 5.0
OBJECT_APPROACH_MAX_VY = 5.0
OBJECT_ORBIT_KP_X = 0.05
OBJECT_ORBIT_KP_Y = -0.15
OBJECT_ORBIT_MIN_SPEED = 0
OBJECT_ORBIT_DEADZONE_X_PX = 15.0
OBJECT_ORBIT_DEADZONE_Y_PX = 8.0
OBJECT_ORBIT_MAX_VX = 5.0
OBJECT_ORBIT_MAX_VY = 5.0
OBJECT_APPROACH_TARGET_X_PX = 160.0
OBJECT_APPROACH_TARGET_Y_PX = 200.0
OBJECT_ORBIT_TARGET_X_PX = 160.0
OBJECT_ORBIT_TARGET_Y_PX = 200.0
ASSISTANT_TRANSPORT_TARGET_Y_PX = 240.0
OBJECT_MIN_AREA = 50.0
OBJECT_X_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_X_PX
OBJECT_Y_TOLERANCE_PX = OBJECT_APPROACH_DEADZONE_Y_PX
OBJECT_STABLE_FRAMES = 3

RETURN_LINE_SAMPLE_HALF_WIDTH_PX = 5
RETURN_LINE_TARGET_Y_PX = 220.0
RETURN_LINE_DEADZONE_Y_PX = 4.0
RETURN_LINE_KP_Y = -0.05
RETURN_LINE_MAX_VY = 5.0
RETURN_LINE_MIN_SPEED = 0.0
RETURN_LINE_MAX_THICKNESS_PX = 30
RETURN_LINE_MIN_HORIZONTAL_CONNECTED_PX = 50
RETURN_LINE_MISSING_FINISH_FRAMES = 5

_I16_MIN = -32768
_I16_MAX = 32767
_SCALE = 1000


def _require_u8(value):
    value = int(value)
    if value < 0 or value > 0xFF:
        raise ValueError("u8 out of range")
    return value


def _saturate_i16(value):
    if value < _I16_MIN:
        return _I16_MIN
    if value > _I16_MAX:
        return _I16_MAX
    return value


def _pack_i16(value):
    value = _saturate_i16(int(value))
    if value < 0:
        value += 0x10000
    return bytes((value & 0xFF, (value >> 8) & 0xFF))


def _unpack_i16(body, offset):
    value = int(body[offset]) | (int(body[offset + 1]) << 8)
    if value >= 0x8000:
        value -= 0x10000
    return value


def _pack_scaled(value):
    return _pack_i16(round(float(value) * _SCALE))


def _unpack_scaled(body, offset):
    return _unpack_i16(body, offset) / float(_SCALE)


def encode_frame(mode, topic, seq, body):
    mode = _require_u8(mode)
    topic = _require_u8(topic)
    seq = _require_u8(seq)
    if not isinstance(body, bytes):
        body = bytes(body)
    if len(body) > FRAME_BODY_SIZE:
        raise ValueError("body too large")
    payload = bytes((mode, topic, seq)) + body + (b"\x00" * (FRAME_BODY_SIZE - len(body)))
    return bytes((FRAME_HEAD,)) + payload + bytes((_crc8(payload),))


def decode_frame(frame_bytes):
    if isinstance(frame_bytes, memoryview):
        frame_bytes = frame_bytes.tobytes()
    elif isinstance(frame_bytes, bytearray):
        frame_bytes = bytes(frame_bytes)
    if not isinstance(frame_bytes, bytes):
        return None
    if len(frame_bytes) != FRAME_SIZE:
        return None
    if frame_bytes[0] != FRAME_HEAD:
        return None
    payload = frame_bytes[1:-1]
    if _crc8(payload) != frame_bytes[-1]:
        return None
    return {
        "mode": payload[0],
        "topic": payload[1],
        "seq": payload[2],
        "body": payload[3:],
    }


def _crc8(data):
    crc = 0
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_velocity_body(vx, vy, omega=0.0, has_omega=False):
    return (
        _pack_scaled(vx)
        + _pack_scaled(vy)
        + _pack_scaled(omega)
        + bytes((1 if has_omega else 0,))
    )


def decode_velocity_body(body):
    return {
        "vx": _unpack_scaled(body, 0),
        "vy": _unpack_scaled(body, 2),
        "omega": _unpack_scaled(body, 4),
        "has_omega": bool(body[6]),
    }


def encode_assistant_vision_task_sync_body(state_value, target, arg):
    return bytes((_require_u8(state_value), _require_u8(target))) + _pack_i16(arg)


def decode_assistant_vision_task_sync_body(body):
    return {
        "state": int(body[0]),
        "target": int(body[1]),
        "arg": _unpack_i16(body, 2),
    }


def pack_task_arg(config_id, object_id):
    packed = (int(config_id) & 0xFF) | ((int(object_id) & 0xFF) << 8)
    if packed >= 0x8000:
        packed -= 0x10000
    return packed


def unpack_task_arg_config(arg):
    return int(arg) & 0xFF


def unpack_task_arg_object_id(arg):
    return (int(arg) >> 8) & 0xFF


def encode_assistant_vision_event_report_body(event, value):
    return bytes((_require_u8(event),)) + _pack_i16(value)


def decode_assistant_vision_event_report_body(body):
    return {
        "event": int(body[0]),
        "value": _unpack_i16(body, 1),
    }


def format_vision_frame(vx, vy):
    return encode_frame(
        Mode.UDP,
        Topic.LOCAL_VISION_VELOCITY,
        0,
        encode_velocity_body(vx, vy, 0.0, False),
    )


def format_ack_frame(reliable_seq):
    return encode_frame(Mode.ACK, Topic.ASSISTANT_VISION_TASK_SYNC, reliable_seq, b"")


def format_event_frame(reliable_seq, event, value):
    return encode_frame(
        Mode.TCP,
        Topic.ASSISTANT_VISION_EVENT_REPORT,
        reliable_seq,
        encode_assistant_vision_event_report_body(event, value),
    )


def parse_sync_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.TCP or frame["topic"] != Topic.ASSISTANT_VISION_TASK_SYNC:
        return None
    packet = decode_assistant_vision_task_sync_body(frame["body"])
    return {
        "reliable_seq": int(frame["seq"]),
        "state": int(packet["state"]),
        "target": int(packet["target"]),
        "arg": int(packet["arg"]),
    }


def parse_ack_packet(frame_bytes):
    frame = decode_frame(frame_bytes)
    if frame is None:
        return None
    if frame["mode"] != Mode.ACK or frame["topic"] != Topic.ASSISTANT_VISION_EVENT_REPORT:
        return None
    return {"reliable_seq": int(frame["seq"])}


def is_newer_seq(seq, last_seq):
    if last_seq is None:
        return True
    diff = (int(seq) - int(last_seq)) % SEQ_RING_SIZE
    return diff != 0 and diff < SEQ_HALF_RING


def default_now_ms():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


def should_resend(now_ms, last_sent_ms, interval_ms):
    if last_sent_ms is None:
        return True
    interval_ms = int(interval_ms)
    if interval_ms <= 0:
        return True
    if int(now_ms) < int(last_sent_ms):
        return True
    return int(now_ms) - int(last_sent_ms) >= interval_ms


def _apply_min_speed(value, min_speed, limit=None):
    value = float(value)
    if limit is not None:
        limit = abs(float(limit))
        if value > limit:
            value = limit
        if value < -limit:
            value = -limit
    min_speed = abs(float(min_speed))
    if value == 0.0:
        return 0.0
    if limit is not None and min_speed > limit:
        min_speed = limit
    if 0.0 < value < min_speed:
        return min_speed
    if -min_speed < value < 0.0:
        return -min_speed
    return value


def _clamp(value, limit):
    value = float(value)
    limit = abs(float(limit))
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def current_frame_time_scale():
    frame_interval_ms = float(state.current_frame_interval_ms)
    if frame_interval_ms <= 0.0:
        return 1.0
    reference_interval_ms = 1000.0 / VISION_REFERENCE_FPS
    return reference_interval_ms / frame_interval_ms


def _build_axis_velocity(error, kp, limit=None):
    value = float(error) * float(kp) * current_frame_time_scale()
    if limit is None:
        return value
    return _clamp(value, float(limit) * current_frame_time_scale())


def _axis_p_velocity(error, deadzone, kp, limit, min_speed):
    error = float(error)
    if abs(error) <= float(deadzone):
        return 0.0
    if float(kp) == 0.0:
        return 0.0
    return _apply_min_speed(
        _build_axis_velocity(error, kp, limit),
        float(min_speed) * current_frame_time_scale(),
        float(limit) * current_frame_time_scale(),
    )


def build_follow_command(valid, err_x, err_y):
    if int(valid) != 1:
        return {
            "valid": 0,
            "err_x": 0.0,
            "err_y": 0.0,
            "command_vx": 0.0,
            "command_vy": 0.0,
        }
    command_vx = _axis_p_velocity(
        err_x,
        FOLLOW_X_DEADZONE_PX,
        FOLLOW_CONTROL_KP_X,
        FOLLOW_CONTROL_MAX_Y,
        FOLLOW_CONTROL_MIN_SPEED,
    )
    command_vy = _axis_p_velocity(
        err_y,
        FOLLOW_Y_DEADZONE_PX,
        FOLLOW_CONTROL_KP_Y,
        FOLLOW_CONTROL_MAX_Y,
        FOLLOW_CONTROL_MIN_SPEED,
    )
    return {
        "valid": 1,
        "err_x": float(err_x),
        "err_y": float(err_y),
        "command_vx": command_vx,
        "command_vy": command_vy,
    }


def blob_rect_to_bbox(rect):
    left, top, width, height = rect
    return float(left), float(top), float(left + width), float(top + height)


def normalize_bbox_for_protocol(left, top, right, bottom, img_height):
    normalized_top = float(img_height) - float(bottom)
    normalized_bottom = float(img_height) - float(top)
    return float(left), normalized_top, float(right), normalized_bottom


def compute_lateral_error(blob_cx, cx_screen):
    return float(blob_cx) - float(cx_screen)


def edge_length(p0, p1):
    dx = float(p0[0]) - float(p1[0])
    dy = float(p0[1]) - float(p1[1])
    return (dx * dx + dy * dy) ** 0.5


def get_marker_corners(blob):
    corners_fn = getattr(blob, "min_corners", None)
    if corners_fn is not None:
        return corners_fn()
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def compute_marker_span(corners):
    if len(corners) < 4:
        return 0.0
    return max(edge_length(corners[0], corners[1]), edge_length(corners[1], corners[2]))


def compute_vertical_error(marker_span, target_span):
    return float(marker_span) - float(target_span)


def blob_area(blob):
    area_fn = getattr(blob, "area", None)
    if area_fn is not None:
        return float(area_fn())
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(0.0, float(right) - float(left)) * max(0.0, float(bottom) - float(top))


class YoloDetectionBlob:
    def __init__(self, left, top, right, bottom, label, score):
        self._left = float(left)
        self._top = float(top)
        self._right = float(right)
        self._bottom = float(bottom)
        self.label = int(label)
        self.score = float(score)

    def rect(self):
        left = int(round(self._left))
        top = int(round(self._top))
        right = int(round(self._right))
        bottom = int(round(self._bottom))
        return left, top, right - left, bottom - top

    def cx(self):
        return (self._left + self._right) / 2.0

    def cy(self):
        return (self._top + self._bottom) / 2.0

    def area(self):
        return max(0.0, self._right - self._left) * max(0.0, self._bottom - self._top)

    def min_corners(self):
        return (
            (self._left, self._top),
            (self._right, self._top),
            (self._right, self._bottom),
            (self._left, self._bottom),
        )


def blob_max_side_length(blob):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    return max(float(right - left), float(bottom - top))


def task_thresholds(thresholds):
    if len(thresholds) == 6 and not isinstance(thresholds[0], (tuple, list)):
        return (thresholds,)
    return thresholds


def object_task_parts(task):
    if len(task) >= 7:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            max(0, int(task[5])),
            bool(task[6]),
        )
    if len(task) >= 6:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            0,
            bool(task[5]),
        )
    if len(task) >= 5:
        return (
            task[0],
            task_thresholds(task[1]),
            int(task[2]),
            int(task[3]),
            int(task[4]),
            0,
            True,
        )
    return (
        task[0],
        task_thresholds(task[1]),
        OBJECT_BLOB_MERGE_MARGIN,
        OBJECT_BLOB_PIXELS_THRESHOLD,
        OBJECT_BLOB_AREA_THRESHOLD,
        0,
        True,
    )


def object_task_name_from_id(object_id):
    object_id = int(object_id)
    if object_id <= 0:
        return None
    index = object_id - 1
    if index >= len(OBJECT_TASKS):
        return None
    return OBJECT_TASKS[index][0]


def load_yolo_model():
    if state.yolo_net is None:
        state.yolo_net = tf.load(YOLO_MODEL_PATH)
    return state.yolo_net


def _copy_image_for_yolo(img):
    return img.copy(YOLO_IMAGE_COPY_SCALE, 1)


def _label_name(label):
    label = int(label)
    if 0 <= label < len(YOLO_LABELS):
        return YOLO_LABELS[label]
    return "unknown"


def _build_yolo_object_candidates(img, yolo_net=None):
    net = yolo_net
    if net is None:
        net = load_yolo_model()
    detect_img = _copy_image_for_yolo(img)
    image_width = float(img.width())
    image_height = float(img.height())
    allowed_task_names = {task[0] for task in OBJECT_TASKS}
    candidates = []
    for detected in tf.detect(net, detect_img):
        x1, y1, x2, y2, label, score = detected
        if float(score) <= float(YOLO_MIN_SCORE):
            continue
        task_name = _label_name(label)
        if task_name not in allowed_task_names:
            continue
        left = float(x1) * image_width
        top = float(y1) * image_height
        right = float(x2) * image_width
        bottom = float(y2) * image_height
        if right <= left or bottom <= top:
            continue
        blob = YoloDetectionBlob(left, top, right, bottom, label, score)
        _, _, _, protocol_bottom = normalize_bbox_for_protocol(
            left,
            top,
            right,
            bottom,
            image_height,
        )
        candidates.append((task_name, blob.cx(), blob.cy(), protocol_bottom, blob.area(), blob))
    return candidates


def _find_blobs_with_task_config(img, thresholds, pixels_threshold, area_threshold, merge_margin):
    try:
        return img.find_blobs(
            list(thresholds),
            pixels_threshold=pixels_threshold,
            area_threshold=1,
            merge=True,
            margin=max(0, int(merge_margin)),
        )
    except TypeError:
        return img.find_blobs(
            list(thresholds),
            pixels_threshold=pixels_threshold,
            area_threshold=1,
            merge=True,
        )


def _blob_code(blob):
    code_fn = getattr(blob, "code", None)
    if code_fn is not None:
        return int(code_fn())
    try:
        return int(blob[8])
    except Exception:
        return None


def _blob_matches_required_thresholds(blob, threshold_count, require_all_thresholds):
    if int(threshold_count) <= 1 or not bool(require_all_thresholds):
        return True
    code = _blob_code(blob)
    if code is None:
        return True
    expected_code = (1 << int(threshold_count)) - 1
    return (code & expected_code) == expected_code


def blob_bbox_overlaps(left, top, right, bottom, other_blob):
    other_left, other_top, other_right, other_bottom = blob_rect_to_bbox(other_blob.rect())
    return (
        min(float(right), float(other_right)) > max(float(left), float(other_left))
        and min(float(bottom), float(other_bottom)) > max(float(top), float(other_top))
    )


def blob_matches_all_thresholds(img, blob, thresholds, pixels_threshold, area_threshold, merge):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    for threshold in thresholds[1:]:
        blobs = img.find_blobs(
            [threshold],
            pixels_threshold=pixels_threshold,
            area_threshold=area_threshold,
            merge=merge,
        )
        matched = False
        for other_blob in blobs:
            if blob_bbox_overlaps(left, top, right, bottom, other_blob):
                matched = True
                break
        if not matched:
            return False
    return True


def build_blob_candidates(img):
    candidates = []
    img_height = img.height()
    for task_name, threshold in FOLLOW_TASKS:
        blobs = img.find_blobs(
            [threshold],
            pixels_threshold=OBJECT_BLOB_PIXELS_THRESHOLD,
            area_threshold=OBJECT_BLOB_AREA_THRESHOLD,
            merge=True,
        )
        for blob in blobs:
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, bottom = normalize_bbox_for_protocol(left, top, right, bottom, img_height)
            marker_span = compute_marker_span(get_marker_corners(blob))
            candidates.append((task_name, blob.cx(), blob.cy(), bottom, marker_span, blob))
    return candidates


def _build_blob_object_candidates(img):
    candidates = []
    img_height = img.height()
    for task in OBJECT_TASKS:
        (
            task_name,
            thresholds,
            merge_margin,
            pixels_threshold,
            area_threshold,
            max_side_length,
            require_all_thresholds,
        ) = object_task_parts(task)
        if len(thresholds) <= 0:
            continue
        blobs = _find_blobs_with_task_config(
            img,
            thresholds,
            pixels_threshold,
            area_threshold,
            merge_margin,
        )
        for blob in blobs:
            if not _blob_matches_required_thresholds(blob, len(thresholds), require_all_thresholds):
                continue
            if len(thresholds) > 1 and not blob_matches_all_thresholds(
                img,
                blob,
                thresholds,
                pixels_threshold,
                area_threshold,
                True,
            ):
                continue
            if blob_area(blob) < float(area_threshold):
                continue
            if int(max_side_length) > 0 and blob_max_side_length(blob) > float(max_side_length):
                continue
            left, top, right, bottom = blob_rect_to_bbox(blob.rect())
            _, _, _, bottom = normalize_bbox_for_protocol(left, top, right, bottom, img_height)
            candidates.append((task_name, blob.cx(), blob.cy(), bottom, blob_area(blob), blob))
    return candidates


def build_object_candidates(img, yolo_net=None):
    if getattr(img, "copy", None) is not None:
        return _build_yolo_object_candidates(img, yolo_net)
    return _build_blob_object_candidates(img)


def build_object_blob_candidates(img, yolo_net=None):
    return build_object_candidates(img, yolo_net)


def _pixel_to_lab(pixel):
    if pixel is None:
        return None
    return image.rgb_to_lab(pixel)


def _pixel_matches_threshold(pixel, threshold):
    lab = _pixel_to_lab(pixel)
    if lab is None:
        return False
    try:
        if len(lab) < 3:
            return False
    except TypeError:
        return False
    return (
        float(threshold[0]) <= float(lab[0]) <= float(threshold[1])
        and float(threshold[2]) <= float(lab[1]) <= float(threshold[3])
        and float(threshold[4]) <= float(lab[2]) <= float(threshold[5])
    )


def _return_line_pixel_matches(img, x, y, image_width, image_height):
    max_x = int(image_width) - 1
    max_y = int(image_height) - 1
    return _pixel_matches_threshold(
        img.get_pixel(max_x - int(x), max_y - int(y)),
        RETURN_LINE_YELLOW_THRESHOLD,
    )


def _return_line_has_horizontal_connected_at(img, x, y, image_width, image_height, required_connected):
    if not _return_line_pixel_matches(img, x, y, image_width, image_height):
        return False
    connected = 0
    left = int(x) - 1
    while left >= 0 and _return_line_pixel_matches(img, left, y, image_width, image_height):
        connected += 1
        if connected >= int(required_connected):
            return True
        left -= 1
    right = int(x) + 1
    max_x = int(image_width) - 1
    while right <= max_x and _return_line_pixel_matches(img, right, y, image_width, image_height):
        connected += 1
        if connected >= int(required_connected):
            return True
        right += 1
    return False


def _return_line_sample_columns(center_x, half_width):
    yield int(center_x)
    for offset in range(1, int(half_width) + 1):
        yield int(center_x) - offset
        yield int(center_x) + offset


def _return_line_y_on_column(img, x, image_width, image_height):
    top = None
    bottom = None
    for y in range(0, int(image_height)):
        if not _return_line_pixel_matches(img, x, y, image_width, image_height):
            continue
        if top is None:
            top = int(y)
        bottom = int(y)
    if top is None or bottom is None:
        return None
    max_thickness = int(RETURN_LINE_MAX_THICKNESS_PX)
    if max_thickness > 0 and int(bottom) - int(top) > max_thickness:
        top = int(bottom) - max_thickness
    return (float(top) + float(bottom)) / 2.0


def _build_return_line_y_from_pixels(img, image_width, image_height, previous_line_y=None):
    get_pixel = getattr(img, "get_pixel", None)
    if get_pixel is None:
        return None
    center_x = int(float(image_width) / 2.0)
    half_width = int(RETURN_LINE_SAMPLE_HALF_WIDTH_PX)
    saw_candidate = False
    for x in _return_line_sample_columns(center_x, half_width):
        if x < 0 or x >= int(image_width):
            continue
        line_y = _return_line_y_on_column(img, x, image_width, image_height)
        if line_y is None:
            continue
        saw_candidate = True
        if _return_line_has_horizontal_connected_at(
            img,
            x,
            int(round(line_y)),
            image_width,
            image_height,
            RETURN_LINE_MIN_HORIZONTAL_CONNECTED_PX,
        ):
            return line_y
    if saw_candidate:
        return previous_line_y
    return None


def build_return_line_y_from_image(img, image_width, image_height, previous_line_y=None):
    return _build_return_line_y_from_pixels(img, image_width, image_height, previous_line_y)


def build_return_line_velocity_from_y(line_y):
    if line_y is None:
        return 0.0, 0.0
    err_y = float(line_y) - float(RETURN_LINE_TARGET_Y_PX)
    if abs(err_y) <= float(RETURN_LINE_DEADZONE_Y_PX):
        return 0.0, 0.0
    return 0.0, _apply_min_speed(
        err_y * float(RETURN_LINE_KP_Y) * current_frame_time_scale(),
        float(RETURN_LINE_MIN_SPEED) * current_frame_time_scale(),
        float(RETURN_LINE_MAX_VY) * current_frame_time_scale(),
    )


def choose_best_candidate(candidates, target_x, target_y):
    return min(
        candidates,
        key=lambda item: (float(item[1]) - float(target_x)) ** 2
        + (float(item[3]) - float(target_y)) ** 2,
    )


def choose_largest_area_candidate(candidates):
    return max(candidates, key=lambda item: float(item[4]))


def filter_candidates_in_target_window(candidates, target_x, target_y, tolerance_x, tolerance_y):
    _ = (target_x, tolerance_x)
    return [
        candidate
        for candidate in candidates
        if abs(float(candidate[3]) - float(target_y)) <= float(tolerance_y)
    ]


def should_filter_candidates_by_target_window(vision_state):
    if vision_state.current_sync is None:
        return False
    return (
        int(vision_state.current_sync["state"]) == int(State.TRANSPORT_OBJECT)
        and int(vision_state.current_object_config_id()) == int(Task.TRANSPORT)
    )


def _draw_debug_line(img, x0, y0, x1, y1):
    draw_line = getattr(img, "draw_line", None)
    if draw_line is None:
        return
    try:
        draw_line(int(x0), int(y0), int(x1), int(y1), color=(255, 255, 0))
    except TypeError:
        draw_line(int(x0), int(y0), int(x1), int(y1))


def _draw_debug_cross(img, x, y):
    draw_cross = getattr(img, "draw_cross", None)
    if draw_cross is None:
        return
    try:
        draw_cross(int(x), int(y), color=(255, 0, 0))
    except TypeError:
        draw_cross(int(x), int(y))


def _draw_debug_protocol_point(img, image_width, image_height, x, y):
    draw_x = int(image_width) - 1 - int(x)
    draw_y = int(image_height) - 1 - int(y)
    _draw_debug_cross(img, draw_x, draw_y)


def _draw_debug_text(img, x, y, text):
    draw_string = getattr(img, "draw_string", None)
    if draw_string is None:
        return
    try:
        draw_string(int(x), int(y), str(text), color=(255, 255, 255))
    except TypeError:
        draw_string(int(x), int(y), str(text))


def draw_selected_marker(img, blob, pixel_x, pixel_y):
    left, top, right, bottom = blob_rect_to_bbox(blob.rect())
    width = int(right) - int(left)
    height = int(bottom) - int(top)
    draw_rectangle = getattr(img, "draw_rectangle", None)
    if draw_rectangle is None:
        return
    try:
        draw_rectangle(int(left), int(top), int(width), int(height))
    except TypeError:
        draw_rectangle((int(left), int(top), int(width), int(height)))
    _draw_debug_text(img, int(pixel_x) + 4, int(pixel_y) - 10, "SELECT")


def build_object_observation(valid, center_x, bottom_y, area, image_width, image_height, config_id=Task.SEARCH):
    if int(valid) != 1:
        return 0.0, 0.0, 0.0
    target_x, target_y = build_object_target_point(image_width, image_height, config_id)
    return (
        float(center_x) - target_x,
        float(bottom_y) - target_y,
        float(area),
    )


def build_object_target_point(image_width, image_height, config_id=Task.SEARCH):
    _ = (image_width, image_height)
    target_x = float(OBJECT_APPROACH_TARGET_X_PX)
    if int(config_id) == int(Task.ORBIT):
        return float(OBJECT_ORBIT_TARGET_X_PX), float(OBJECT_ORBIT_TARGET_Y_PX)
    if int(config_id) == int(Task.TRANSPORT):
        return target_x, float(ASSISTANT_TRANSPORT_TARGET_Y_PX)
    return target_x, float(OBJECT_APPROACH_TARGET_Y_PX)


def _build_object_y_velocity(err_y, image_height):
    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_APPROACH_DEADZONE_Y_PX):
        return 0.0
    scaled_error = err_y * (
        float(OBJECT_APPROACH_MAX_VY)
        / abs(float(OBJECT_APPROACH_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_APPROACH_KP_Y) * current_frame_time_scale(),
        float(OBJECT_APPROACH_MIN_SPEED) * current_frame_time_scale(),
        float(OBJECT_APPROACH_MAX_VY) * current_frame_time_scale(),
    )


def build_object_approach_velocity_from_error(err_x, err_y, image_height):
    return (
        _axis_p_velocity(
            err_x,
            OBJECT_APPROACH_DEADZONE_X_PX,
            OBJECT_APPROACH_KP_X,
            OBJECT_APPROACH_MAX_VX,
            OBJECT_APPROACH_MIN_SPEED,
        ),
        _build_object_y_velocity(err_y, image_height),
    )


def build_object_approach_velocity_from_observation(observation, image_height):
    x, y, value = observation
    if float(value) <= 0.0:
        return float(OBJECT_MISSING_SEARCH_VX), float(OBJECT_MISSING_SEARCH_VY)
    return build_object_approach_velocity_from_error(x, y, image_height)


def _build_object_orbit_y_velocity(err_y, image_height):
    err_y = float(err_y)
    if abs(err_y) <= float(OBJECT_ORBIT_DEADZONE_Y_PX):
        return 0.0
    if float(OBJECT_ORBIT_KP_Y) == 0.0:
        return 0.0
    scaled_error = err_y * (
        float(OBJECT_ORBIT_MAX_VY)
        / abs(float(OBJECT_ORBIT_KP_Y))
        / float(image_height)
    )
    return _apply_min_speed(
        scaled_error * float(OBJECT_ORBIT_KP_Y) * current_frame_time_scale(),
        float(OBJECT_ORBIT_MIN_SPEED) * current_frame_time_scale(),
        float(OBJECT_ORBIT_MAX_VY) * current_frame_time_scale(),
    )


def build_object_orbit_velocity_from_observation(observation, image_height):
    x, y, value = observation
    if float(value) <= 0.0:
        return 0.0, 0.0
    return (
        _axis_p_velocity(
            x,
            OBJECT_ORBIT_DEADZONE_X_PX,
            OBJECT_ORBIT_KP_X,
            OBJECT_ORBIT_MAX_VX,
            OBJECT_ORBIT_MIN_SPEED,
        ),
        _build_object_orbit_y_velocity(y, image_height),
    )


class AssistantVisionState:
    def __init__(
        self,
        min_area=OBJECT_MIN_AREA,
        tolerance_x=OBJECT_X_TOLERANCE_PX,
        tolerance_y=OBJECT_Y_TOLERANCE_PX,
        stable_frames=OBJECT_STABLE_FRAMES,
        now_ms=None,
        event_resend_interval_ms=RELIABLE_RESEND_INTERVAL_MS,
    ):
        self._now_ms = now_ms or default_now_ms
        self._event_resend_interval_ms = int(event_resend_interval_ms)
        self._default_min_area = float(min_area)
        self._default_tolerance_x = float(tolerance_x)
        self._default_tolerance_y = float(tolerance_y)
        self._default_stable_frames = int(stable_frames)
        self.reset()

    def reset(self):
        self.min_area = float(self._default_min_area)
        self.tolerance_x = float(self._default_tolerance_x)
        self.tolerance_y = float(self._default_tolerance_y)
        self.required_stable_frames = int(self._default_stable_frames)
        self.mode = RunMode.FOLLOW
        self.current_sync = None
        self._last_sync_seq = None
        self._stable_count = 0
        self._pending_event = None
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = None
        self._last_return_line_y = None
        self.yolo_net = None
        self.uart_device = None
        self.rx_buffer = b""
        self.current_frame_interval_ms = 1000.0 / VISION_REFERENCE_FPS

    def handle_control_line(self, line):
        sync_packet = parse_sync_packet(line)
        if sync_packet is not None:
            return self._handle_sync_packet(sync_packet)
        ack_packet = parse_ack_packet(line)
        if ack_packet is not None:
            self._handle_ack_packet(ack_packet)
        return None

    def _handle_sync_packet(self, packet):
        reliable_seq = int(packet["reliable_seq"])
        if self._should_apply_sync(reliable_seq):
            self.current_sync = {
                "reliable_seq": reliable_seq,
                "state": int(packet["state"]),
                "target": int(packet["target"]),
                "arg": int(packet["arg"]),
            }
            self._last_sync_seq = reliable_seq
            self.mode = self._mode_from_sync(self.current_sync)
            self._stable_count = 0
            self._last_return_line_y = None
        return format_ack_frame(reliable_seq)

    def _should_apply_sync(self, reliable_seq):
        if self._last_sync_seq is None:
            return True
        if int(reliable_seq) == int(self._last_sync_seq):
            return False
        return is_newer_seq(reliable_seq, self._last_sync_seq)

    def _mode_from_sync(self, sync):
        config_id = unpack_task_arg_config(sync["arg"])
        if (
            int(sync["state"]) == State.APPROACH_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and config_id in (Task.SEARCH, Task.TRANSPORT)
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.TRANSPORT_OBJECT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.TRANSPORT
        ):
            return RunMode.APPROACH_OBJECT
        if (
            int(sync["state"]) == State.ORBIT
            and int(sync["target"]) == Target.OBJECT
            and config_id == Task.ORBIT
        ):
            return RunMode.ORBIT_OBJECT
        if (
            int(sync["state"]) == State.RETURN_FOLLOW
            and int(sync["target"]) == Target.NONE
            and config_id == Task.RETURN_GARAGE_LINE
        ):
            return RunMode.RETURN_LINE
        return RunMode.FOLLOW

    def current_object_config_id(self):
        if self.current_sync is None:
            return Task.SEARCH
        return unpack_task_arg_config(self.current_sync["arg"])

    def current_object_id(self):
        if self.current_sync is None:
            return 0
        return unpack_task_arg_object_id(self.current_sync["arg"])

    def last_return_line_y(self):
        return self._last_return_line_y

    def remember_return_line_y(self, line_y):
        if line_y is not None:
            self._last_return_line_y = float(line_y)

    def _handle_ack_packet(self, packet):
        if self._pending_event is None:
            return
        if int(packet["reliable_seq"]) == int(self._pending_event["reliable_seq"]):
            self._pending_event = None
            self._pending_event_last_sent_ms = None

    def has_pending_event(self):
        return self._pending_event is not None

    def accept_object_observation(self, observation):
        if self.mode != RunMode.APPROACH_OBJECT or self.current_sync is None:
            self._stable_count = 0
            return
        current_sync_seq = int(self.current_sync["reliable_seq"])
        if self._pending_event is not None:
            return
        if self._completed_event_sync_seq == current_sync_seq:
            return
        event_id = self._current_event_id()
        if event_id is None:
            self._stable_count = 0
            return
        x, y, value = observation
        if self._observation_matches_target(x, y, value):
            self._stable_count += 1
        else:
            self._stable_count = 0
            return
        if self._stable_count >= self.required_stable_frames:
            self._create_event(current_sync_seq, event_id, value)

    def accept_return_line_observation(self, line_y):
        _ = line_y
        self._stable_count = 0

    def _observation_matches_target(self, x, y, value):
        return (
            float(value) >= self.min_area
            and abs(float(x)) <= self.tolerance_x
            and abs(float(y)) <= self.tolerance_y
        )

    def _current_event_id(self):
        if self.current_sync is None:
            return None
        sync_state = int(self.current_sync["state"])
        target = int(self.current_sync["target"])
        config_id = unpack_task_arg_config(self.current_sync["arg"])
        if sync_state == State.APPROACH_OBJECT and target == Target.OBJECT and config_id == Task.SEARCH:
            return Event.TARGET_FOUND
        if sync_state == State.APPROACH_OBJECT and target == Target.OBJECT and config_id == Task.TRANSPORT:
            return Event.ALIGNED
        if sync_state == State.ORBIT and target == Target.OBJECT and config_id == Task.TRANSPORT:
            return Event.ALIGNED
        if sync_state == State.RETURN_FOLLOW and target == Target.NONE and config_id == Task.RETURN_GARAGE_LINE:
            return Event.RETURN_GARAGE_FINISHED
        return None

    def _create_event(self, reliable_seq, event, value):
        self._pending_event = {
            "reliable_seq": int(reliable_seq),
            "event": int(event),
            "value": int(float(value)),
        }
        self._pending_event_last_sent_ms = None
        self._completed_event_sync_seq = int(reliable_seq)

    def next_event_frame(self):
        if self._pending_event is None:
            return None
        now_ms = self._now_ms()
        if not should_resend(now_ms, self._pending_event_last_sent_ms, self._event_resend_interval_ms):
            return None
        self._pending_event_last_sent_ms = now_ms
        return format_event_frame(
            self._pending_event["reliable_seq"],
            self._pending_event["event"],
            self._pending_event["value"],
        )


state = AssistantVisionState()


def reset_runtime_state():
    state.reset()


def write_line(uart, line):
    if not isinstance(line, bytes):
        line = bytes(line)
    remaining = line
    while remaining:
        written = uart.write(remaining)
        if written is None:
            written = len(remaining)
        written = int(written)
        if written <= 0:
            return
        remaining = remaining[written:]


def init_uart():
    return UART(UART_ID, baudrate=UART_BAUDRATE)


def init_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.skip_frames(time=2000)
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False, exposure_us=EXP_TIME_US)
    return sensor.width(), sensor.height()


def _find_control_frame_start(rx_buffer):
    limit = len(rx_buffer) - FRAME_SIZE + 1
    for index in range(limit):
        frame = decode_frame(rx_buffer[index : index + FRAME_SIZE])
        if frame is None:
            continue
        mode = frame["mode"]
        topic = frame["topic"]
        if mode == Mode.TCP and topic == Topic.ASSISTANT_VISION_TASK_SYNC:
            return index
        if mode == Mode.ACK and topic == Topic.ASSISTANT_VISION_EVENT_REPORT:
            return index
    return -1


def process_uart_input(uart, rx_buffer, vision_state):
    any_fn = getattr(uart, "any", None)
    if any_fn is None:
        return rx_buffer
    size = uart.any()
    if not size:
        return rx_buffer
    try:
        data = uart.read(size)
        if data is None:
            return rx_buffer
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        elif not isinstance(data, bytes):
            return rx_buffer
        rx_buffer += data
    except Exception:
        return rx_buffer
    while len(rx_buffer) >= FRAME_SIZE:
        frame_start = _find_control_frame_start(rx_buffer)
        if frame_start < 0:
            return rx_buffer[-(FRAME_SIZE - 1) :]
        if frame_start > 0:
            rx_buffer = rx_buffer[frame_start:]
        if len(rx_buffer) < FRAME_SIZE:
            return rx_buffer
        line = rx_buffer[:FRAME_SIZE]
        rx_buffer = rx_buffer[FRAME_SIZE:]
        reply = vision_state.handle_control_line(line)
        if reply is not None:
            write_line(uart, reply)
    return rx_buffer


def process_follow_frame(uart, img, image_width, image_height):
    candidates = build_blob_candidates(img)
    if not candidates:
        follow_command = build_follow_command(valid=0, err_x=0, err_y=0)
        write_line(uart, format_vision_frame(follow_command["command_vx"], follow_command["command_vy"]))
        return
    cx_screen = float(image_width) / 2.0
    _, pixel_x, pixel_y, _, marker_span, best_blob = choose_best_candidate(
        candidates,
        cx_screen,
        image_height,
    )
    err_x = compute_lateral_error(blob_cx=pixel_x, cx_screen=cx_screen)
    err_y = compute_vertical_error(marker_span=marker_span, target_span=FOLLOW_TARGET_Y)
    follow_command = build_follow_command(valid=1, err_x=err_x, err_y=err_y)
    draw_selected_marker(img, best_blob, pixel_x, pixel_y)
    write_line(uart, format_vision_frame(follow_command["command_vx"], follow_command["command_vy"]))


def process_object_frame(uart, vision_state, img, image_width, image_height, yolo_net=None):
    candidates = build_object_candidates(img, yolo_net)
    object_id = vision_state.current_object_id()
    if object_id > 0:
        selected_task_name = object_task_name_from_id(object_id)
        if selected_task_name is not None:
            candidates = [candidate for candidate in candidates if candidate[0] == selected_task_name]
    if not candidates:
        observation = build_object_observation(0, 0, 0, 0, image_width, image_height)
    else:
        config_id = vision_state.current_object_config_id()
        target_x, target_y = build_object_target_point(image_width, image_height, config_id)
        use_target_window_filter = should_filter_candidates_by_target_window(vision_state)
        if use_target_window_filter:
            candidates = filter_candidates_in_target_window(
                candidates,
                target_x,
                target_y,
                OBJECT_X_TOLERANCE_PX,
                OBJECT_Y_TOLERANCE_PX,
            )
        if not candidates:
            observation = build_object_observation(0, 0, 0, 0, image_width, image_height)
        else:
            if use_target_window_filter:
                _, pixel_x, pixel_y, bottom_y, area, best_blob = choose_largest_area_candidate(candidates)
            else:
                _, pixel_x, pixel_y, bottom_y, area, best_blob = choose_best_candidate(
                    candidates,
                    target_x,
                    target_y,
                )
            observation = build_object_observation(
                1,
                pixel_x,
                bottom_y,
                area,
                image_width,
                image_height,
                config_id,
            )
            _draw_debug_protocol_point(img, image_width, image_height, target_x, target_y)
            draw_selected_marker(img, best_blob, pixel_x, pixel_y)
    if vision_state.mode == RunMode.ORBIT_OBJECT:
        vx, vy = build_object_orbit_velocity_from_observation(observation, image_height)
    else:
        vx, vy = build_object_approach_velocity_from_observation(observation, image_height)
    write_line(uart, format_vision_frame(vx, vy))
    vision_state.accept_object_observation(observation)


def process_return_line_frame(uart, vision_state, img, image_width, image_height):
    line_y = build_return_line_y_from_image(
        img,
        image_width,
        image_height,
        vision_state.last_return_line_y(),
    )
    vision_state.remember_return_line_y(line_y)
    vx, vy = build_return_line_velocity_from_y(line_y)
    write_line(uart, format_vision_frame(vx, vy))
    vision_state.accept_return_line_observation(line_y)


def draw_assistant_return_line_debug(img, image_width, image_height, line_y, vx, vy):
    center_x = int(float(image_width) / 2.0)
    half_width = int(RETURN_LINE_SAMPLE_HALF_WIDTH_PX)
    max_thickness = int(RETURN_LINE_MAX_THICKNESS_PX)
    target_y = int(RETURN_LINE_TARGET_Y_PX)
    bottom_y = int(image_height) - 1

    def flip_x(x):
        return int(image_width) - 1 - int(x)

    def flip_y(y):
        return int(image_height) - 1 - int(y)

    def draw_line(x0, y0, x1, y1):
        _draw_debug_line(img, flip_x(x0), flip_y(y0), flip_x(x1), flip_y(y1))

    draw_line(center_x - half_width, 0, center_x - half_width, bottom_y)
    draw_line(center_x + half_width, 0, center_x + half_width, bottom_y)
    draw_line(0, target_y, int(image_width) - 1, target_y)
    if line_y is not None:
        draw_line(0, int(line_y), int(image_width) - 1, int(line_y))
        _draw_debug_cross(img, flip_x(center_x), flip_y(int(line_y)))
        line_text = "line_y=%.1f" % float(line_y)
    else:
        line_text = "line_y=none"
    _draw_debug_text(img, 2, 2, "assistant return line debug")
    _draw_debug_text(
        img,
        2,
        14,
        "max h=%d x=%d..%d" % (max_thickness, center_x - half_width, center_x + half_width),
    )
    _draw_debug_text(img, 2, 26, "%s found=%d" % (line_text, 1 if line_y is not None else 0))
    _draw_debug_text(img, 2, 38, "vx=%.1f vy=%.1f" % (float(vx), float(vy)))


def run_assistant_return_line_debug():
    image_width, image_height = init_sensor()
    last_line_y = None
    while True:
        img = sensor.snapshot()
        apply_lens_correction(img)
        line_y = build_return_line_y_from_image(img, image_width, image_height, last_line_y)
        if line_y is not None:
            last_line_y = line_y
        vx, vy = build_return_line_velocity_from_y(line_y)
        draw_assistant_return_line_debug(img, image_width, image_height, line_y, vx, vy)


def process_frame(uart, vision_state, img, image_width, image_height, yolo_net=None):
    if vision_state.has_pending_event():
        event_frame = vision_state.next_event_frame()
        if event_frame is not None:
            write_line(uart, event_frame)
        return
    if vision_state.mode in (RunMode.APPROACH_OBJECT, RunMode.ORBIT_OBJECT):
        process_object_frame(uart, vision_state, img, image_width, image_height, yolo_net)
    elif vision_state.mode == RunMode.RETURN_LINE:
        process_return_line_frame(uart, vision_state, img, image_width, image_height)
    else:
        process_follow_frame(uart, img, image_width, image_height)


def apply_lens_correction(img):
    try:
        img.lens_corr(strength=2.8, zoom=1.0)
    except MemoryError:
        pass


def run():
    reset_runtime_state()
    state.uart_device = init_uart()
    image_width, image_height = init_sensor()
    state.yolo_net = load_yolo_model()
    last_frame_ms = default_now_ms()
    while True:
        state.rx_buffer = process_uart_input(state.uart_device, state.rx_buffer, state)
        img = sensor.snapshot()
        now_ms = default_now_ms()
        if now_ms >= last_frame_ms:
            state.current_frame_interval_ms = now_ms - last_frame_ms
        else:
            state.current_frame_interval_ms = 1000.0 / VISION_REFERENCE_FPS
        last_frame_ms = now_ms
        apply_lens_correction(img)
        process_frame(state.uart_device, state, img, image_width, image_height)
        gc.collect()


if __name__ == "__main__":
    run()
