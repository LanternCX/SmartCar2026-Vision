"""辅车视觉 main_v2 行为测试."""

import pytest

from tests.test_support import load_role_entry_module
import tests.unit.test_assistant_object_approach as legacy_tests


def load_assistant_v2():
    """加载辅车视觉 main_v2 入口模块."""

    module = load_role_entry_module("assistant", "main_v2.py", "assistant_main_v2_test_module")
    module.reset_runtime_state()
    module.state.yolo_net = "fake-yolo-net"
    return module


legacy_tests.load_assistant = load_assistant_v2

for _name in dir(legacy_tests):
    if not _name.startswith("test_"):
        continue
    if _name == "test_assistant_object_candidates_use_yolo_when_flag_enabled":
        continue
    globals()[_name] = getattr(legacy_tests, _name)


def test_assistant_main_v2_object_candidates_use_yolo_by_default() -> None:
    """辅车 main_v2 默认使用 YOLO 生成找物体候选."""

    module = load_assistant_v2()

    class FakeYoloTf:
        def __init__(self):
            self.loaded_paths = []
            self.detect_calls = []

        def load(self, path):
            self.loaded_paths.append(path)
            return "fake-yolo-net"

        def detect(self, net, img):
            self.detect_calls.append((net, img))
            return [(0.25, 0.125, 0.75, 0.2083333333, 1, 0.95)]

    class FakeImage:
        def __init__(self):
            self.copy_calls = []

        def width(self):
            return legacy_tests.IMAGE_WIDTH

        def height(self):
            return legacy_tests.IMAGE_HEIGHT

        def copy(self, scale, copy_to_fb):
            self.copy_calls.append((scale, copy_to_fb))
            return "detect-image"

        def find_blobs(self, thresholds, pixels_threshold, area_threshold, merge, margin=0):
            _ = (thresholds, pixels_threshold, area_threshold, merge, margin)
            raise AssertionError("main_v2 找物体主线不应回退到色块识别")

    module.tf = FakeYoloTf()
    module.state.yolo_net = None
    img = FakeImage()

    candidates = module.build_object_candidates(img)

    assert module.tf.loaded_paths == [module.YOLO_MODEL_PATH]
    assert module.tf.detect_calls == [("fake-yolo-net", "detect-image")]
    assert img.copy_calls == [(module.YOLO_IMAGE_COPY_SCALE, 1)]
    assert candidates[0][0] == "red"
    assert candidates[0][1] == pytest.approx(160.0)
    assert candidates[0][3] == pytest.approx(210.0)
    assert candidates[0][4] == pytest.approx(3200.0)
