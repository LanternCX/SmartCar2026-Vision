import json
import os
import subprocess

from calibration.chromaforge_export_adapter import (
    DEFAULT_RULES_PATH,
    build_openart_config,
    build_role_source,
    load_rules_json,
)


def _run_role_build_script_preserving_sources(script_path, target_dir, preserved_sources=None):
    root_dir = DEFAULT_RULES_PATH.parent.parent
    if preserved_sources is None:
        preserved_sources = ("master/main.py", "assistant/main.py")
    originals = {}
    for relative_path in preserved_sources:
        source_path = root_dir / relative_path
        originals[source_path] = source_path.read_text(encoding="utf-8")
    try:
        return subprocess.run(
            ["bash", script_path],
            check=True,
            cwd=root_dir,
            env={
                "PATH": os.environ["PATH"],
                "TARGET_DIR": str(target_dir),
            },
            capture_output=True,
            text=True,
        )
    finally:
        for source_path, original_text in originals.items():
            source_path.write_text(original_text, encoding="utf-8")


def _first_yellow_threshold(document):
    for item in document["objects"]:
        if str(item.get("name", "")).strip().lower() != "yellow":
            continue
        thresholds = item.get("thresholds", [])
        if not thresholds:
            return None
        return "(" + ", ".join(str(value) for value in thresholds[0]) + ")"
    return None


def test_chromaforge_export_builds_openart_threshold_config() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "recognition_merge_gap": 6,
                "min_recognition_block_area": 12,
                "min_recognition_target_area": 40,
                "max_recognition_side_length": 120,
                "thresholds": [[42, 91, -24, 6, 28, 85], [10, 20, 30, 40, 50, 60]],
                "clusters": [
                    {"id": "obj_red_cluster_1", "name": "body", "threshold_index": 0},
                    {"id": "obj_red_cluster_2", "name": "shadow", "threshold_index": 1},
                ],
            }
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="TASKS")

    assert 'TASKS = (' in config
    assert "'red'" in config
    assert "(42, 91, -24, 6, 28, 85)" in config
    assert "(10, 20, 30, 40, 50, 60)" in config
    assert "6, 12, 40, 120" in config


def test_chromaforge_export_preserves_optional_cluster_mode_in_single_task() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_marker",
                "name": "marker",
                "require_all_clusters": False,
                "recognition_merge_gap": 3,
                "min_recognition_block_area": 8,
                "min_recognition_target_area": 30,
                "max_recognition_side_length": 64,
                "thresholds": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                "clusters": [],
            }
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="FOLLOW_TASKS")

    assert config.count("'marker'") == 1
    assert "('marker', ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)), 3, 8, 30, 64, False)" in config


def test_chromaforge_export_uses_legacy_global_recognition_settings_as_fallback() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "recognition_merge_gap": 9,
        "min_recognition_block_area": 14,
        "min_recognition_target_area": 28,
        "max_recognition_side_length": 72,
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "thresholds": [[42, 91, -24, 6, 28, 85]],
                "clusters": [],
            }
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="TASKS")

    assert "('red', ((42, 91, -24, 6, 28, 85),), 9, 14, 28, 72, True)" in config


def test_chromaforge_export_builds_deployable_role_source() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "recognition_merge_gap": 4,
                "min_recognition_block_area": 11,
                "min_recognition_target_area": 33,
                "max_recognition_side_length": 96,
                "thresholds": [[1, 2, 3, 4, 5, 6]],
                "clusters": [],
            }
        ],
    }
    source = "\n".join(
        [
            "OBJECT_BLOB_MERGE_MARGIN = 0",
            "OBJECT_BLOB_PIXELS_THRESHOLD = 200",
            "OBJECT_BLOB_AREA_THRESHOLD = 200",
            "TASKS = (",
            "    ('old', (0, 0, 0, 0, 0, 0)),",
            ")",
            "def keep():",
            "    return True",
        ]
    )

    result = build_role_source(
        source, json.dumps(document), task_constant_name="TASKS"
    )

    assert "OBJECT_BLOB_MERGE_MARGIN = 0" in result
    assert "OBJECT_BLOB_PIXELS_THRESHOLD = 200" in result
    assert "OBJECT_BLOB_AREA_THRESHOLD = 200" in result
    assert "('red', ((1, 2, 3, 4, 5, 6),), 4, 11, 33, 96, True)" in result
    assert "'old'" not in result
    assert "def keep():" in result


def test_chromaforge_export_preserves_other_task_tables() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_marker",
                "name": "marker",
                "require_all_clusters": False,
                "recognition_merge_gap": 4,
                "min_recognition_block_area": 11,
                "min_recognition_target_area": 33,
                "max_recognition_side_length": 48,
                "thresholds": [[1, 2, 3, 4, 5, 6]],
                "clusters": [],
            }
        ],
    }
    source = "\n".join(
        [
            "OBJECT_BLOB_MERGE_MARGIN = 0",
            "OBJECT_BLOB_PIXELS_THRESHOLD = 200",
            "OBJECT_BLOB_AREA_THRESHOLD = 200",
            "FOLLOW_TASKS = (('old_follow', (0, 0, 0, 0, 0, 0)),)",
            "OBJECT_TASKS = (",
            "    ('old_object', (9, 9, 9, 9, 9, 9)),",
            ")",
        ]
    )

    result = build_role_source(
        source, json.dumps(document), task_constant_name="OBJECT_TASKS"
    )

    assert "FOLLOW_TASKS = (('old_follow', (0, 0, 0, 0, 0, 0)),)" in result
    assert "'old_object'" not in result
    assert "('marker', ((1, 2, 3, 4, 5, 6),), 4, 11, 33, 48, False)" in result


def test_chromaforge_export_excludes_yellow_from_task_entries() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "thresholds": [[1, 2, 3, 4, 5, 6]],
                "clusters": [],
            },
            {
                "id": "obj_yellow",
                "name": "Yellow",
                "require_all_clusters": True,
                "thresholds": [[7, 8, 9, 10, 11, 12]],
                "clusters": [],
            },
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="TASKS")

    assert "'red'" in config
    assert "'Yellow'" not in config
    assert "(7, 8, 9, 10, 11, 12)" not in config


def test_chromaforge_export_binds_yellow_threshold_to_role_source() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "thresholds": [[1, 2, 3, 4, 5, 6]],
                "clusters": [],
            },
            {
                "id": "obj_yellow",
                "name": "yellow",
                "require_all_clusters": True,
                "thresholds": [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
                "clusters": [],
            },
        ],
    }
    source = "\n".join(
        [
            "TASKS = (",
            "    ('old', ((0, 0, 0, 0, 0, 0),), 0, 1, 1, 0, True),",
            ")",
            "FINISH_HOOK_YELLOW_THRESHOLD = (0, 0, 0, 0, 0, 0)",
            "RETURN_GARAGE_LINE_YELLOW_THRESHOLD = (1, 1, 1, 1, 1, 1)",
            "RETURN_LINE_YELLOW_THRESHOLD = (2, 2, 2, 2, 2, 2)",
        ]
    )

    result = build_role_source(source, json.dumps(document), task_constant_name="TASKS")

    assert "('red', ((1, 2, 3, 4, 5, 6),), 0, 1, 1, 0, True)" in result
    assert "'yellow'" not in result
    assert "FINISH_HOOK_YELLOW_THRESHOLD = (7, 8, 9, 10, 11, 12)" in result
    assert "RETURN_GARAGE_LINE_YELLOW_THRESHOLD = (7, 8, 9, 10, 11, 12)" in result
    assert "RETURN_LINE_YELLOW_THRESHOLD = (7, 8, 9, 10, 11, 12)" in result


def test_chromaforge_export_loads_default_rules_from_tool_directory() -> None:
    rules = json.loads(load_rules_json())

    assert DEFAULT_RULES_PATH.name == "chromaforge-rules.json"
    assert rules["format"] == "chromaforge-v1"
    assert rules["target"] == "openmv-find-blobs"


def test_chromaforge_export_cli_uses_default_rules_file() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "calibration.chromaforge_export_adapter",
            "--task-constant-name",
            "TASKS",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TASKS = (" in result.stdout
    assert "obj_1" not in result.stdout


def test_role_build_script_generates_master_output_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "master-device"
    target_dir.mkdir()

    _run_role_build_script_preserving_sources("master/build.sh", target_dir)

    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
    rules = json.loads(load_rules_json())
    yellow_threshold = _first_yellow_threshold(rules)

    assert "('red'" in uploaded
    assert "'yellow'" not in uploaded
    if yellow_threshold is not None:
        assert "FINISH_HOOK_YELLOW_THRESHOLD = %s" % yellow_threshold in uploaded
        assert "RETURN_GARAGE_LINE_YELLOW_THRESHOLD = %s" % yellow_threshold in uploaded
    assert "threshold_index" not in uploaded


def test_role_build_v2_script_generates_master_v2_output_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "master-v2-device"
    target_dir.mkdir()

    _run_role_build_script_preserving_sources(
        "master/build_v2.sh",
        target_dir,
        preserved_sources=("master/main_v2.py",),
    )

    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
    rules = json.loads(load_rules_json())
    yellow_threshold = _first_yellow_threshold(rules)

    assert "OBJECT_TASKS = (" in uploaded
    assert "('red'" in uploaded
    assert "'yellow'" not in uploaded
    if yellow_threshold is not None:
        assert "FINISH_HOOK_YELLOW_THRESHOLD = %s" % yellow_threshold in uploaded
        assert "RETURN_GARAGE_LINE_YELLOW_THRESHOLD" not in uploaded
    assert "threshold_index" not in uploaded


def test_role_build_v2_script_rewrites_master_v2_source_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "master-v2-copy-only-device"
    target_dir.mkdir()
    root_dir = DEFAULT_RULES_PATH.parent.parent
    source_path = root_dir / "master" / "main_v2.py"
    original_text = source_path.read_text(encoding="utf-8")
    modified_text = original_text.replace(
        "OBJECT_TASKS = (",
        "OBJECT_TASKS = (\n    ('runtime_only_marker', 1, 2, 3, 4, False),",
        1,
    )
    source_path.write_text(modified_text, encoding="utf-8")
    try:
        subprocess.run(
            ["bash", "master/build_v2.sh"],
            check=True,
            cwd=root_dir,
            env={
                "PATH": os.environ["PATH"],
                "TARGET_DIR": str(target_dir),
            },
            capture_output=True,
            text=True,
        )
        uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
        generated_source = source_path.read_text(encoding="utf-8")
        assert generated_source != modified_text
        assert uploaded == generated_source
        assert "runtime_only_marker" not in generated_source
        assert "threshold_index" not in generated_source
    finally:
        source_path.write_text(original_text, encoding="utf-8")


def test_role_build_script_generates_assistant_output_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "assistant-device"
    target_dir.mkdir()

    _run_role_build_script_preserving_sources("assistant/build.sh", target_dir)

    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
    rules = json.loads(load_rules_json())
    yellow_threshold = _first_yellow_threshold(rules)

    assert "OBJECT_TASKS = (" in uploaded
    assert "('red'" in uploaded
    assert "'yellow'" not in uploaded
    if yellow_threshold is not None:
        assert "RETURN_LINE_YELLOW_THRESHOLD = %s" % yellow_threshold in uploaded


def test_role_build_v2_script_generates_assistant_v2_output_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "assistant-v2-device"
    target_dir.mkdir()

    _run_role_build_script_preserving_sources(
        "assistant/build_v2.sh",
        target_dir,
        preserved_sources=("assistant/main_v2.py",),
    )

    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
    rules = json.loads(load_rules_json())
    yellow_threshold = _first_yellow_threshold(rules)

    assert "OBJECT_TASKS = (" in uploaded
    assert "('red'" in uploaded
    assert "'yellow'" not in uploaded
    if yellow_threshold is not None:
        assert "RETURN_LINE_YELLOW_THRESHOLD = %s" % yellow_threshold in uploaded
    assert "threshold_index" not in uploaded


def test_role_build_v2_script_rewrites_assistant_v2_source_from_shared_rules(tmp_path) -> None:
    target_dir = tmp_path / "assistant-v2-copy-only-device"
    target_dir.mkdir()
    root_dir = DEFAULT_RULES_PATH.parent.parent
    source_path = root_dir / "assistant" / "main_v2.py"
    original_text = source_path.read_text(encoding="utf-8")
    modified_text = original_text.replace(
        "OBJECT_TASKS = (",
        "OBJECT_TASKS = (\n    ('runtime_only_marker', 1, 2, 3, 4, False),",
        1,
    )
    source_path.write_text(modified_text, encoding="utf-8")
    try:
        subprocess.run(
            ["bash", "assistant/build_v2.sh"],
            check=True,
            cwd=root_dir,
            env={
                "PATH": os.environ["PATH"],
                "TARGET_DIR": str(target_dir),
            },
            capture_output=True,
            text=True,
        )
        uploaded = (target_dir / "main.py").read_text(encoding="utf-8")
        generated_source = source_path.read_text(encoding="utf-8")
        assert generated_source != modified_text
        assert uploaded == generated_source
        assert "runtime_only_marker" not in generated_source
        assert "threshold_index" not in generated_source
    finally:
        source_path.write_text(original_text, encoding="utf-8")
