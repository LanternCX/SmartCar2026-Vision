import json
import os
import subprocess

from calibration.chromaforge_export_adapter import (
    DEFAULT_RULES_PATH,
    build_openart_config,
    build_role_source,
    load_rules_json,
)


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
    output_dir = tmp_path / "master-build"
    target_dir = tmp_path / "master-device"
    target_dir.mkdir()

    subprocess.run(
        ["bash", "master/build.sh"],
        check=True,
        cwd=DEFAULT_RULES_PATH.parent.parent,
        env={
            "PATH": os.environ["PATH"],
            "OUTPUT_DIR": str(output_dir),
            "TARGET_DIR": str(target_dir),
        },
        capture_output=True,
        text=True,
    )

    result = (output_dir / "main.py").read_text(encoding="utf-8")
    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")

    assert "('red'" in result
    assert "threshold_index" not in result
    assert result == uploaded


def test_role_build_script_generates_assistant_output_from_shared_rules(tmp_path) -> None:
    output_dir = tmp_path / "assistant-build"
    target_dir = tmp_path / "assistant-device"
    target_dir.mkdir()

    subprocess.run(
        ["bash", "assistant/build.sh"],
        check=True,
        cwd=DEFAULT_RULES_PATH.parent.parent,
        env={
            "PATH": os.environ["PATH"],
            "OUTPUT_DIR": str(output_dir),
            "TARGET_DIR": str(target_dir),
        },
        capture_output=True,
        text=True,
    )

    result = (output_dir / "main.py").read_text(encoding="utf-8")
    uploaded = (target_dir / "main.py").read_text(encoding="utf-8")

    assert "OBJECT_TASKS = (" in result
    assert "('red'" in result
    assert result == uploaded
