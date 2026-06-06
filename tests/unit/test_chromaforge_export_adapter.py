import json
import subprocess

from tools.chromaforge_export_adapter import (
    DEFAULT_RULES_PATH,
    build_openart_config,
    build_role_source,
    load_rules_json,
)


def test_chromaforge_export_builds_openart_threshold_config() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "recognition_merge_gap": 6,
        "min_recognition_block_area": 12,
        "min_recognition_target_area": 40,
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
                "thresholds": [[42, 91, -24, 6, 28, 85], [10, 20, 30, 40, 50, 60]],
                "clusters": [
                    {"id": "obj_red_cluster_1", "name": "body", "threshold_index": 0},
                    {"id": "obj_red_cluster_2", "name": "shadow", "threshold_index": 1},
                ],
            }
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="TASKS")

    assert "OBJECT_BLOB_MERGE_MARGIN = 6" in config
    assert "OBJECT_BLOB_PIXELS_THRESHOLD = 12" in config
    assert "OBJECT_BLOB_AREA_THRESHOLD = 40" in config
    assert 'TASKS = (' in config
    assert "'red'" in config
    assert "(42, 91, -24, 6, 28, 85)" in config
    assert "(10, 20, 30, 40, 50, 60)" in config


def test_chromaforge_export_splits_optional_clusters_as_alternative_tasks() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "recognition_merge_gap": 3,
        "min_recognition_block_area": 8,
        "min_recognition_target_area": 30,
        "objects": [
            {
                "id": "obj_marker",
                "name": "marker",
                "require_all_clusters": False,
                "thresholds": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                "clusters": [],
            }
        ],
    }

    config = build_openart_config(json.dumps(document), task_constant_name="FOLLOW_TASKS")

    assert config.count("'marker'") == 2
    assert "('marker', (1, 2, 3, 4, 5, 6))" in config
    assert "('marker', (7, 8, 9, 10, 11, 12))" in config


def test_chromaforge_export_builds_deployable_role_source() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "recognition_merge_gap": 4,
        "min_recognition_block_area": 11,
        "min_recognition_target_area": 33,
        "objects": [
            {
                "id": "obj_red",
                "name": "red",
                "require_all_clusters": True,
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

    assert "OBJECT_BLOB_MERGE_MARGIN = 4" in result
    assert "OBJECT_BLOB_PIXELS_THRESHOLD = 11" in result
    assert "OBJECT_BLOB_AREA_THRESHOLD = 33" in result
    assert "('red', ((1, 2, 3, 4, 5, 6),))" in result
    assert "'old'" not in result
    assert "def keep():" in result


def test_chromaforge_export_preserves_other_task_tables() -> None:
    document = {
        "format": "chromaforge-v1",
        "target": "openmv-find-blobs",
        "recognition_merge_gap": 4,
        "min_recognition_block_area": 11,
        "min_recognition_target_area": 33,
        "objects": [
            {
                "id": "obj_marker",
                "name": "marker",
                "require_all_clusters": False,
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
    assert "('marker', (1, 2, 3, 4, 5, 6))" in result


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
            "tools.chromaforge_export_adapter",
            "--task-constant-name",
            "TASKS",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TASKS = (" in result.stdout
    assert "OBJECT_BLOB_PIXELS_THRESHOLD" in result.stdout
