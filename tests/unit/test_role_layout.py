"""! @brief 主辅视觉入口目录布局测试"""

# pyright: reportAttributeAccessIssue=false

import os
import subprocess

from tests.test_support import ROOT, load_role_main_module, role_main_path


def test_role_main_files_are_maintained_under_role_directories() -> None:
    """! @brief 主车和辅车视觉入口必须分别放在角色目录中"""

    assert role_main_path("assistant").is_file()
    assert role_main_path("master").is_file()
    for role in ("assistant", "master"):
        assert (ROOT / role / "main.py").is_file()
        assert (ROOT / role / "run.py").is_file()
        assert not any(path.stem.endswith("_" + "v" + "2") for path in (ROOT / role).glob("main*.py"))
        assert not any(path.stem.endswith("_" + "v" + "2") for path in (ROOT / role).glob("build*.sh"))
    assert not (ROOT / "main.py").exists()


def test_role_boot_entries_import_and_start_run_module() -> None:
    for role in ("assistant", "master"):
        source = (ROOT / role / "main.py").read_text(encoding="utf-8")
        assert "gc.collect()" in source
        assert "import run" in source
        assert "run.run()" in source


def test_role_main_modules_expose_role_protocol_api() -> None:
    """! @brief 主辅视觉入口分别暴露本角色的协议格式化入口"""

    assistant = load_role_main_module("assistant", "assistant_role_main_module")
    master = load_role_main_module("master", "master_role_main_module")

    assistant_frame = assistant.decode_frame(assistant.format_search_velocity_frame(0, 0))
    master_frame = master.decode_frame(master.format_search_velocity_frame(0, 0))

    assert assistant_frame is not None
    assert assistant_frame["mode"] == assistant.Mode.UDP
    assert assistant_frame["topic"] == assistant.Topic.LOCAL_VISION_VELOCITY
    assert master_frame is not None
    assert master_frame["mode"] == master.Mode.UDP
    assert master_frame["topic"] == master.Topic.LOCAL_VISION_VELOCITY
    assert not hasattr(master, "format_observation_frame")


def test_role_main_files_avoid_board_unstable_int_byte_helpers() -> None:
    """! @brief 角色入口不得保留板端不稳定的整数打包辅助写法"""

    for role in ("assistant", "master"):
        source = role_main_path(role).read_text(encoding="utf-8")
        assert ".to_bytes(" not in source
        assert "int.from_bytes(" not in source


def test_role_build_scripts_upload_role_entry_without_modifying_source(tmp_path) -> None:
    """! @brief 每个角色构建脚本直接上传源码入口且不改写源文件"""

    master_source = role_main_path("master")
    assistant_source = role_main_path("assistant")
    original_master = master_source.read_text(encoding="utf-8")
    original_assistant = assistant_source.read_text(encoding="utf-8")
    for role in ("assistant", "master"):
        try:
            built = role_main_path(role)
            original_text = original_assistant if role == "assistant" else original_master
            expected_text = original_text.replace(
                "OBJECT_TASKS = (",
                "OBJECT_TASKS = (\n    ('build_marker', 1, 2, 3, 4, False),",
                1,
            )
            built.write_text(expected_text, encoding="utf-8")
            target_dir = tmp_path / (role + "-device")
            target_dir.mkdir()
            script_path = ROOT / role / "build.sh"
            env = dict(os.environ)
            env["TARGET_DIR"] = str(target_dir)

            subprocess.run(
                ["bash", str(script_path)],
                cwd=str(ROOT),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            uploaded = target_dir / "run.py"
            assert built.is_file()
            assert (target_dir / "main.py").is_file()
            assert uploaded.is_file()
            built_text = built.read_text(encoding="utf-8")
            uploaded_text = uploaded.read_text(encoding="utf-8")
            assert built_text == expected_text
            assert built_text == uploaded_text
            assert "build_marker" in uploaded_text
            assert "threshold_index" not in built_text
            assert "OBJECT_TASKS = (" in built_text
        finally:
            master_source.write_text(original_master, encoding="utf-8")
            assistant_source.write_text(original_assistant, encoding="utf-8")


def test_role_build_scripts_copy_yolo_model_when_requested(tmp_path) -> None:
    """! @brief 角色构建脚本传入 yolo 参数时同时复制模型文件"""

    model_path = ROOT / "yolo" / "yolo.tflite"
    original_model = model_path.read_bytes() if model_path.exists() else None
    model_path.write_bytes(b"fake-yolo-model")

    try:
        for role in ("assistant", "master"):
            target_dir = tmp_path / (role + "-yolo-device")
            target_dir.mkdir()
            script_path = ROOT / role / "build.sh"
            env = dict(os.environ)
            env["TARGET_DIR"] = str(target_dir)

            subprocess.run(
                ["bash", str(script_path), "yolo"],
                cwd=str(ROOT),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            assert (target_dir / "main.py").is_file()
            assert (target_dir / "run.py").is_file()
            assert (target_dir / "yolo.tflite").read_bytes() == model_path.read_bytes()
    finally:
        if original_model is None:
            model_path.unlink()
        else:
            model_path.write_bytes(original_model)
