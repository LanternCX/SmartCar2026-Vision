"""! @brief 主辅视觉入口目录布局测试"""

import os
import subprocess

from tests.test_support import ROOT, load_role_main_module, role_main_path


def test_role_main_files_are_maintained_under_role_directories() -> None:
    """! @brief 主车和辅车视觉入口必须分别放在角色目录中"""

    assert role_main_path("assistant").is_file()
    assert role_main_path("master").is_file()
    assert not (ROOT / "main.py").exists()


def test_role_main_modules_expose_role_protocol_api() -> None:
    """! @brief 主辅视觉入口分别暴露本角色的协议格式化入口"""

    assistant = load_role_main_module("assistant", "assistant_role_main_module")
    master = load_role_main_module("master", "master_role_main_module")

    assistant_frame = assistant.decode_frame(assistant.format_vision_frame(0, 0))
    master_frame = master.decode_frame(master.format_search_velocity_frame(0, 0))

    assert assistant_frame is not None
    assert assistant_frame["mode"] == assistant.MODE_UDP
    assert assistant_frame["topic"] == assistant.TOPIC_LOCAL_VISION_VELOCITY
    assert master_frame is not None
    assert master_frame["mode"] == master.MODE_UDP
    assert master_frame["topic"] == master.TOPIC_LOCAL_VISION_VELOCITY
    assert not hasattr(master, "format_observation_frame")


def test_role_main_files_avoid_board_unstable_int_byte_helpers() -> None:
    """! @brief 角色入口不得保留板端不稳定的整数打包辅助写法"""

    for role in ("assistant", "master"):
        source = role_main_path(role).read_text(encoding="utf-8")
        assert ".to_bytes(" not in source
        assert "int.from_bytes(" not in source


def test_role_build_scripts_copy_local_main_to_device_entry(tmp_path) -> None:
    """! @brief 每个角色构建脚本只复制本角色目录下的 main.py"""

    for role in ("assistant", "master"):
        target_dir = tmp_path / role
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

        copied = target_dir / "main.py"
        assert copied.read_text(encoding="utf-8") == role_main_path(role).read_text(
            encoding="utf-8"
        )


def test_root_build_script_generates_both_role_entries(tmp_path) -> None:
    """! @brief 根构建脚本生成主车和辅车两个角色入口"""

    env = dict(os.environ)
    env["BUILD_DIR"] = str(tmp_path)

    subprocess.run(
        ["bash", "build.sh"],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    master_output = tmp_path / "master" / "main.py"
    assistant_output = tmp_path / "assistant" / "main.py"
    assert master_output.is_file()
    assert assistant_output.is_file()
    assert "TASKS = (" in master_output.read_text(encoding="utf-8")
    assert "OBJECT_TASKS = (" in assistant_output.read_text(encoding="utf-8")
