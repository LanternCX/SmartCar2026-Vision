"""主辅视觉入口目录布局测试."""

import os
import subprocess

from tests.test_support import ROOT, load_role_main_module, role_main_path


def test_role_main_files_are_maintained_under_role_directories() -> None:
    """主车和辅车视觉入口必须分别放在角色目录中."""

    assert role_main_path("assistant").is_file()
    assert role_main_path("master").is_file()
    assert not (ROOT / "main.py").exists()


def test_role_main_modules_keep_short_velocity_frame_api() -> None:
    """主车和辅车视觉入口都必须输出短速度包."""

    assistant = load_role_main_module("assistant", "assistant_role_main_module")
    master = load_role_main_module("master", "master_role_main_module")

    assert assistant.format_vision_frame(0, 0) == "v,0,0"
    assert master.format_vision_frame(0, 0) == "v,0,0"


def test_role_build_scripts_copy_local_main_to_device_entry(tmp_path) -> None:
    """每个角色构建脚本只复制本角色目录下的 main.py."""

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
