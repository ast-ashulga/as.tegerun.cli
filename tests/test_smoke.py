"""Cheap guards for bugs no unit test happens to touch: a file that does not compile or import
(e.g. the model workers run in their own venvs and are never imported by tests), or an undefined name."""

import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tegerun

ROOT = Path(__file__).resolve().parent.parent
PY_FILES = sorted(
    p for d in ("src", "classifier", "tests") for p in (ROOT / d).rglob("*.py") if ".venv" not in str(p)
)


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_python_file_compiles(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("name", [m.name for m in pkgutil.iter_modules(tegerun.__path__)])
def test_every_package_module_imports(name):
    importlib.import_module(f"tegerun.{name}")


def test_the_worker_dirs_are_covered_by_the_compile_check():
    names = {p.name for p in PY_FILES}
    assert {"d1_worker.py", "pplx_worker.py", "fetch.py", "reader.py"} <= names


@pytest.mark.skipif(shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff").exists(), reason="ruff not installed")
def test_ruff_finds_no_undefined_names_or_unused_imports():
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    done = subprocess.run([ruff, "check", "--no-cache", str(ROOT / "src"), str(ROOT / "classifier"), str(ROOT / "tests")],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
