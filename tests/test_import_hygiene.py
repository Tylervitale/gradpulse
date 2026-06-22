"""Guard: tests must import the package as ``gradpulse...``, never bare local modules.

A bare ``import validate`` / ``from multiqubit import X`` only resolves when the repo
directory happens to be on ``sys.path`` (e.g. ``python -m pytest`` run from the repo
root), so it passes locally but fails under plain ``pytest`` / CI / importlib mode.
This regressed three times historically. The test below fails fast with a clear,
located message if the anti-pattern reappears, independent of how pytest is invoked.
See ``conftest.py`` and the import-mode note in ``pyproject.toml``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent

# Top-level modules of the package (under src/gradpulse/) -- their bare names are
# importable ONLY by accident of CWD; tests must reach them via
# ``from gradpulse import ...`` / ``from gradpulse.x import ...``.
_PKG_DIR = _REPO_ROOT / "src" / "gradpulse"
_LOCAL_MODULES = {
    p.stem for p in _PKG_DIR.glob("*.py")
    if p.stem not in {"__init__", "__main__", "conftest", "setup"}
}


@pytest.mark.parametrize(
    "path", sorted(_TESTS_DIR.glob("test_*.py")), ids=lambda p: p.name)
def test_no_bare_local_module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in _LOCAL_MODULES:
                offenders.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _LOCAL_MODULES:
                    offenders.append((node.lineno, f"import {alias.name}"))
    assert not offenders, (
        f"{path.name} imports local modules bare -- use `from gradpulse import ...`:\n"
        + "\n".join(f"  line {ln}: {src}" for ln, src in offenders))
