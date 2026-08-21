from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


def test_package_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import reactionflow; print(reactionflow.__version__)"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


def test_core_imports_only_declared_dependencies_or_standard_library() -> None:
    allowed_external_roots = {"ase", "networkx", "numpy", "reactionflow"}
    source_root = Path(__file__).parents[1] / "src" / "reactionflow"
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        allowed_for_path = set(allowed_external_roots)
        if path.is_relative_to(source_root / "adapters"):
            allowed_for_path.update({"torch", "torchani"})
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = set() if node.level else {node.module.partition(".")[0]}
            else:
                continue

            unexpected = roots - sys.stdlib_module_names - allowed_for_path
            if unexpected:
                violations.append(f"{path.relative_to(source_root)}: {sorted(unexpected)}")

    assert not violations, "Core dependency violations:\n" + "\n".join(violations)


def test_base_dependencies_match_architecture_boundary() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    names = {
        dependency.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dependency in dependencies
    }

    assert names == {"ase", "networkx", "numpy"}
