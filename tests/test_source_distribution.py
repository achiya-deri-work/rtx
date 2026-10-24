"""Public source archives must not include ignored application handoffs."""
from pathlib import Path


def test_manifest_keeps_private_notes_out_and_dev_helpers_in():
    text=(Path(__file__).resolve().parents[1]/"MANIFEST.in").read_text()
    for directive in (
        "exclude docs/afrl_*_202609*.md",
        "exclude docs/mlp_*_202609*.md",
        "exclude docs/mlp_deployment_goal_status.md",
        "recursive-include tools *.py",
        "recursive-include benchmarks/configs *.json",
        "recursive-include docs/examples *.py",
    ):
        assert directive in text


def test_release_runner_qualifies_its_own_checkout():
    root = Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/run_release_qualification.sh").read_text()
    assert 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert script.index("export PYTHONPATH") < script.index("validate_runtime_environment")
    assert "pillow>=10" in (root / "requirements-dev.txt").read_text()
    import tomllib
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert "pillow>=10" in metadata["project"]["optional-dependencies"]["dev"]
