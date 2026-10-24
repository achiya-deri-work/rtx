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
