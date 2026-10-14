"""No workflow may be startable by an outsider.

Only triggers that need write access to the repo are allowed. That rules out pull_request,
pull_request_target, issue_comment, issues, fork, watch and the rest: a stranger's fork PR
must run nothing at all.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"push", "workflow_dispatch", "workflow_call"}
FILES = sorted(ROOT.glob(".github/workflows/*.y*ml"))


def _triggers(doc: dict) -> set[str]:
    on = doc.get("on", doc.get(True))  # PyYAML (YAML 1.1) parses a bare `on` key as True
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    return set(on or {})


def test_workflows_found():
    assert FILES, "no workflow files found"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_only_write_access_triggers(path):
    triggers = _triggers(yaml.safe_load(path.read_text()))
    assert triggers, f"{path.name} has no triggers"
    assert triggers <= ALLOWED, f"{path.name} has outsider-reachable triggers: {triggers - ALLOWED}"
