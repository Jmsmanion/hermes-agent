"""Pre-spawn policy hook prevents a worker process from starting on failed preflight."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def pre_spawn_hook():
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    calls = []
    ready = {"value": False}

    def guard(**payload):
        calls.append(payload)
        if not ready["value"]:
            return {"action": "block", "reason": "dispatcher context envelope missing"}
        return None

    manager._hooks.setdefault("kanban_task_pre_spawn", []).append(guard)
    try:
        yield calls, ready
    finally:
        manager._hooks = saved


def test_missing_context_blocks_before_spawn_and_retries_only_after_prerequisite_changes(
    kanban_home, all_assignees_spawnable, pre_spawn_hook,
):
    assert "kanban_task_pre_spawn" in VALID_HOOKS
    calls, ready = pre_spawn_hook
    spawn_calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="pre-spawn context gate", assignee="hybridworker")

        first = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 123)
        task = kb.get_task(conn, task_id)
        runs = kb.list_runs(conn, task_id)
        assert first.spawned == []
        assert first.pre_spawn_blocked == [(task_id, "dispatcher context envelope missing")]
        assert spawn_calls == []  # no model worker process was created
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert runs[-1].status == "blocked"
        assert calls[-1]["run_id"] == runs[-1].id
        assert calls[-1]["workspace_path"]

        # Another unchanged dispatch tick leaves the blocked task parked.
        second = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 124)
        assert second.spawned == []
        assert second.pre_spawn_blocked == []
        assert spawn_calls == []

        # A changed precondition plus supported unblock gets a new run and an allow result.
        ready["value"] = True
        assert kb.unblock_task(conn, task_id) is True
        third = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 125)
        assert [row[0] for row in third.spawned] == [task_id]
        assert len(spawn_calls) == 1
        assert calls[-1]["run_id"] != calls[0]["run_id"]
        assert calls[-1]["workspace_path"] == calls[0]["workspace_path"]

