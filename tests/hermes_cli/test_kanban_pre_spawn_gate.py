"""Pre-spawn policy hook prevents a worker process from starting on failed preflight."""
from __future__ import annotations

import time
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


def test_pre_spawn_callback_exception_fails_closed_before_spawn(
    kanban_home, all_assignees_spawnable, pre_spawn_hook,
):
    manager = get_plugin_manager()

    def broken_guard(**_payload):
        raise RuntimeError("context database unavailable")

    manager._hooks["kanban_task_pre_spawn"].append(broken_guard)
    spawn_calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="raising pre-spawn guard", assignee="hybridworker")
        result = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 123)
        task = kb.get_task(conn, task_id)
        assert result.spawned == []
        assert any("raised RuntimeError" in reason for _, reason in result.pre_spawn_blocked)
        assert spawn_calls == []
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_pre_spawn_callback_timeout_fails_closed_before_spawn(
    kanban_home, all_assignees_spawnable, pre_spawn_hook, monkeypatch,
):
    manager = get_plugin_manager()
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)

    def slow_guard(**_payload):
        time.sleep(0.1)

    manager._hooks["kanban_task_pre_spawn"].append(slow_guard)
    spawn_calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="timed out pre-spawn guard", assignee="hybridworker")
        result = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 123)
        task = kb.get_task(conn, task_id)
        assert result.spawned == []
        assert any("kanban_task_pre_spawn plugin callback timed out" in reason
                   for _, reason in result.pre_spawn_blocked)
        assert spawn_calls == []
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_state_change_race_never_spawns_and_later_failure_is_parked(
    kanban_home, all_assignees_spawnable, pre_spawn_hook, monkeypatch,
):
    original_block = kb.block_task
    race_once = {"value": True}

    def race_block(conn, task_id, **kwargs):
        if race_once["value"]:
            race_once["value"] = False
            # Model another supported actor blocking and then explicitly reopening the run
            # while this dispatcher is handling its stale preflight result.
            assert original_block(conn, task_id, reason="parallel preflight failure",
                                  kind="needs_input", expected_run_id=kwargs["expected_run_id"])
            assert kb.unblock_task(conn, task_id)
        return original_block(conn, task_id, **kwargs)

    monkeypatch.setattr(kb, "block_task", race_block)
    spawn_calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="pre-spawn state race", assignee="hybridworker")
        first = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 123)
        task = kb.get_task(conn, task_id)
        assert first.spawned == []
        assert "claimed run changed" in first.pre_spawn_blocked[0][1]
        assert spawn_calls == []
        assert task.status == "ready"  # the concurrent supported unblock is preserved

        second = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawn_calls.append(args) or 124)
        task = kb.get_task(conn, task_id)
        assert second.spawned == []
        assert second.pre_spawn_blocked
        assert spawn_calls == []
        # Repeating the same typed failure after the simulated concurrent unblock escalates
        # through the existing loop breaker instead of permitting another dispatch.
        assert task.status == "triage"
        assert task.block_kind == "needs_input"
