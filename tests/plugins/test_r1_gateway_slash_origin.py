"""R1 Gen-1-laundering boundary tests: gateway-direct ``/kanban create``.

Sibling to ``test_r1_worker_depriv.py``. Those tests prove the
``created_by == "gateway-voice-background"`` voice path de-privileges its
worker. This file closes the OTHER Gen-1 origin: a ``/kanban create`` (or
``/kanban dispatch``) issued from an untrusted chat against the gateway,
routed through ``gateway/slash_commands._handle_kanban_command`` →
``hermes_cli.kanban.run_slash`` → the CLI ``create`` handler →
``kanban_db.create_task``.

That CLI path runs in the gateway MAIN process and stamps
``created_by=<profile>`` (NOT the voice marker), and the gateway main
process does not carry ``HERMES_UNTRUSTED_ORIGIN`` (only ``_default_spawn``
sets that, on a child worker env). Without a guard the card would be
created ``untrusted_origin=0`` → ``_default_spawn`` would see
``is_untrusted=False`` → launch the worker FULL-privilege (no EXEC_ASK,
inherited yolo, full toolset incl. terminal/execute_code/delegate). That is
Gen-1 laundering: the same content the ``/background`` sibling path
de-privileges would slip through privileged via ``/kanban``.

The fix is at the single worker-reachable inserter chokepoint
(``create_task``): when running directly in the gateway main process
(``_HERMES_GATEWAY == "1"`` AND no ``HERMES_KANBAN_TASK``) and the caller
did not pass an explicit ``untrusted_origin=``, the card is stamped
``untrusted_origin=1``. ``_default_spawn`` then de-privileges it via the
existing ``bool(task.untrusted_origin)`` branch — which simultaneously
covers the ``/kanban dispatch`` half (dispatch reads the stored row, so a
stamped card is de-privileged at spawn without a separate dispatch guard).

The DB-layer chokepoint is reached identically by the CLI ``run_slash``
path (un-editable here) and the in-process ``kanban_create`` tool, so a
single test at the ``create_task`` + ``_default_spawn`` boundary proves the
closure for both surfaces.

We intercept ``subprocess.Popen`` to capture the worker argv + env without
spawning a real hermes subprocess, mirroring ``test_r1_worker_depriv.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_and_spawn(monkeypatch, *, created_by="some-profile", create_kwargs=None):
    """Create a task (in whatever env the test set up), then run
    ``_default_spawn`` under a fake Popen.

    Returns ``(stored_untrusted_origin, cmd, env)`` where
    ``stored_untrusted_origin`` is the persisted ``tasks.untrusted_origin``
    column for the created card (proving the create-time stamp), and
    ``cmd`` / ``env`` are captured from the spawn (proving the dispatch-time
    de-privilege that reads that stored row).
    """
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # Bundled kanban-worker skill is absent from the isolated home; pin
    # availability False so argv shape doesn't depend on the test machine.
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="r1 gateway-slash probe",
            assignee="some-profile",
            created_by=created_by,
            **(create_kwargs or {}),
        )
        task = kb.get_task(conn, tid)
        stored_untrusted = int(task.untrusted_origin)
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    assert pid == 4242
    assert "cmd" in captured and "env" in captured, "Popen was not invoked"
    return stored_untrusted, captured["cmd"], captured["env"]


def _assert_deprivileged(cmd, env):
    """Assert the worker was launched with the full R1 untrusted sandbox."""
    # 1.1a env approval gate
    assert env.get("HERMES_EXEC_ASK") == "1", (
        "gateway-origin worker must force the approval gate on"
    )
    assert "HERMES_YOLO_MODE" not in env, (
        "inherited yolo bypass must be stripped for the gateway-origin worker"
    )
    # Transitive closure: the child env must re-stamp untrusted so any
    # Gen-2 card it creates inherits the sandbox.
    assert env.get("HERMES_UNTRUSTED_ORIGIN") == "1", (
        "gateway-origin worker env must re-stamp HERMES_UNTRUSTED_ORIGIN for "
        "transitive (Gen-2+) closure"
    )
    # 1.1b toolset narrowing
    assert "-t" in cmd, f"gateway-origin argv must narrow toolsets: {cmd}"
    idx = cmd.index("-t")
    assert cmd[idx + 1] == "web,vision", (
        f"expected research-only toolsets, got {cmd[idx + 1]!r}"
    )
    assert "chat" in cmd, f"argv missing chat subcommand: {cmd}"
    assert cmd.index("-t") > cmd.index("chat"), (
        f"-t must come after the chat token: {cmd}"
    )
    for forbidden in ("terminal", "process", "execute_code", "delegation"):
        assert forbidden not in cmd[idx + 1].split(","), (
            f"{forbidden} must not be in the narrowed toolset"
        )


def _assert_full_privilege(cmd, env):
    """Assert the worker kept its full (trusted) capability surface."""
    assert env.get("HERMES_EXEC_ASK") is None, (
        "trusted-origin worker must not be force-gated by the R1 deprivilege"
    )
    assert "-t" not in cmd, (
        f"trusted-origin worker must keep its full toolset (no -t): {cmd}"
    )
    assert env.get("HERMES_UNTRUSTED_ORIGIN") is None, (
        "trusted-origin worker env must not carry the untrusted re-stamp"
    )


# ---------------------------------------------------------------------------
# Gen-1 laundering: gateway-direct /kanban create|dispatch
# ---------------------------------------------------------------------------


def test_gateway_main_process_create_is_stamped_untrusted(kanban_home, monkeypatch):
    """``/kanban create`` in the gateway main process => card stamped
    ``untrusted_origin=1`` even though ``created_by`` is a normal profile and
    no ``HERMES_UNTRUSTED_ORIGIN`` is present."""
    # Gateway main process: marker present, NOT a dispatcher-spawned worker.
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
    # Inherited yolo that must be stripped for the dispatched worker.
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch, created_by="some-profile"
    )

    assert stored_untrusted == 1, (
        "gateway-main-process create must persist untrusted_origin=1 "
        "(closes the /kanban dispatch half transitively via the stored row)"
    )
    _assert_deprivileged(cmd, env)


def test_gateway_dispatch_half_deprivileges_from_stored_row(kanban_home, monkeypatch):
    """The ``/kanban dispatch`` half is covered transitively: a card stamped
    untrusted at create time de-privileges at spawn even when the dispatcher
    tick runs LATER with the gateway env no longer relevant to the stamp.

    We simulate this by creating under the gateway env, then clearing the
    gateway marker before spawning — the de-privilege must come from the
    persisted ``untrusted_origin`` column, not from the live env."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="r1 gateway dispatch-half probe",
            assignee="some-profile",
            created_by="some-profile",
        )
        # Dispatcher tick: gateway-origin marker gone, but the stored row
        # still carries untrusted_origin=1.
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        task = kb.get_task(conn, tid)
        assert int(task.untrusted_origin) == 1
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    assert pid == 4242
    _assert_deprivileged(captured["cmd"], captured["env"])


# ---------------------------------------------------------------------------
# Non-regression: trusted create paths must stay full-privilege
# ---------------------------------------------------------------------------


def test_dispatcher_worker_create_is_not_over_stamped(kanban_home, monkeypatch):
    """A create from inside a dispatcher-spawned worker (``HERMES_KANBAN_TASK``
    set, running under the gateway) must NOT be stamped untrusted *by the
    gateway-origin signal* — its own untrusted state is carried separately by
    ``HERMES_UNTRUSTED_ORIGIN``. With neither set, the child stays trusted."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # This IS a dispatcher-spawned worker context.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch, created_by="some-profile"
    )

    assert stored_untrusted == 0, (
        "a trusted dispatcher-worker create must not be stamped untrusted by "
        "the gateway-origin signal (HERMES_KANBAN_TASK excludes it)"
    )
    _assert_full_privilege(cmd, env)


def test_dispatcher_worker_untrusted_env_still_propagates(kanban_home, monkeypatch):
    """Regression guard for the existing worker chokepoint: a worker that DOES
    carry ``HERMES_UNTRUSTED_ORIGIN`` still stamps its children untrusted, even
    though ``HERMES_KANBAN_TASK`` excludes the gateway-origin branch."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_UNTRUSTED_ORIGIN", "1")

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch, created_by="some-profile"
    )

    assert stored_untrusted == 1, (
        "an already-untrusted worker must still stamp its Gen-2 cards untrusted"
    )
    _assert_deprivileged(cmd, env)


def test_plain_cli_create_stays_trusted(kanban_home, monkeypatch):
    """A standalone ``hermes kanban create`` (no gateway, no worker) must stay
    full-privilege — the gateway-origin signal is absent."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch, created_by="some-profile"
    )

    assert stored_untrusted == 0, (
        "a standalone CLI/dashboard create has no gateway-origin signal and "
        "must stay trusted"
    )
    _assert_full_privilege(cmd, env)


def test_explicit_untrusted_false_overrides_gateway_signal(kanban_home, monkeypatch):
    """An explicit ``untrusted_origin=False`` argument wins outright over the
    gateway-origin auto-derivation.

    This preserves the decompose-propagation invariant: ``_decompose_task``
    passes the already-computed root ``untrusted_origin`` explicitly, and that
    value must never be re-derived from ambient env. A trusted root decomposed
    inside the gateway must keep producing trusted children."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch,
        created_by="some-profile",
        create_kwargs={"untrusted_origin": False},
    )

    assert stored_untrusted == 0, (
        "explicit untrusted_origin=False must override the gateway-origin "
        "signal (decompose propagation invariant)"
    )
    _assert_full_privilege(cmd, env)


def test_explicit_untrusted_true_is_honored_without_gateway(kanban_home, monkeypatch):
    """Symmetric guard: an explicit ``untrusted_origin=True`` stamps untrusted
    even with no gateway/worker env signal (trusted caller opting a card into
    the sandbox)."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    stored_untrusted, cmd, env = _create_and_spawn(
        monkeypatch,
        created_by="some-profile",
        create_kwargs={"untrusted_origin": True},
    )

    assert stored_untrusted == 1
    _assert_deprivileged(cmd, env)
