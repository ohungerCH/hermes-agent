"""R1 Gen-2+ transitive-closure tests for the untrusted-origin flag.

Round 1 deprivileged the Gen-1 gateway voice/chat origin
(``created_by == "gateway-voice-background"``): the worker gets the approval
gate forced on, the inherited yolo bypass stripped, and a narrowed
``-t web,vision`` toolset surface.

Round 2 closes the *transitive* hole: a deprivileged Gen-1 worker can still
reach ``kanban_create`` (the ``kanban`` toolset is auto-appended whenever
HERMES_KANBAN_TASK is set, even under ``-t web,vision``). Without closure, the
Gen-2 card it spawns would dispatch with FULL privileges, escaping the sandbox.

The closure mechanism:

  * ``_default_spawn`` sets ``HERMES_UNTRUSTED_ORIGIN=1`` in an untrusted
    worker's env (both the Gen-1 marker origin and the Gen-2+ flag origin).
  * ``create_task`` reads HERMES_UNTRUSTED_ORIGIN as the default for the new
    ``untrusted_origin`` column — a single DB-level chokepoint, so EVERY
    worker-reachable create path inherits the flag without enumeration.
  * On dispatch, ``_default_spawn`` treats ``untrusted_origin=1`` exactly like
    the Gen-1 marker (``is_untrusted`` disjunct) — re-deprivileging Gen-2,
    which re-stamps Gen-3, ... = transitive closure.

Critically, ``created_by`` is NOT overwritten with the marker: it stays the
real profile so attribution and the completion-time ``_verify_created_cards``
hallucinated-cards guard keep working (regression covered explicitly below).

The ``_default_spawn`` test intercepts ``subprocess.Popen`` to capture the
worker argv + env without spawning a real hermes subprocess, mirroring
tests/plugins/test_r1_worker_depriv.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


VOICE_ORIGIN = "gateway-voice-background"


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


# ---------------------------------------------------------------------------
# (a) create_task chokepoint: env HERMES_UNTRUSTED_ORIGIN -> untrusted_origin=1
#     and created_by STAYS the worker profile (no marker overwrite).
# ---------------------------------------------------------------------------


def test_create_task_stamps_untrusted_origin_from_env(kanban_home, monkeypatch):
    """With HERMES_UNTRUSTED_ORIGIN set, a created card is flagged untrusted,
    while created_by keeps the real worker profile (NOT the marker)."""
    monkeypatch.setenv("HERMES_UNTRUSTED_ORIGIN", "1")
    monkeypatch.setenv("HERMES_PROFILE", "gen1-worker")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="gen-2 child from untrusted worker",
            assignee="some-profile",
            created_by="gen1-worker",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    assert task is not None
    assert task.untrusted_origin == 1, (
        "card created under HERMES_UNTRUSTED_ORIGIN must be flagged untrusted"
    )
    # The marker must NOT clobber attribution — created_by stays the profile so
    # _verify_created_cards can still match the card to its completing worker.
    assert task.created_by == "gen1-worker", (
        "created_by must remain the real worker profile, not the untrusted marker"
    )
    assert task.created_by != VOICE_ORIGIN


def test_create_task_no_env_is_not_untrusted(kanban_home, monkeypatch):
    """Without HERMES_UNTRUSTED_ORIGIN, a card is trusted (untrusted_origin=0)
    and created_by is the normal HERMES_PROFILE — the trusted-path baseline."""
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "normal-worker")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="trusted child",
            assignee="some-profile",
            created_by="normal-worker",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    assert task is not None
    assert task.untrusted_origin == 0, (
        "without the env marker the card must NOT be flagged untrusted"
    )
    assert task.created_by == "normal-worker"


def test_create_task_explicit_arg_overrides_env(kanban_home, monkeypatch):
    """An explicit untrusted_origin=False (trusted caller) wins over an
    ambient env marker; explicit True wins without the env too."""
    monkeypatch.setenv("HERMES_UNTRUSTED_ORIGIN", "1")

    conn = kb.connect()
    try:
        tid_forced_trusted = kb.create_task(
            conn,
            title="explicit trusted despite env",
            assignee="some-profile",
            untrusted_origin=False,
        )
        monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
        tid_forced_untrusted = kb.create_task(
            conn,
            title="explicit untrusted without env",
            assignee="some-profile",
            untrusted_origin=True,
        )
        t_trusted = kb.get_task(conn, tid_forced_trusted)
        t_untrusted = kb.get_task(conn, tid_forced_untrusted)
    finally:
        conn.close()

    assert t_trusted.untrusted_origin == 0
    assert t_untrusted.untrusted_origin == 1


# ---------------------------------------------------------------------------
# (b) _default_spawn transitivity: an untrusted_origin=1 card whose created_by
#     is a NORMAL profile (a Gen-2 card) is still deprivileged AND its worker
#     env carries HERMES_UNTRUSTED_ORIGIN so Gen-3 inherits (Gen-2 -> Gen-3).
# ---------------------------------------------------------------------------


def _spawn_and_capture(monkeypatch, *, created_by, untrusted_origin=None):
    """Create a task with the given origin, run _default_spawn under a fake
    Popen, and return (cmd, env) captured from the spawn call."""
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # The bundled kanban-worker skill is absent from the isolated home; pin
    # availability False so argv shape doesn't depend on the test machine.
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    conn = kb.connect()
    try:
        create_kwargs = dict(
            title="gen-2 transitive probe",
            assignee="some-profile",
            created_by=created_by,
        )
        if untrusted_origin is not None:
            create_kwargs["untrusted_origin"] = untrusted_origin
        tid = kb.create_task(conn, **create_kwargs)
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    assert pid == 4242
    assert "cmd" in captured and "env" in captured, "Popen was not invoked"
    return captured["cmd"], captured["env"]


def test_gen2_flag_deprivileges_like_gen1_marker(kanban_home, monkeypatch):
    """A Gen-2 card (created_by = a NORMAL profile, untrusted_origin=1) is
    deprivileged exactly like the Gen-1 marker origin."""
    # Make sure the ambient env doesn't already carry the flag — we want to
    # attribute the worker env solely to the per-task flag.
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    cmd, env = _spawn_and_capture(
        monkeypatch, created_by="gen1-worker", untrusted_origin=True
    )

    # 1.1a env gate — forced on, yolo stripped, even though created_by is NOT
    # the gateway-voice-background marker.
    assert env.get("HERMES_EXEC_ASK") == "1", (
        "untrusted_origin=1 card must force the approval gate on"
    )
    assert "HERMES_YOLO_MODE" not in env, (
        "inherited yolo bypass must be stripped for the untrusted Gen-2 worker"
    )
    # 1.1b toolset narrowing — same -t web,vision after chat.
    assert "-t" in cmd, f"Gen-2 worker argv must narrow toolsets: {cmd}"
    idx = cmd.index("-t")
    assert cmd[idx + 1] == "web,vision"
    assert cmd.index("-t") > cmd.index("chat")


def test_gen2_worker_env_carries_untrusted_origin_for_gen3(kanban_home, monkeypatch):
    """Transitivity: the deprivileged Gen-2 worker's env must itself carry
    HERMES_UNTRUSTED_ORIGIN, so any Gen-3 card it creates inherits the flag."""
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    _cmd, env = _spawn_and_capture(
        monkeypatch, created_by="gen1-worker", untrusted_origin=True
    )

    assert env.get("HERMES_UNTRUSTED_ORIGIN") == "1", (
        "Gen-2 worker env must propagate HERMES_UNTRUSTED_ORIGIN so Gen-3 "
        "cards created via the create_task chokepoint inherit the sandbox"
    )


def test_gen1_marker_still_propagates_untrusted_origin(kanban_home, monkeypatch):
    """Regression: the Gen-1 marker origin (Round 1) also now propagates
    HERMES_UNTRUSTED_ORIGIN into its worker env (the closure entry point)."""
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    _cmd, env = _spawn_and_capture(monkeypatch, created_by=VOICE_ORIGIN)

    assert env.get("HERMES_EXEC_ASK") == "1"
    assert env.get("HERMES_UNTRUSTED_ORIGIN") == "1", (
        "Gen-1 marker worker must seed HERMES_UNTRUSTED_ORIGIN to start closure"
    )


def test_trusted_card_is_not_deprivileged_and_no_marker_leaks(kanban_home, monkeypatch):
    """A fully trusted card (no marker, untrusted_origin=0) keeps full
    privileges AND does not gain HERMES_UNTRUSTED_ORIGIN — the regression guard
    that the Gen-2 plumbing did not over-broaden the untrusted surface."""
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    cmd, env = _spawn_and_capture(
        monkeypatch, created_by="some-user", untrusted_origin=False
    )

    assert "-t" not in cmd, f"trusted worker must keep full toolset: {cmd}"
    assert env.get("HERMES_EXEC_ASK") is None, (
        "trusted worker must not be force-gated"
    )
    assert env.get("HERMES_YOLO_MODE") == "1", "trusted worker env left intact"
    assert "HERMES_UNTRUSTED_ORIGIN" not in env, (
        "trusted worker must NOT be marked untrusted"
    )


# ---------------------------------------------------------------------------
# (3) No HallucinatedCardsError regression: a Gen-2 card (created_by=profile,
#     untrusted_origin=1) still verifies for its completing worker. This is the
#     test that proves choosing the flag column over overwriting created_by was
#     correct — overwriting would have made this card phantom.
# ---------------------------------------------------------------------------


def test_gen2_card_still_verifies_no_hallucination_regression(kanban_home, monkeypatch):
    """A Gen-2 card whose created_by matches the completing worker's assignee
    is VERIFIED (not phantom), despite untrusted_origin=1 — proving the flag
    design does not break completion-time created_cards verification."""
    monkeypatch.setenv("HERMES_UNTRUSTED_ORIGIN", "1")
    monkeypatch.setenv("HERMES_PROFILE", "gen1-worker")

    conn = kb.connect()
    try:
        # The completing (Gen-1) task is itself assigned to profile gen1-worker.
        parent_tid = kb.create_task(
            conn,
            title="gen-1 completing task",
            assignee="gen1-worker",
            created_by="dispatcher",
        )
        # The Gen-2 child the Gen-1 worker spawns via kanban_create: created_by
        # is stamped with the worker's own profile (gen1-worker), untrusted=1.
        child_tid = kb.create_task(
            conn,
            title="gen-2 child",
            assignee="some-profile",
            created_by="gen1-worker",
        )
        child = kb.get_task(conn, child_tid)
        assert child.untrusted_origin == 1, "precondition: child is untrusted"

        verified, phantom = kb._verify_created_cards(
            conn, parent_tid, [child_tid]
        )
    finally:
        conn.close()

    assert child_tid in verified, (
        "Gen-2 card must be VERIFIED — created_by==completing_assignee matches; "
        "the untrusted_origin flag must not make it phantom"
    )
    assert child_tid not in phantom
    assert phantom == []


def test_marker_overwrite_would_have_regressed_control(kanban_home, monkeypatch):
    """Control: had created_by been overwritten with the marker (the rejected
    design), the very same Gen-2 card WOULD be phantom — demonstrating why the
    flag column was chosen instead."""
    conn = kb.connect()
    try:
        parent_tid = kb.create_task(
            conn,
            title="gen-1 completing task",
            assignee="gen1-worker",
            created_by="dispatcher",
        )
        # Simulate the REJECTED design: created_by overwritten with the marker.
        bad_child_tid = kb.create_task(
            conn,
            title="gen-2 child (marker-overwrite simulation)",
            assignee="some-profile",
            created_by=VOICE_ORIGIN,
        )
        verified, phantom = kb._verify_created_cards(
            conn, parent_tid, [bad_child_tid]
        )
    finally:
        conn.close()

    # created_by==marker != completing assignee (gen1-worker), and the child is
    # not linked as a parent's child -> phantom. This is the regression the
    # flag-column design avoids.
    assert bad_child_tid in phantom, (
        "marker-overwrite would make the Gen-2 card phantom (the avoided regression)"
    )
    assert bad_child_tid not in verified


# ---------------------------------------------------------------------------
# Second-inserter closure: decompose_triage_task bypasses the create_task env
# chokepoint, so it must propagate untrusted_origin from the triage root. An
# untrusted worker can create a triage=True card (untrusted via the chokepoint)
# which the auto-decompose path then fans out — without propagation the
# children would dispatch with FULL privileges, escaping the sandbox.
# ---------------------------------------------------------------------------


def test_decompose_propagates_untrusted_origin_to_children(kanban_home, monkeypatch):
    """An untrusted triage root decomposed into children stamps each child
    untrusted_origin=1, closing the decompose-path Gen-2 escape."""
    monkeypatch.setenv("HERMES_UNTRUSTED_ORIGIN", "1")

    conn = kb.connect()
    try:
        # The untrusted worker creates a triage card via the chokepoint.
        root_tid = kb.create_task(
            conn, title="untrusted rough idea", triage=True
        )
        assert kb.get_task(conn, root_tid).untrusted_origin == 1
        # The auto-decompose path (auxiliary LLM) fans it out. Clear the env
        # first to prove propagation comes from the ROOT flag, not ambient env.
        monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)
        child_ids = kb.decompose_triage_task(
            conn,
            root_tid,
            root_assignee="orchestrator",
            children=[
                {"title": "child a", "assignee": "researcher", "parents": []},
                {"title": "child b", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
        )
        children = [kb.get_task(conn, cid) for cid in child_ids]
    finally:
        conn.close()

    assert child_ids is not None and len(child_ids) == 2
    for child in children:
        assert child.untrusted_origin == 1, (
            "decompose children of an untrusted triage root must inherit the "
            "untrusted flag (else they dispatch with full privileges)"
        )


def test_decompose_trusted_root_keeps_children_trusted(kanban_home, monkeypatch):
    """A trusted triage root's decomposed children stay trusted — the fix
    propagates only when the root is untrusted (no over-broadening)."""
    monkeypatch.delenv("HERMES_UNTRUSTED_ORIGIN", raising=False)

    conn = kb.connect()
    try:
        root_tid = kb.create_task(conn, title="trusted rough idea", triage=True)
        assert kb.get_task(conn, root_tid).untrusted_origin == 0
        child_ids = kb.decompose_triage_task(
            conn,
            root_tid,
            root_assignee="orchestrator",
            children=[{"title": "child a", "assignee": "researcher", "parents": []}],
            author="decomposer",
        )
        children = [kb.get_task(conn, cid) for cid in child_ids]
    finally:
        conn.close()

    assert child_ids is not None and len(child_ids) == 1
    for child in children:
        assert child.untrusted_origin == 0, (
            "trusted triage root must not produce untrusted children"
        )
