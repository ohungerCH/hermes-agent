"""Contract test: the s6-overlay stage2 hook makes the Bridge->Engine research-enqueue
handoff deploy-durable (#65 P1).

The Bridge (UID 10024) WRITES cron/enqueue/<job>.json; the Engine (UID 10000) reads/
claims. The unconditional `chown -R hermes:hermes "$HERMES_HOME/cron"` block resets
cron/enqueue to 10000-only on every boot, so the Bridge EACCESes
(RESEARCH_ENQUEUE_WRITE_FAILED).

The image has no `acl` package, so the fix is a dedicated shared GID + setgid directory
(standard POSIX): cron/enqueue is owned by GID 10240 with mode 2770 (rwx owner, rwx group,
setgid so new files inherit the group, NO other bit -> never world-writable). This MUST run
AFTER the `chown -R cron` block, otherwise that block clobbers it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def test_setgid_block_present(stage2_text: str) -> None:
    """The dedicated research-enqueue setgid block must exist and use the shared GID."""
    assert "Research-Enqueue cross-UID handoff" in stage2_text, (
        "stage2-hook.sh must contain the #65 P1 research-enqueue setgid block"
    )
    assert "JARVIS_RESEARCH_GID" in stage2_text
    assert 'chmod 2770 "$HERMES_HOME/cron/enqueue"' in stage2_text, (
        "the block must chmod cron/enqueue to 2770 (setgid + group rwx, no other bit)"
    )
    assert 'chgrp "$JARVIS_RESEARCH_GID" "$HERMES_HOME/cron/enqueue"' in stage2_text, (
        "the block must chgrp cron/enqueue to the shared research GID"
    )
    assert 'mkdir -p "$HERMES_HOME/cron/enqueue"' in stage2_text, (
        "the block must create cron/enqueue (Subpath-Mount caveat: it must exist)"
    )


def test_setgid_block_runs_after_chown_cron(stage2_text: str) -> None:
    """The setgid block MUST come AFTER the unconditional `chown -R … cron` block, else
    the recursive chown clobbers the group/setgid bit on every boot."""
    chown_idx = stage2_text.find('chown -R hermes:hermes "$HERMES_HOME/cron"')
    setgid_idx = stage2_text.find("Research-Enqueue cross-UID handoff")
    assert chown_idx != -1, "the `chown -R … cron` block must exist"
    assert setgid_idx != -1, "the setgid block must exist"
    assert setgid_idx > chown_idx, (
        "the research-enqueue setgid block must run AFTER the `chown -R … cron` block "
        "(otherwise the recursive chown clobbers the shared group + setgid bit)"
    )


def test_not_world_writable(stage2_text: str) -> None:
    """The handoff must NOT be world-writable: only the shared 2770 mode, never o+w/0777
    on cron/enqueue (the Worker-threat-model keeps it to Bridge+Engine only)."""
    # No 0777 / o+w chmod targeting cron/enqueue.
    assert not re.search(r"chmod\s+0?777\s+\"?\$\{?HERMES_HOME\}?/cron/enqueue", stage2_text)
    assert not re.search(r"chmod\s+o\+w\s+\"?\$\{?HERMES_HOME\}?/cron/enqueue", stage2_text)


def _extract_block(text: str) -> str:
    """Extract the research-enqueue setgid block (from its banner comment to the final
    chmod 2770 line) so it can be run standalone in a sandbox."""
    start = text.find("# --- Research-Enqueue cross-UID handoff")
    assert start != -1
    end = text.find('chmod 2770 "$HERMES_HOME/cron/enqueue"', start)
    assert end != -1
    # include the rest of that chmod statement (through the next `fi`/end-of-line continuation)
    end = text.find("fi", end)
    assert end != -1
    return text[start:end + 2]


def test_block_runs_idempotently_and_sets_2770(stage2_text: str) -> None:
    """Run the block twice in a sandbox with chgrp/chmod/groupadd/usermod/getent/id
    stubbed; assert it records a 2770 chmod on cron/enqueue and the second run is a no-op
    error-free (idempotent)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _extract_block(stage2_text)

    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        home = dpath / "home"
        home.mkdir()
        log = dpath / "calls.log"
        # Stubs: record invocations; getent finds nothing (so groupadd path runs), id -G
        # returns empty (so usermod path runs); none of them need real privileges.
        script = (
            "set -eu\n"
            f'HERMES_HOME="{home}"\n'
            f'chgrp() {{ echo "chgrp $*" >> "{log}"; }}\n'
            f'chmod() {{ echo "chmod $*" >> "{log}"; }}\n'
            f'groupadd() {{ echo "groupadd $*" >> "{log}"; }}\n'
            f'usermod() {{ echo "usermod $*" >> "{log}"; }}\n'
            'getent() { return 1; }\n'
            'id() { return 0; }\n'
            + block
            + "\n"
            # run it a second time to prove idempotency (no set -e abort).
            + block
            + "\n"
        )
        script_path = dpath / "harness.sh"
        script_path.write_text(script)
        proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        calls = log.read_text() if log.exists() else ""
        assert "chmod 2770" in calls, "the block must chmod cron/enqueue to 2770"
        assert "cron/enqueue" in calls
        # idempotent: chmod 2770 recorded at least twice (both runs), no error.
        assert calls.count("chmod 2770") >= 2, "second run must also succeed (idempotent)"


def test_creates_enqueue_dir(stage2_text: str) -> None:
    """The block must create cron/enqueue so a fresh-volume first boot satisfies the
    Bridge's Subpath mount (which fails the `up` if the dir is absent)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _extract_block(stage2_text)
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        home = dpath / "home"
        home.mkdir()
        script = (
            "set -eu\n"
            f'HERMES_HOME="{home}"\n'
            "chgrp() { :; }\n"
            "chmod() { :; }\n"
            "groupadd() { :; }\n"
            "usermod() { :; }\n"
            "getent() { return 1; }\n"
            "id() { return 0; }\n"
            + block
        )
        (dpath / "h.sh").write_text(script)
        proc = subprocess.run([bash, str(dpath / "h.sh")], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert (home / "cron" / "enqueue").is_dir(), "cron/enqueue must be created"
