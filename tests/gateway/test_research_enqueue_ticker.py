"""Tests for the periodic TRIGGER of the research enqueue worker (B3).

The worker (``tools.research_enqueue_worker``) deliberately ships only
``run_once`` / ``process_one`` and no scheduler of its own — that is its
integration boundary. The periodic trigger lives in the gateway's in-process
cron ticker (``gateway.run._start_cron_ticker``), which already piggy-backs the
curator / cache cleanup / paste sweep on ``tick_count``. These tests exercise the
extracted, directly-callable ``_research_enqueue_scan_tick`` helper — a hermetic
"scan-tick simulation" with no thread / timing dependency — to prove:

- the worker's ``run_once`` is invoked on a due tick (cadence honoured);
- it is NOT invoked off-cadence / when disabled (``every <= 0``);
- a worker exception is swallowed (fail-soft) so the ticker never dies;
- the scan is wired into the live ticker loop (regression pin against the hook
  being silently dropped).

No deploy / docker / start: this is the build-side unit gate for the trigger.
"""

import logging
from unittest.mock import patch

import gateway.run as gw


# ---------------------------------------------------------------------------
# 1. Cadence: the worker fires on a due tick, and is skipped off-cadence.
# ---------------------------------------------------------------------------

class TestScanCadence:
    def test_fires_on_due_tick(self):
        with patch("tools.research_enqueue_worker.run_once", return_value=[]) as m:
            due = gw._research_enqueue_scan_tick(tick_count=5, every=5)
        assert due is True
        m.assert_called_once_with()

    def test_skips_off_cadence_tick(self):
        with patch("tools.research_enqueue_worker.run_once") as m:
            due = gw._research_enqueue_scan_tick(tick_count=4, every=5)
        assert due is False
        m.assert_not_called()

    def test_every_one_fires_each_tick(self):
        """With the default RESEARCH_SCAN_EVERY == 1 the worker scans every tick."""
        calls = 0
        with patch("tools.research_enqueue_worker.run_once", return_value=[]) as m:
            for t in range(1, 4):
                if gw._research_enqueue_scan_tick(tick_count=t, every=1):
                    calls += 1
        assert calls == 3
        assert m.call_count == 3

    def test_disabled_when_every_zero_or_negative(self):
        with patch("tools.research_enqueue_worker.run_once") as m:
            assert gw._research_enqueue_scan_tick(tick_count=10, every=0) is False
            assert gw._research_enqueue_scan_tick(tick_count=10, every=-1) is False
        m.assert_not_called()

    def test_default_every_is_one_tick(self):
        # The module default cadence: pick up enqueued research once per tick.
        assert gw.RESEARCH_SCAN_EVERY == 1


# ---------------------------------------------------------------------------
# 2. Fail-soft: a worker error must never propagate out of the ticker.
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_worker_exception_is_swallowed(self, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch("tools.research_enqueue_worker.run_once",
                       side_effect=RuntimeError("enqueue dir gone")):
                # Must NOT raise — a transient enqueue-dir problem cannot kill the
                # ticker thread (which also drives every other cron job).
                due = gw._research_enqueue_scan_tick(tick_count=1, every=1)
        assert due is True  # the scan was attempted this tick.
        assert "research enqueue scan error" in caplog.text.lower()

    def test_import_error_is_swallowed(self):
        # Even if the worker module fails to import, the ticker survives.
        with patch.dict("sys.modules", {"tools.research_enqueue_worker": None}):
            # sys.modules[name] = None makes ``import`` raise ImportError.
            due = gw._research_enqueue_scan_tick(tick_count=1, every=1)
        assert due is True


# ---------------------------------------------------------------------------
# 3. Value-free logging: the trigger never logs raw assignment content. The
#    worker returns only opaque job_ids; we log a count, never topic/goal.
# ---------------------------------------------------------------------------

class TestNoArt9InTriggerLog:
    def test_only_count_logged_not_payload(self, caplog):
        # run_once returns opaque job_ids; the trigger logs a count only.
        with caplog.at_level(logging.DEBUG):
            with patch("tools.research_enqueue_worker.run_once",
                       return_value=["job-abc", "job-def"]):
                gw._research_enqueue_scan_tick(tick_count=1, every=1)
        assert "fired 2 job(s)" in caplog.text
        # No raw topic/goal can appear: the trigger never receives any.
        for forbidden in ("topic", "goal=", "context="):
            assert forbidden not in caplog.text


# ---------------------------------------------------------------------------
# 4. Wiring regression: the scan IS called from the live ticker loop. Guards
#    against the hook being silently dropped from _start_cron_ticker.
# ---------------------------------------------------------------------------

class TestTickerLoopWiring:
    def test_ticker_loop_invokes_scan(self):
        """Run ONE iteration of the real _start_cron_ticker loop and assert the
        research scan is invoked. Everything heavy (cron_tick + the periodic
        chores) is stubbed; the loop is stopped after the first iteration via the
        scan hook itself so this is bounded and hermetic (no real sleep/timing)."""
        import threading

        stop_event = threading.Event()
        seen = {"scan": 0}

        def _fake_scan(tick_count, *a, **k):
            seen["scan"] += 1
            stop_event.set()  # exit the loop after the first iteration.
            return True

        with patch.object(gw, "_research_enqueue_scan_tick", side_effect=_fake_scan), \
             patch("cron.scheduler.tick", return_value=0), \
             patch("gateway.platforms.base.cleanup_image_cache", return_value=0), \
             patch("gateway.platforms.base.cleanup_document_cache", return_value=0), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
            # interval=0 so the post-iteration stop_event.wait returns immediately.
            gw._start_cron_ticker(stop_event, adapters=None, loop=None, interval=0)

        assert seen["scan"] == 1, "research enqueue scan was not invoked by the ticker loop"
