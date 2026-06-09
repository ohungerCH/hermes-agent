"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_mixers = {}
    adapter._voice_fx_cfg = {"enabled": False}
    adapter._client = MagicMock()
    setattr(adapter._client, "voice_clients", [])
    return adapter


def _discard_coroutine_task(coro):
    """Test helper: replace ensure_future without leaking created coroutines."""
    coro.close()
    return asyncio.create_task(asyncio.sleep(0))


@pytest.mark.asyncio
async def test_concurrent_joins_do_not_double_connect():
    """Two concurrent join_voice_channel calls on the same guild must
    serialize through the per-guild lock — only ONE channel.connect()
    actually fires; the second sees the _voice_clients entry the first
    just installed."""
    adapter = _make_adapter()

    connect_count = [0]
    release = asyncio.Event()

    class FakeVC:
        def __init__(self, channel):
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, _channel):
            return None

    async def slow_connect(self):
        connect_count[0] += 1
        await release.wait()
        return FakeVC(self)

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver",
                      MagicMock(return_value=MagicMock(start=lambda: None))):
        with patch.object(discord_mod.asyncio, "ensure_future", _discard_coroutine_task):
            t1 = asyncio.create_task(adapter.join_voice_channel(channel))
            t2 = asyncio.create_task(adapter.join_voice_channel(channel))
            await asyncio.sleep(0.05)
            release.set()
            r1, r2 = await asyncio.gather(t1, t2)

    assert connect_count[0] == 1, (
        f"expected 1 channel.connect() call, got {connect_count[0]} — "
        "per-guild lock is not serializing join_voice_channel"
    )
    assert r1 is True and r2 is True
    assert 42 in adapter._voice_clients


@pytest.mark.asyncio
async def test_leave_recovers_discord_py_voice_client_when_adapter_map_is_stale():
    """If adapter state lost the VC but discord.py still has it, /voice leave
    must still disconnect instead of claiming the bot is absent."""
    adapter = _make_adapter()

    class FakeVC:
        def __init__(self):
            self.guild = MagicMock(id=42)
            self.channel = MagicMock(id=111)
            self.disconnected = False

        def is_connected(self):
            return not self.disconnected

        def is_playing(self):
            return False

        async def disconnect(self):
            self.disconnected = True

    vc = FakeVC()
    setattr(adapter._client, "voice_clients", [vc])

    assert adapter.is_in_voice_channel(42) is True
    await adapter.leave_voice_channel(42)

    assert vc.disconnected is True
    assert adapter.is_in_voice_channel(42) is False


@pytest.mark.asyncio
async def test_join_recovers_when_channel_connect_says_already_connected():
    """A stale discord.py voice client should be reconciled rather than
    surfacing an contradictory "Already connected" join failure."""
    adapter = _make_adapter()

    class FakeVC:
        def __init__(self, channel):
            self.guild = MagicMock(id=42)
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, channel):
            self.channel = channel

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42

    async def connect_raises():
        raise RuntimeError("Already connected to a voice channel.")

    channel.connect = connect_raises
    vc = FakeVC(channel)
    setattr(adapter._client, "voice_clients", [vc])

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver",
                      MagicMock(return_value=MagicMock(start=lambda: None, _running=True))):
        with patch.object(discord_mod.asyncio, "ensure_future", _discard_coroutine_task):
            assert await adapter.join_voice_channel(channel) is True

    assert adapter._voice_clients[42] is vc
    assert 42 in adapter._voice_receivers
