"""Tests for hermes-api-server toolset and API server tool availability."""
import pytest
from unittest.mock import patch, MagicMock


from toolsets import resolve_toolset, get_toolset, validate_toolset


class TestHermesApiServerToolset:
    """Tests for the hermes-api-server toolset definition."""

    def test_toolset_exists(self):
        ts = get_toolset("hermes-api-server")
        assert ts is not None

    def test_toolset_validates(self):
        assert validate_toolset("hermes-api-server")

    def test_toolset_includes_web_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("hermes-api-server")
        expected = [
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "execute_code", "delegate_task",
            "todo", "memory", "session_search", "cronjob",
        ]
        for tool in expected:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool in tools, f"Missing browser tool: {tool}"

    def test_toolset_includes_homeassistant_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]:
            assert tool in tools, f"Missing HA tool: {tool}"

    def test_toolset_excludes_clarify(self):
        tools = resolve_toolset("hermes-api-server")
        assert "clarify" not in tools

    def test_toolset_excludes_send_message(self):
        tools = resolve_toolset("hermes-api-server")
        assert "send_message" not in tools

    def test_toolset_excludes_text_to_speech(self):
        tools = resolve_toolset("hermes-api-server")
        assert "text_to_speech" not in tools


class TestApiServerPlatformConfig:
    def test_platforms_dict_includes_api_server(self):
        from hermes_cli.tools_config import PLATFORMS
        assert "api_server" in PLATFORMS
        assert PLATFORMS["api_server"]["default_toolset"] == "hermes-api-server"

    @pytest.mark.skip(
        reason="jarvis-divergence (0.18.2-Merge): upstream nimmt an, dass "
        "api_server per Default Toolsets aufloest (terminal-Composite-Mitglied). "
        "Unser Fork klemmt api_server als untrusted Voice-/Inbound-Boden "
        "UNBEDINGT auf 0 Toolsets/MCP (FAIL_CLOSED_PLATFORMS exit-clamp in "
        "hermes_cli/tools_config.py). Kanon: ADR-0039 Harte Invariante 1 "
        "('untrusted Voice-/Inbound-Pfad bleibt no_mcp und tool-los') + "
        "ADR-0031 (fail-closed-Boden). Gegen-Test: "
        "test_create_agent_missing_config_resolves_fail_closed (asserts 0)."
    )
    def test_default_api_server_includes_terminal_toolset(self):
        """Regression #49622: desktop-only read_terminal is registered into the
        'terminal' toolset (ships in-repo), so resolve_toolset('terminal') grows
        to include it after discovery. read_terminal is NOT in the
        hermes-api-server composite, so the old all-tools subset test dropped
        'terminal' entirely. Its static membership (terminal, process) IS in the
        composite, so it must stay enabled."""
        from tools.registry import discover_builtin_tools
        from hermes_cli.tools_config import _get_platform_tools
        discover_builtin_tools()
        assert "terminal" in _get_platform_tools({}, "api_server")

    @pytest.mark.skip(
        reason="jarvis-divergence (0.18.2-Merge): upstream verlangt, dass ein in "
        "eine Configurable-Toolset registriertes Tool das Toolset auf api_server "
        "erhaelt. Unser Fork klemmt api_server UNBEDINGT auf 0 Toolsets "
        "(FAIL_CLOSED_PLATFORMS exit-clamp). Kanon: ADR-0039 Harte Invariante 1 "
        "(no_mcp/tool-los) + ADR-0031 (fail-closed-Boden). Kein realer Defekt."
    )
    def test_registering_tool_into_toolset_does_not_drop_toolset_from_inference(self):
        """Class invariant (covers the delegate_cli overlay case): registering a
        NEW tool into an existing configurable toolset must never remove that
        toolset from a platform whose composite lists the toolset's static
        tools. Synthetic registration keeps the test hermetic in CI."""
        from tools.registry import registry
        from hermes_cli.tools_config import _get_platform_tools

        sentinel = "test_sentinel_delegation_tool"
        registry.register(
            name=sentinel,
            toolset="delegation",
            schema={"name": sentinel, "description": "test",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "{}",
        )
        try:
            # delegation's static membership (delegate_task) is in the composite,
            # so the toolset must survive inference despite the extra registry tool.
            assert "delegation" in _get_platform_tools({}, "api_server"), (
                "registering a tool into 'delegation' dropped it from api_server"
            )
        finally:
            registry.deregister(sentinel)

    def test_default_off_and_restricted_toolsets_stay_off_on_api_server(self):
        """Negative contract: the static-membership comparison must NOT newly
        enable default-off or platform-restricted toolsets."""
        import os
        from unittest.mock import patch
        from hermes_cli.tools_config import _get_platform_tools
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HASS_TOKEN", None)
            os.environ.pop("XAI_API_KEY", None)
            enabled = _get_platform_tools({}, "api_server")
        assert "homeassistant" not in enabled
        assert "discord" not in enabled
        assert "discord_admin" not in enabled
        assert "x_search" not in enabled


class TestApiServerAdapterToolset:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_missing_config_resolves_fail_closed(self):
        """fail-CLOSED (ADR-0031): a missing/empty config for the untrusted Voice
        platform (api_server) must resolve to ZERO toolsets, NOT the full
        hermes-api-server default. Inversion of the prior fail-open assertion
        (was: ``len(toolsets) > 0``)."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # No platform_toolsets override — fail-closed Voice-Boden (0 Tools),
            # NOT a fallback to the full hermes-api-server toolset.
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) == 0
            assert call_kwargs.kwargs.get("platform") == "api_server"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_config_override_cannot_lift_failclosed_boden(self):
        """jarvis-Invariante (ADR-0039 Harte Invariante 1 + 7, ADR-0031):
        der untrusted Voice-/Inbound-Pfad (api_server) bleibt UNBEDINGT tool-los.
        Ein explizites ``platform_toolsets: {api_server: [...]}`` in config.yaml
        (auch via korruptes/injiziertes YAML) darf den fail-closed-Boden NICHT
        anheben -- api_server-Tools erfordern einen SEPARATEN trusted-surface-
        Adapterpfad (ADR-0039), kein stilles Toolset-Umschalten im untrusted
        Voice-Pfad. Diese Assertion ersetzt den upstream-Test
        ``test_create_agent_respects_config_override`` (der Overrides HONORIERTE);
        sie haertet die Klemme als Regressionswache statt sie nur zu skippen."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # Explicit override attempt — MUST be ignored by the fail-closed clamp.
            mock_config.return_value = {
                "platform_toolsets": {"api_server": ["web", "terminal"]}
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) == 0, (
                "api_server-Boden angehoben: explizites platform_toolsets-Override "
                "durfte den untrusted Voice-Pfad NICHT mit Tools versorgen "
                "(ADR-0039 Inv. 1/7)"
            )
            assert call_kwargs.kwargs.get("platform") == "api_server"
