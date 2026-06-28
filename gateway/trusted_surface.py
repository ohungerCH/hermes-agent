"""Trusted-surface dark-launch seam for ADR-0040.

This module prepares a separate credential, identity, and session boundary for
future trusted-surface work without changing the live untrusted
``api_server(no_mcp)`` path. The seam is intentionally inert by default and is
not mounted as a live HTTP endpoint in Wave 1.
"""

from __future__ import annotations

import hmac
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_text(value: Any, *, max_len: int = 128) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    return text[:max_len]


def _normalize_csv(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = _normalize_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _env_or_mapping(
    env: Mapping[str, str],
    mapping: Mapping[str, Any],
    env_name: str,
    key: str,
) -> Any:
    if env_name in env:
        return env.get(env_name)
    return mapping.get(key)


@dataclass(frozen=True)
class TrustedSurfaceSessionIdentity:
    """Server-authoritative scope bundle for a trusted-surface session."""

    principal_id: str
    role: str
    workspace_id: str
    tenant_id: str
    user_id: str
    owner_id: str
    device_id: str
    session_id: str
    auth_strength: str = "trusted_surface"
    allowed_toolsets: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    surface: str = "trusted_surface"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface,
            "principal_id": self.principal_id,
            "role": self.role,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "owner_id": self.owner_id,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "auth_strength": self.auth_strength,
            "allowed_toolsets": list(self.allowed_toolsets),
            "allowed_capabilities": list(self.allowed_capabilities),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        authoritative_auth_strength: str = "trusted_surface",
        authoritative_allowed_toolsets: Any = (),
        authoritative_allowed_capabilities: Any = (),
        authoritative_surface: str = "trusted_surface",
    ) -> "TrustedSurfaceSessionIdentity":
        # Server-authoritative fields are NEVER trusted back from serialized payload.
        # A caller that restores a persisted trusted-surface session must re-bind those
        # values from trusted config/runtime state instead of accepting widened scope from
        # tampered session data.
        return cls(
            principal_id=_normalize_text(data.get("principal_id")),
            role=_normalize_text(data.get("role")),
            workspace_id=_normalize_text(data.get("workspace_id")),
            tenant_id=_normalize_text(data.get("tenant_id")),
            user_id=_normalize_text(data.get("user_id")),
            owner_id=_normalize_text(data.get("owner_id")),
            device_id=_normalize_text(data.get("device_id")),
            session_id=_normalize_text(data.get("session_id"), max_len=256),
            auth_strength=_normalize_text(authoritative_auth_strength) or "trusted_surface",
            allowed_toolsets=_normalize_csv(authoritative_allowed_toolsets),
            allowed_capabilities=_normalize_csv(authoritative_allowed_capabilities),
            surface=_normalize_text(authoritative_surface) or "trusted_surface",
        )


@dataclass(frozen=True)
class TrustedSurfaceConfig:
    """Dark-launch config for the future trusted-surface adapter seam."""

    enabled: bool = False
    credential: str = ""
    principal_id: str = ""
    role: str = ""
    workspace_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    owner_id: str = ""
    device_id: str = ""
    auth_strength: str = "trusted_surface"
    allowed_toolsets: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    config_error: Optional[str] = None

    @classmethod
    def from_sources(
        cls,
        raw: Any = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        api_server_key: str = "",
    ) -> "TrustedSurfaceConfig":
        env_map = env or os.environ
        mapping = raw if isinstance(raw, Mapping) else {}

        enabled = _coerce_bool(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_ENABLED", "enabled"),
            False,
        )
        credential = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_CREDENTIAL", "credential"),
            max_len=512,
        )
        principal_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_PRINCIPAL_ID", "principal_id"),
            max_len=256,
        )
        role = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_ROLE", "role"),
        )
        workspace_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_WORKSPACE_ID", "workspace_id"),
        )
        tenant_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_TENANT_ID", "tenant_id"),
        )
        user_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_USER_ID", "user_id"),
        )
        owner_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_OWNER_ID", "owner_id"),
        )
        device_id = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_DEVICE_ID", "device_id"),
        )
        auth_strength = _normalize_text(
            _env_or_mapping(env_map, mapping, "TRUSTED_SURFACE_AUTH_STRENGTH", "auth_strength"),
        ) or "trusted_surface"
        allowed_toolsets = _normalize_csv(
            _env_or_mapping(
                env_map,
                mapping,
                "TRUSTED_SURFACE_ALLOWED_TOOLSETS",
                "allowed_toolsets",
            )
        )
        allowed_capabilities = _normalize_csv(
            _env_or_mapping(
                env_map,
                mapping,
                "TRUSTED_SURFACE_ALLOWED_CAPABILITIES",
                "allowed_capabilities",
            )
        )

        config_error: Optional[str] = None
        if enabled:
            required = (
                credential,
                principal_id,
                role,
                workspace_id,
                tenant_id,
                user_id,
                owner_id,
                device_id,
            )
            if not all(required):
                config_error = "trusted_surface_missing_identity_fields"
            elif api_server_key and hmac.compare_digest(credential, api_server_key):
                config_error = "trusted_surface_credential_matches_api_server_key"

        return cls(
            enabled=enabled,
            credential=credential,
            principal_id=principal_id,
            role=role,
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            owner_id=owner_id,
            device_id=device_id,
            auth_strength=auth_strength,
            allowed_toolsets=allowed_toolsets,
            allowed_capabilities=allowed_capabilities,
            config_error=config_error,
        )

    @property
    def ready(self) -> bool:
        return self.enabled and not self.config_error


class TrustedSurfaceAuthError(Exception):
    """Raised when the dark trusted-surface seam rejects a credential."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class TrustedSurfaceAdapterSeam:
    """Separate dark-launch adapter seam for future trusted-surface requests."""

    _IDENTITY_DIMENSIONS = [
        "principal_id",
        "role",
        "workspace_id",
        "tenant_id",
        "user_id",
        "owner_id",
        "device_id",
        "session_id",
        "auth_strength",
        "allowed_toolsets",
        "allowed_capabilities",
    ]

    def __init__(self, config: TrustedSurfaceConfig):
        self._config = config

    @property
    def config(self) -> TrustedSurfaceConfig:
        return self._config

    def describe_public(self) -> Dict[str, Any]:
        return {
            "dark_launch": True,
            "live_endpoint": False,
            "enabled": self._config.enabled,
            "ready": self._config.ready,
            "config_error": self._config.config_error,
            "separate_credential_required": True,
            "separate_from_api_server_key": True,
            "reserved_adapter_path": None,
            "session_model": {
                "server_authoritative": True,
                "separate_session_boundary": True,
                "identity_dimensions": list(self._IDENTITY_DIMENSIONS),
            },
        }

    def authenticate_bearer(
        self,
        authorization_header: str,
        *,
        requested_device_id: Optional[str] = None,
    ) -> TrustedSurfaceSessionIdentity:
        if not self._config.ready:
            raise TrustedSurfaceAuthError(self._config.config_error or "trusted_surface_not_ready")

        token = ""
        if authorization_header.startswith("Bearer "):
            token = authorization_header[7:].strip()
        if not token:
            raise TrustedSurfaceAuthError("missing_trusted_surface_bearer")
        if not hmac.compare_digest(token, self._config.credential):
            raise TrustedSurfaceAuthError("invalid_trusted_surface_credential")

        inbound_device_id = _normalize_text(requested_device_id)
        if inbound_device_id and inbound_device_id != self._config.device_id:
            raise TrustedSurfaceAuthError("trusted_surface_device_mismatch")

        return TrustedSurfaceSessionIdentity(
            principal_id=self._config.principal_id,
            role=self._config.role,
            workspace_id=self._config.workspace_id,
            tenant_id=self._config.tenant_id,
            user_id=self._config.user_id,
            owner_id=self._config.owner_id,
            device_id=self._config.device_id,
            session_id=f"trusted_surface_{uuid.uuid4().hex[:24]}",
            auth_strength=self._config.auth_strength,
            allowed_toolsets=self._config.allowed_toolsets,
            allowed_capabilities=self._config.allowed_capabilities,
        )
