"""Tests für tools/vault/db_runtime._read_dsn -- die DSN-Quelle (_FILE-Secret vs env).

Nur die DSN-Auflösung; der Pool selbst braucht psycopg (nicht im venv) und wird am Deploy
verifiziert. Der _FILE-Pfad ist security-relevant (Passwort NIE als env-Literal, Leak-Hygiene
gegen docker-inspect/Crash-Dumps/Diag-Telemetrie) -> hier fail-closed abgesichert.
"""
import pytest

from tools.vault import db_runtime as dr
from tools.vault.db_runtime import VaultPoolUnavailable


def _clear_env(monkeypatch):
    monkeypatch.delenv(dr._DSN_ENV, raising=False)
    monkeypatch.delenv(dr._DSN_FILE_ENV, raising=False)


def test_dsn_from_file_wins(monkeypatch, tmp_path):
    """_FILE gesetzt + lesbar -> DSN aus der Datei (gestrippt), auch wenn env gesetzt ist."""
    _clear_env(monkeypatch)
    f = tmp_path / "vault_db_dsn"
    f.write_text("postgresql://app@vault-db.51.jarvis.internal:5432/jarvis_vault\n")
    monkeypatch.setenv(dr._DSN_FILE_ENV, str(f))
    monkeypatch.setenv(dr._DSN_ENV, "postgresql://ENV-SHOULD-NOT-WIN/db")
    assert dr._read_dsn() == "postgresql://app@vault-db.51.jarvis.internal:5432/jarvis_vault"


def test_dsn_file_unreadable_is_fail_closed(monkeypatch, tmp_path):
    """_FILE gesetzt aber unlesbar -> VaultPoolUnavailable (NICHT still auf env fallen)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(dr._DSN_FILE_ENV, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv(dr._DSN_ENV, "postgresql://ENV-STALE/db")
    with pytest.raises(VaultPoolUnavailable):
        dr._read_dsn()


def test_dsn_env_fallback(monkeypatch):
    """Kein _FILE -> env-Fallback (v.a. Tests)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(dr._DSN_ENV, "postgresql://envonly/db")
    assert dr._read_dsn() == "postgresql://envonly/db"


def test_dsn_none_set_returns_empty(monkeypatch):
    """Weder _FILE noch env -> "" (get_vault_pool wirft dann VaultPoolUnavailable = fail-soft)."""
    _clear_env(monkeypatch)
    assert dr._read_dsn() == ""


def test_dsn_empty_file_env_ignored(monkeypatch):
    """Leerer _FILE-Env-Wert zählt als nicht-gesetzt -> env-Fallback."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(dr._DSN_FILE_ENV, "   ")
    monkeypatch.setenv(dr._DSN_ENV, "postgresql://envonly/db")
    assert dr._read_dsn() == "postgresql://envonly/db"
