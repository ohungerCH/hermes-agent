# Vendor-Sync-Pflicht (tools/vault/)

Zwei Dateien hier sind **vendored Kopien** (byte-identisch) aus dem ops-Baum. Die
getrackte **Referenz** (samt Unit-Tests) bleibt dort; die Kopie hier macht sie im
Engine-Deployable importierbar (dasselbe Vendored-Copy-Muster wie Bridge/diag-ingest/m365).

| Vendor-Kopie (hier) | Referenz (SSOT) | Tests der Referenz |
|---|---|---|
| `vault_context.py` | `ops/services/vault-db/client/vault_context.py` | `ops/services/vault-db/client/test_vault_context.py` |
| `object_store_crypto.py` | `ops/services/vault-db/crypto/object_store_crypto.py` | `ops/services/vault-db/tests/test_object_store_crypto.py` |

`vault_store.py` ist **NEU** und lebt nur hier (kein Vendoring) - es ist der INV-3-Schreiber
und gehört zum ausführenden Engine-Trunk (beim Gate).

## Drift-Check (vor jedem Commit, der eine Referenz berührt)

```
diff -q ops/services/vault-db/client/vault_context.py       runtime/hermes-main/tools/vault/vault_context.py
diff -q ops/services/vault-db/crypto/object_store_crypto.py runtime/hermes-main/tools/vault/object_store_crypto.py
```

Beide MÜSSEN identisch sein. Ein CI-Drift-Guard (Byte-Diff Referenz vs Vendor) ist der
Zielzustand (spec §2); bis dahin ist dieser Diff der manuelle Gate-Schritt. Driftet der
Krypto-/RLS-Kern still auseinander, kann eine Referenz-Fix (z.B. ein Krypto-Härtungs-Patch)
die Engine NICHT erreichen - genau die Klasse, die die GAP-D-`threat_patterns`-Vendor-Kopien
schon einmal hatte.
