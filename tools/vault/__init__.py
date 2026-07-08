"""Vault-Schreibpfad (Stufe 5, INV-3 single-writer) im Engine-Trunk.

Dies ist das ausführende Zuhause des VaultStore: der EINZIGE Schreiber, der das
Schreib-Gate erzwingt (INV-3). Er sitzt beim Gate (tools/write_approval.py), nicht
über eine Repo-Naht danach.

VENDORING (spec ops/services/vault-db/VAULTSTORE_WRITE_PATH_SPEC.md §2):
  * vault_context.py       <- ops/services/vault-db/client/vault_context.py
  * object_store_crypto.py <- ops/services/vault-db/crypto/object_store_crypto.py
Die getrackte Referenz + die Tests dieser zwei Module bleiben im ops-Baum. JEDE
Änderung an der Referenz MUSS byte-genau hierher propagiert werden (Drift-Check
siehe VENDOR_SYNC.md). vault_store.py ist NEU und lebt nur hier.
"""
