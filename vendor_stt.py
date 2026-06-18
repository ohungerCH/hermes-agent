#!/usr/bin/env python3
"""Vendoring-Mechanik fuer stt_redaction in den Hermes-Engine-Fork (#2c).

Vorbild: ops/services/media-discovery/vendor_stt.py (BUILD_SPEC 7.3, Option B).
stt-redaction hat keinen eigenen Container; seine Bibliothek wird nach
agent/_vendor/stt_redaction kopiert und im Image als ``agent._vendor.stt_redaction``
importiert (die Quelle ops/services/stt-redaction/ liegt NICHT im Engine-Image ->
der Drift-Check laeuft im Repo/CI, nicht zur Laufzeit).

WARUM Vendoring statt Patterns-Port: die Lib ist die SSOT (Luhn/mod-97/E.164-
Pruefsummen, fail-closed, leak-sicherer Report). Ein Pattern-Port nach
agent/redact.py wuerde driften und die Pruefsummen-Logik duplizieren. Vendoring
spiegelt media-discovery 1:1 und haelt EINE Quelle.

  vendor_stt.py --sync    kopiert Quelle -> Kopie (regeneriert die vendored copy)
  vendor_stt.py --check   sha256 Quelle-vs-Kopie; exit 1 bei Drift/fehlend (fail-closed, CI-Gate)

git-tracken: die Kopie (agent/_vendor/) wird committet (reproduzierbarer Build-
Kontext: COPY . . zieht sie ins Image); --check faengt einen Drift, falls die
Quelle ohne Re-Sync geaendert wird.

QUELLE: ein Worktree-Fork sieht ops/services/stt-redaction NICHT relativ zu sich
(der Fork ist ein eigener Checkout unter .phase2-worktree). Der Pfad wird darum
ueber JARVIS_STT_REDACTION_SRC ueberschreibbar gemacht und faellt sonst auf den
bekannten /srv-Standort zurueck.
"""
from __future__ import annotations
import hashlib, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Quelle: ENV-Override zuerst (CI/portabilitaet), sonst der /srv-Standort der Lib.
_ENV_SRC = os.environ.get("JARVIS_STT_REDACTION_SRC")
SRC = (
    Path(_ENV_SRC).resolve()
    if _ENV_SRC
    else Path("/srv/services/jarvis/ops/services/stt-redaction/stt_redaction").resolve()
)
DST = HERE / "agent" / "_vendor" / "stt_redaction"
VENDOR_INIT = HERE / "agent" / "_vendor" / "__init__.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _py_files(root: Path):
    return sorted(f.relative_to(root) for f in root.rglob("*.py"))


def sync() -> int:
    if not SRC.is_dir():
        print(f"FATAL: Quelle fehlt: {SRC} (setze JARVIS_STT_REDACTION_SRC)"); return 2
    if DST.exists():
        shutil.rmtree(DST)
    # __pycache__/.pyc nie vendoren: generiert + python-version-spezifisch (Drift-Rauschen).
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    VENDOR_INIT.parent.mkdir(parents=True, exist_ok=True)
    VENDOR_INIT.write_text(
        "# Vendoring-Paket-Marker (#2c, Vorbild BUILD_SPEC 7.3). Inhalt via\n"
        "# vendor_stt.py --sync, Drift-gesichert via --check. NICHT von Hand editieren.\n"
    )
    print(f"synced {len(_py_files(SRC))} files: {SRC} -> {DST}")
    return 0


def check() -> int:
    if not SRC.is_dir():
        print(f"FATAL: Quelle fehlt: {SRC} (setze JARVIS_STT_REDACTION_SRC)"); return 2
    if not DST.is_dir():
        print(f"DRIFT: Kopie fehlt: {DST} (run --sync)"); return 1
    src_files, dst_files = _py_files(SRC), _py_files(DST)
    if src_files != dst_files:
        print(f"DRIFT: Dateimenge weicht ab\n  nur Quelle: {set(src_files)-set(dst_files)}\n  nur Kopie: {set(dst_files)-set(src_files)}")
        return 1
    bad = [str(f) for f in src_files if _sha(SRC / f) != _sha(DST / f)]
    if bad:
        print(f"DRIFT: Inhalt weicht ab in: {bad} (run --sync)"); return 1
    print(f"OK: {len(src_files)} vendored files sha256-identisch zur Quelle")
    return 0


if __name__ == "__main__":
    if "--sync" in sys.argv: sys.exit(sync())
    if "--check" in sys.argv: sys.exit(check())
    print(__doc__); sys.exit(64)
