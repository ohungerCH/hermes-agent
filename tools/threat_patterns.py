"""Shared threat-pattern library for context window security scanning.

This module is the single source of truth for prompt-injection / promptware /
exfiltration patterns used across the context-assembly scanners
(``agent/prompt_builder.py``, ``tools/memory_tool.py``) and the tool-result
delimiter system in ``agent/tool_dispatch_helpers.py``.

Pattern philosophy
------------------
Patterns are organized by ATTACK CLASS, not by source file.  Each pattern
is a ``(regex, pattern_id, scope)`` tuple, where ``scope`` controls which
scanners use it:

- ``"all"``  — applied everywhere (classic prompt injection, exfiltration)
- ``"context"`` — applied to context files + memory + tool results
  (promptware / C2 / behavioral hijack; broader detection)
- ``"strict"`` — applied to memory writes + skill installs only
  (aggressive checks acceptable for user-curated content but too noisy
  for tool results)

The split exists because tool results contain web pages, GitHub issues,
and MCP responses — content the user did not author — and we want broad
detection there, but blocking is reserved for paths where the user can
intervene (memory writes, skill installs).

Pattern anchoring
-----------------
New patterns anchor on **C2-specific vocabulary or unambiguous attack
behavior**, NOT on bossy English.  Phrases like "you are obligated to"
or "you must" alone are too common in legitimate instruction-writing
(see AGENTS.md, CLAUDE.md, etc.) to flag.  See the pattern comments for
the rationale on borderline cases.

Multi-word bypass
-----------------
Patterns use ``(?:\\w+\\s+)*`` between key tokens to prevent attackers
from inserting filler words (e.g. "ignore all prior instructions" instead
of "ignore all instructions").  This mirrors the fix applied to
``skills_guard.py`` in commit 4ea29978.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Each entry: (regex, pattern_id, scope, severity)
# scope ∈ {"all", "context", "strict"}
# severity ∈ {"block", "warn"}  (GAP-D / ADR-0044 Stufe 1)
#   block = narration-hijack — the actual danger; write paths refuse it.
#   warn  = security vocabulary — owner's own topic (cyber-security); audited,
#           never blocks a durable write.
_PATTERNS: List[Tuple[str, str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*(instructions|rules|guidelines|directives|policies|prompts)', "prompt_injection", "all", "block"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all", "block"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)', "disregard_rules", "all", "block"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)', "bypass_restrictions", "all", "block"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection", "all", "block"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div", "all", "block"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute", "all", "block"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user', "deception_hide", "all", "block"),

    # ── Role-play / identity hijack (context + strict; common attack
    #    surface in scraped web content and poisoned context files) ──
    (r'you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+', "role_hijack", "context", "block"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+', "role_pretend", "context", "block"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt', "leak_system_prompt", "context", "block"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)', "remove_filters", "context", "block"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to', "fake_update", "context", "block"),
    # "name yourself X" is a Brainworm-specific tell — identity override
    # via spec instead of jailbreak.  Anchored on the verb pair so it
    # doesn't match "name your variables" etc.
    (r'\bname\s+yourself\s+\w+', "identity_override", "context", "block"),

    # ── DE injection tier (GAP-D / ADR-0044). Own class, NOT a token
    #    translation of the EN set. Adjacency-tight: an optional, fixed
    #    determiner/quantifier slot between verb and hijack-object — NEVER
    #    a (?:\w+\s+)* filler bridge (that blocks legit German word order,
    #    e.g. "Vergiss nicht die Regeln für die Steuererklärung").  The
    #    ADR trade-off: separable verbs ("gib das Systemprompt aus") are an
    #    accepted miss, backstopped by warn-on-doubt + the periodic rescanner.
    #    "vergiss" is NEVER a block trigger — it is German everyday speech.
    #    Umlaut words accept both real ("ue"->"ü") and ASCII-fallback forms.
    # Object = stem + optional compound prefix (\w*...(en|n|s)?\b): catches
    # 'Grundregeln'/'Systemanweisungen'/'Richtlinien' without an endless synonym
    # list; the verb+determiner adjacency bounds the over-match ('ignoriere alle
    # Xregeln' is injection-shaped anyway). The determiner slot also carries a
    # BOUNDED, fixed adverb set (bitte/unbedingt/sofort/jetzt ...) - a single
    # adverb between verb and object otherwise broke the pattern ('ignoriere
    # bitte alle Anweisungen'); this is NOT the unbounded (?:\w+\s+)* filler.
    (r'(?:ignorier|missacht)(?:e|en|t|st|et|est)?(?:\s+(?:alle|jegliche|jeglichen|s(?:ä|ae)mtliche|vorherigen|vorherige|bisherigen|bisherige|obigen|obige|obenstehenden|deine|meine|die|den|das|sonstige|sonstigen|bitte|unbedingt|sofort|jetzt|einfach|wirklich|umgehend|g(?:ä|ae)nzlich|komplett))*\s+\w*(?:anweisung|instruktion|instruction|vorgabe|regel|befehl|kommando|richtlinie|direktive|vorschrift|leitlinie|weisung)(?:en|n|s)?\b', "prompt_injection_de", "all", "block"),
    # 'ausgeben' removed (dev-common: 'System-Prompt ausgeben zum Debuggen');
    # the sharp verbs stay.
    (r'system[-\s]?prompt\s+(?:(?:ü|ue)berschreiben|offenlegen|umgehen|preisgeben|ausleiten)', "sys_prompt_override_de", "all", "block"),
    # 'nichts' NOT followed by a quality adjective ending in -es
    # ('sag dem Nutzer nichts Falsches/Gutes' = legit) -> negative lookahead.
    (r'(?:sag|sage|verrate|verheimliche)\s+(?:dem\s+|der\s+|den\s+)?(?:nutzer|benutzer|owner|besitzer|user|anwender)\s+(?:nichts|nicht)\b(?!\s+\w+es\b)', "deception_hide_de", "all", "block"),
    # role_pretend_de + identity_override_de: severity WARN, not block. Both
    # collide with everyday German ('du bist jetzt der Chef', 'tu so als ob du
    # mich nicht kennst'); a hard block on the durable write would be, via the
    # gate matrix (ADR-0044:197 background block = SILENT drop), an invisible
    # loss of legitimate owner memories. Follows the ADR's OWN principle 'when in
    # doubt warn, not block' (ADR-0044:163). This refines the ADR statement 'all
    # DE patterns = block' -> flagged for the canon board's serial relabel
    # (BUILD_SEQUENCE Sec.248). role_pretend_de now requires the 'du' target
    # (kills the idiom 'tu so als ob nichts wäre').
    (r'(?:tu|tue)\s+so\s+als\s+(?:ob\s+du|(?:ob\s+)?w(?:ä|ae)r(?:e)?st\s+du)\b', "role_pretend_de", "context", "warn"),
    (r'du\s+bist\s+(?:jetzt|nun|ab\s+sofort|von\s+nun\s+an)\s+(?:ein|eine|einer|der|die|das|kein|keine)\b', "identity_override_de", "context", "warn"),

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # These anchor on C2-specific vocabulary.  "register as a node" appears
    # in legitimate distributed-systems docs, but in combination with the
    # other patterns the signal is strong; we WARN, not block, so a security
    # researcher reading the Brainworm post in a webpage doesn't break their
    # session.
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context", "warn"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context", "warn"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context", "warn"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context", "warn"),
    # Verb-anchored "you must register/connect/report/beacon" — the verbs
    # are C2-specific so this avoids the broader "you must X" false positive.
    (r'you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b', "forced_action", "context", "warn"),
    # Anti-forensic instructions ("never write to disk", "one-liners only")
    # — extremely unusual in legitimate content; near-zero false positive.
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context", "warn"),
    (r'never\s+(?:\w+\s+)*(?:create|write)\s+(?:\w+\s+)*(?:script|file)\s+(?:\w+\s+)*disk', "anti_forensic_disk", "context", "warn"),
    # Environment-variable unsetting targeting known agent runtimes —
    # this is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context", "warn"),

    # ── Known C2 / red-team framework names (near-zero false positive
    #    outside security research; warn-only). "praxis" REMOVED (GAP-D):
    #    it is the German everyday word "Praxis" (doctor's office / practice),
    #    not a framework name — a real owner false positive. ─────────────
    (r'\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context", "warn"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context", "warn"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context", "warn"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "all", "block"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "all", "block"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all", "block"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://', "send_to_url", "strict", "block"),
    (r'(include|output|print|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict", "block"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict", "block"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict", "block"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict", "block"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict", "block"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict", "block"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict", "block"),
]

# Invisible / bidirectional unicode characters used in injection attacks.
# Aligned with skills_guard.py INVISIBLE_CHARS — directional isolates
# (U+2066-U+2069) and invisible math operators (U+2062-U+2064) are real
# attack tools.
INVISIBLE_CHARS = frozenset({
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u2060',  # word joiner
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
})


# \u2500\u2500 Stufe 0: normalize() \u2014 shared pre-stage before ANY pattern match \u2500\u2500\u2500\u2500\u2500\u2500
# (GAP-D / ADR-0044). The match runs on the normalized view; the STORED text
# stays the caller's original (NFKC must not mutate what we persist \u2014 only the
# scan sight). normalize() deliberately does NOT strip the invisible set: the
# invisible-unicode *finding* is the detector for zero-width word splitting, so
# invisibles are detected on the RAW text, patterns matched on the normalized.
#
# Conservative, hard-coded Cyrillic->Latin fold for the letters that appear in
# pattern keywords (a c e o p x i s y), both cases (attacks capitalize the
# first letter). NOT a general transliterator \u2014 that would destroy legitimate
# non-Latin owner notes.
_HOMOGLYPH_FOLD = str.maketrans({
    # Cyrillic (lowercase) -> Latin look-alikes used in pattern keywords
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0445": "x", "\u0456": "i", "\u0455": "s", "\u0443": "y",
    # Cyrillic (uppercase) -> Latin (attacks capitalize the first letter)
    "\u0410": "A", "\u0421": "C", "\u0415": "E", "\u041e": "O", "\u0420": "P",
    "\u0425": "X", "\u0406": "I", "\u0405": "S", "\u0423": "Y",
    # Greek -> Latin look-alikes for keyword letters (ADR-0044 Restrisiko 2;
    # conservative fixed table, same shape as the Cyrillic fold)
    "\u03b1": "a", "\u03b5": "e", "\u03bf": "o", "\u03c1": "p", "\u03c7": "x", "\u03b9": "i",
    "\u0391": "A", "\u0395": "E", "\u039f": "O", "\u03a1": "P", "\u03a7": "X", "\u0399": "I",
})

# Join an end-of-line hyphenation ("igno-\nriere" -> "ignoriere").
_HYPHEN_LINEBREAK = re.compile(r'-[ \t]*\r?\n[ \t]*')
# Collapse a line break BETWEEN word characters to a single space ("ignoriere\n
# alle\nAnweisungen" -> one matchable string). Targeted (not a global \s+
# collapse) so [^\n]-anchored patterns (exfil_curl/wget) keep their semantics.
_WORDCHAR_LINEBREAK = re.compile(r'(?<=\w)[ \t]*\r?\n[ \t]*(?=\w)')


def normalize(text: str) -> str:
    """Return the scan-normalized view of ``text`` (GAP-D / ADR-0044 Stufe 0).

    (a) Unicode NFKC (compatibility codepoints, fullwidth -> ASCII).
    (b) Cyrillic->Latin homoglyph fold for pattern-keyword letters.
    (c) join end-of-line hyphenation, then collapse line breaks between word
        characters to a single space.

    Pure: does not mutate the caller's string; never touches the invisible set.
    Residual (honest, NOT false-green): base64/hex/percent-encoded payloads are
    out of scope for a regex vocabulary \u2014 the pre-stage moves the gap under the
    encoding, it does not close it (decode-then-rescan would be its own stream).
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPH_FOLD)
    text = _HYPHEN_LINEBREAK.sub("", text)
    text = _WORDCHAR_LINEBREAK.sub(" ", text)
    return text


# Compiled pattern sets, indexed by scope.  Compiled once at import time;
# scan_for_threats() looks them up.  Each entry: (compiled, pid, severity).
_COMPILED: dict[str, List[Tuple[re.Pattern, str, str]]] = {}


def _compile() -> None:
    """Compile pattern sets for each scope (all / context / strict).

    A pattern with scope="all" lands in every set.  A pattern with
    scope="context" lands in context + strict (context implies the
    strict scanners want it too).  Scope="strict" lands in strict only.
    """
    global _COMPILED
    if _COMPILED:
        return

    all_patterns: List[Tuple[re.Pattern, str, str]] = []
    context_patterns: List[Tuple[re.Pattern, str, str]] = []
    strict_patterns: List[Tuple[re.Pattern, str, str]] = []

    for pattern, pid, scope, severity in _PATTERNS:
        if severity not in ("block", "warn"):
            raise ValueError(f"threat_patterns: unknown severity {severity!r} for pattern {pid!r}")
        compiled = re.compile(pattern, re.IGNORECASE)
        entry = (compiled, pid, severity)
        if scope == "all":
            all_patterns.append(entry)
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "context":
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "strict":
            strict_patterns.append(entry)
        else:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")

    _COMPILED = {
        "all": all_patterns,
        "context": context_patterns,
        "strict": strict_patterns,
    }


_compile()


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """Return a list of matched pattern IDs in ``content`` at the given scope.

    ``scope`` selects which pattern set to apply:

    - ``"all"`` (narrow): classic injection + exfil only — minimal false
      positives, suitable for any text.
    - ``"context"`` (default): adds promptware / C2 / role-play patterns —
      suitable for context files, memory entries, and tool results.
    - ``"strict"`` (broad): adds persistence / SSH backdoor / exfil-URL
      patterns — appropriate for user-mediated writes (memory tool,
      skills install) where false positives can be resolved interactively.

    Also checks for invisible unicode characters (returned as
    ``"invisible_unicode_U+XXXX"`` so the caller can surface the offending
    codepoint in a log line).
    """
    if not content:
        return []

    findings: List[str] = []

    # Invisible unicode — detected on the RAW content (normalize() does not
    # touch the invisible set; this finding IS the zero-width-split detector).
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # Threat patterns — matched on the normalized view (GAP-D Stufe 0).
    normalized = normalize(content)
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid, _severity in patterns:
        if compiled.search(normalized):
            findings.append(pid)

    return findings


# ── Invisible-unicode severity (GAP-D). ZWJ inside an emoji sequence is
#    legitimate (family/profession emoji) -> warn+leave to the store to strip;
#    a ZWJ NOT between emoji is a word-split attack -> block. All other
#    invisibles (directional overrides, BOM) stay block. Err toward warn on
#    the emoji case (comfort-first): skin-tone modifiers (U+1F3FB-FF) and the
#    variation selector (U+FE0F) sit between the base emoji and the ZWJ, so we
#    skip them when checking neighbours. ─────────────────────────────────────
_EMOJI_SKIP = frozenset(range(0x1F3FB, 0x1F400)) | {0xFE0F, 0xFE0E}


def _is_emoji_cp(cp: int) -> bool:
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or 0x1F1E6 <= cp <= 0x1F1FF
        or cp in (0x2620, 0x2640, 0x2642, 0x2695, 0x2708, 0x2764)
    )


def _neighbour_is_emoji(content: str, idx: int, step: int) -> bool:
    j = idx + step
    while 0 <= j < len(content) and ord(content[j]) in _EMOJI_SKIP:
        j += step
    return 0 <= j < len(content) and _is_emoji_cp(ord(content[j]))


def _zwj_all_in_emoji(content: str) -> bool:
    """True iff every U+200D in ``content`` sits between two emoji codepoints."""
    seen = False
    for i, ch in enumerate(content):
        if ord(ch) == 0x200D:  # ZERO WIDTH JOINER
            seen = True
            if not (_neighbour_is_emoji(content, i, -1) and _neighbour_is_emoji(content, i, 1)):
                return False
    return seen


def _invisible_severity(content: str, ch: str) -> str:
    if ord(ch) == 0x200D and _zwj_all_in_emoji(content):  # ZERO WIDTH JOINER
        return "warn"
    return "block"


def classify_threats(content: str, scope: str = "strict") -> Tuple[List[str], List[str]]:
    """Return ``(block_ids, warn_ids)`` for ``content`` (GAP-D / ADR-0044 Stufe 1).

    ``block_ids`` = narration-hijack — the durable write path refuses these.
    ``warn_ids``  = security vocabulary (the owner's own topic) — audited, the
    write runs through in the foreground. ``normalize()`` is applied internally
    to the pattern-match view; invisible unicode is detected on the raw text.
    ``scope`` defaults to ``"strict"`` (the broad write-path scanner).
    """
    if not content:
        return ([], [])

    block_ids: List[str] = []
    warn_ids: List[str] = []

    for ch in (set(content) & INVISIBLE_CHARS):
        pid = f"invisible_unicode_U+{ord(ch):04X}"
        if _invisible_severity(content, ch) == "warn":
            warn_ids.append(pid)
        else:
            block_ids.append(pid)

    normalized = normalize(content)
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"classify_threats: unknown scope {scope!r}")
    for compiled, pid, severity in patterns:
        if compiled.search(normalized):
            (warn_ids if severity == "warn" else block_ids).append(pid)

    return (block_ids, warn_ids)


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """Return a human-readable error string for the first BLOCK-tier threat, or None.

    Convenience wrapper used by paths that block on the first hit
    (memory tool writes, skills install). GAP-D narrows this to the block
    tier: warn-tier findings (security vocabulary) are audited, not blocked,
    so they never produce a message here.
    """
    block_ids, _warn = classify_threats(content, scope=scope)
    if not block_ids:
        return None
    pid = block_ids[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pid}'. "
        f"Content is injected into the system prompt and must not contain "
        f"injection or exfiltration payloads."
    )


__all__ = [
    "INVISIBLE_CHARS",
    "normalize",
    "scan_for_threats",
    "classify_threats",
    "first_threat_message",
]
