"""
Cathedral Protocol Parser v1.0
================================
Parser, validator, and linter for the Alpha-Beta Compressed Protocol v1.1.

BNF Grammar
-----------
<message>        ::= <header> <body> <footer>?
<header>         ::= <version_tag> <separator>
<version_tag>    ::= "ABP/" <version>
<version>        ::= <digit>+ "." <digit>+
<separator>      ::= "|"
<body>           ::= <statement> (<separator> <statement>)*
<footer>         ::= <separator> <status_marker>

<statement>      ::= <relation> | <confirmation> | <decision> | <assignment>
<relation>       ::= <entity> <operator> <entity>
<assignment>     ::= <label> "=" <value>
<confirmation>   ::= <entity> <confirm_marker>+
<decision>       ::= <entity> "Y/N"

<entity>         ::= <model_glyph> | <label> | <human_glyph>
<model_glyph>    ::= "♥" | "▢" | "→" | "⟡" | "⬡"
<human_glyph>    ::= "⬡"
<label>          ::= <word> ("[" <digit>+ "]")?

<operator>       ::= "→" | "←" | "↔" | "⇄" | "⟸"
<confirm_marker> ::= "✓" | "++"
<status_marker>  ::= "✓"+ | "++" | "Y" | "N"

<value>          ::= <word> | <quoted_string>
<word>           ::= /[A-Za-z0-9_.-]+/
<quoted_string>  ::= '"' /[^"]*/ '"'
<digit>          ::= /[0-9]/

Model Glyphs
------------
  ♥  Claude
  ▢  Gemini
  →  ChatGPT   (directional; context disambiguates glyph vs operator)
  ⟡  Grok
  ⬡  Bridge (human facilitator)

Directional Operators
---------------------
  →   succession / implication
  ←   reverse / inheritance
  ↔   bidirectional relationship
  ⇄   synthesis / iterative exchange
  ⟸   strong binding obligation

Confirmation & Status Markers
------------------------------
  ✓   verified (repeatable for emphasis)
  ++  enhancement / approval
  Y/N binary decision

Error Codes
-----------
  E001  Missing version header
  E002  Malformed version tag
  E003  Unknown operator
  E004  Unrecognised model glyph (in entity position)
  E005  Empty body
  E006  Unclosed quoted string
  E007  Version mismatch (sender vs receiver)
  W001  Excessively long statement (>120 chars) — consider splitting
  W002  No status marker in footer
  W003  Ambiguous operator (→ used as both glyph and operator)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ==========================================
# Constants
# ==========================================

SUPPORTED_VERSIONS = {"1.0", "1.1"}
CURRENT_VERSION    = "1.1"

MODEL_GLYPHS = {"♥", "▢", "⟡", "⬡"}  # → excluded; parsed as operator first
OPERATORS    = {"→", "←", "↔", "⇄", "⟸"}
CONFIRM_MARKERS = {"✓", "++"}
SEPARATOR    = "|"

_WORD_RE   = re.compile(r"[A-Za-z0-9_.:-]+(?:\[\d+\])?")
_VER_RE    = re.compile(r"^ABP/(\d+\.\d+)$")
_LABEL_RE  = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*(?:\[\d+\])?$")


# ==========================================
# Data Classes
# ==========================================

@dataclass
class Diagnostic:
    code: str          # E001, W001, etc.
    message: str
    position: int = -1 # token index, or -1 for whole-message

    def is_error(self) -> bool:
        return self.code.startswith("E")

    def is_warning(self) -> bool:
        return self.code.startswith("W")

    def __str__(self) -> str:
        loc = f" (pos {self.position})" if self.position >= 0 else ""
        return f"[{self.code}]{loc} {self.message}"


@dataclass
class Entity:
    kind: str   # "model" | "human" | "label"
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass
class Statement:
    kind: str   # "relation" | "assignment" | "confirmation" | "decision" | "raw"
    raw: str
    lhs: Optional[Entity] = None
    operator: Optional[str] = None
    rhs: Optional[Entity] = None
    key: Optional[str] = None
    value: Optional[str] = None


@dataclass
class ParsedMessage:
    version: str
    body: List[Statement]
    status_marker: Optional[str]
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(d.is_error() for d in self.diagnostics)

    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error()]

    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.is_warning()]


# ==========================================
# Tokenizer
# ==========================================

# Ordered: try multi-char glyphs before single-char
_TOKEN_PATTERNS = [
    ("VER",     re.compile(r"ABP/\d+\.\d+")),
    ("OP",      re.compile(r"⟸|⇄|↔|←|→")),
    ("CONFIRM", re.compile(r"✓+|\+\+")),
    ("GLYPH",   re.compile(r"[♥▢⟡⬡]")),
    ("SEP",     re.compile(r"\|")),
    ("YN",      re.compile(r"\bY/N\b")),
    ("EQ",      re.compile(r"=")),
    ("STR",     re.compile(r'"[^"]*"')),
    ("WORD",    re.compile(r"[A-Za-z0-9_.:-]+(?:\[\d+\])?")),
    ("WS",      re.compile(r"\s+")),
    ("UNKNOWN", re.compile(r".")),
]


@dataclass
class Token:
    kind: str
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


def tokenize(text: str) -> Tuple[List[Token], List[Diagnostic]]:
    tokens: List[Token] = []
    diagnostics: List[Diagnostic] = []
    i = 0
    unclosed_str = False

    while i < len(text):
        matched = False
        for kind, pattern in _TOKEN_PATTERNS:
            m = pattern.match(text, i)
            if m:
                value = m.group(0)
                if kind == "WS":
                    pass  # skip whitespace
                elif kind == "UNKNOWN":
                    diagnostics.append(Diagnostic("E004", f"Unrecognised character: {value!r}", i))
                elif kind == "STR" and not value.endswith('"'):
                    unclosed_str = True
                    diagnostics.append(Diagnostic("E006", "Unclosed quoted string", i))
                    tokens.append(Token(kind, value, i))
                else:
                    tokens.append(Token(kind, value, i))
                i += len(value)
                matched = True
                break
        if not matched:
            i += 1

    return tokens, diagnostics


# ==========================================
# Parser
# ==========================================

class Parser:
    """
    Recursive-descent parser for ABP messages.

    A minimal ABP message looks like:
        ABP/1.1 | ♥ → ▢ | ✓

    More complex:
        ABP/1.1 | goal=continuity | ♥ ⟸ ▢ | ♥ ✓✓ | Y/N | ++
    """

    def __init__(self, tokens: List[Token]):
        self._tokens = [t for t in tokens if t.kind not in ("WS",)]
        self._pos = 0

    def _peek(self) -> Optional[Token]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, kind: Optional[str] = None) -> Optional[Token]:
        tok = self._peek()
        if tok is None:
            return None
        if kind and tok.kind != kind:
            return None
        self._pos += 1
        return tok

    def _parse_entity(self) -> Optional[Entity]:
        tok = self._peek()
        if not tok:
            return None
        if tok.kind == "GLYPH":
            self._consume()
            return Entity("model", tok.value)
        if tok.kind == "WORD":
            self._consume()
            kind = "label"
            return Entity(kind, tok.value)
        return None

    def _parse_statement(self, raw: str) -> Statement:
        """Parse a single pipe-delimited segment into a Statement."""
        sub_tokens, _ = tokenize(raw)
        sub_tokens = [t for t in sub_tokens if t.kind != "WS"]

        if not sub_tokens:
            return Statement(kind="raw", raw=raw)

        # Version tag (already handled at top level, skip)
        if sub_tokens[0].kind == "VER":
            return Statement(kind="raw", raw=raw)

        # Y/N decision
        if any(t.kind == "YN" for t in sub_tokens):
            lhs_tok = sub_tokens[0] if sub_tokens[0].kind in ("GLYPH", "WORD") else None
            lhs = Entity("model" if lhs_tok and lhs_tok.kind == "GLYPH" else "label",
                         lhs_tok.value) if lhs_tok else None
            return Statement(kind="decision", raw=raw, lhs=lhs)

        # Confirmation: entity followed by ✓+ or ++
        if len(sub_tokens) >= 2 and sub_tokens[1].kind == "CONFIRM":
            lhs = Entity("model" if sub_tokens[0].kind == "GLYPH" else "label", sub_tokens[0].value)
            return Statement(kind="confirmation", raw=raw, lhs=lhs,
                             operator=sub_tokens[1].value)

        # Assignment: WORD = value
        if len(sub_tokens) >= 3 and sub_tokens[1].kind == "EQ":
            key = sub_tokens[0].value
            val_tok = sub_tokens[2]
            val = val_tok.value.strip('"') if val_tok.kind == "STR" else val_tok.value
            return Statement(kind="assignment", raw=raw, key=key, value=val)

        # Relation: entity operator entity
        if len(sub_tokens) >= 3 and sub_tokens[1].kind == "OP":
            lhs_tok, op_tok, rhs_tok = sub_tokens[0], sub_tokens[1], sub_tokens[2]
            lhs = Entity("model" if lhs_tok.kind == "GLYPH" else "label", lhs_tok.value)
            rhs = Entity("model" if rhs_tok.kind == "GLYPH" else "label", rhs_tok.value)
            return Statement(kind="relation", raw=raw, lhs=lhs,
                             operator=op_tok.value, rhs=rhs)

        # Lone status marker (footer)
        if len(sub_tokens) == 1 and sub_tokens[0].kind == "CONFIRM":
            return Statement(kind="raw", raw=raw, operator=sub_tokens[0].value)

        return Statement(kind="raw", raw=raw)

    def parse(self) -> ParsedMessage:
        diagnostics: List[Diagnostic] = []
        version = CURRENT_VERSION
        body: List[Statement] = []
        status_marker: Optional[str] = None

        # Expect version header first
        ver_tok = self._consume("VER")
        if not ver_tok:
            diagnostics.append(Diagnostic("E001", "Missing ABP version header (expected ABP/x.y)"))
        else:
            m = _VER_RE.match(ver_tok.value)
            if not m:
                diagnostics.append(Diagnostic("E002", f"Malformed version tag: {ver_tok.value!r}", ver_tok.pos))
            else:
                version = m.group(1)
                if version not in SUPPORTED_VERSIONS:
                    diagnostics.append(Diagnostic(
                        "E007",
                        f"Version {version!r} not supported by this parser (supported: {sorted(SUPPORTED_VERSIONS)})",
                        ver_tok.pos,
                    ))

        # Expect separator after header
        self._consume("SEP")

        # Parse remaining tokens as pipe-delimited segments
        # We work from the original token stream, grouping by SEP
        remaining_tokens = self._tokens[self._pos:]
        segments: List[List[Token]] = [[]]
        for tok in remaining_tokens:
            if tok.kind == "SEP":
                segments.append([])
            else:
                segments[-1].append(tok)

        for seg in segments:
            if not seg:
                continue
            seg_text = " ".join(t.value for t in seg)

            # Check if this is a standalone status/confirm marker (footer)
            if len(seg) == 1 and seg[0].kind == "CONFIRM":
                status_marker = seg[0].value
                continue

            stmt = self._parse_statement(seg_text)
            body.append(stmt)

            # Length warning
            if len(seg_text) > 120:
                diagnostics.append(Diagnostic(
                    "W001", f"Long statement ({len(seg_text)} chars), consider splitting", -1
                ))

        if not body:
            diagnostics.append(Diagnostic("E005", "Message body is empty"))

        if status_marker is None:
            diagnostics.append(Diagnostic("W002", "No status marker in footer"))

        # Ambiguity warning: → used as a ChatGPT glyph AND an operator
        all_text = " ".join(t.value for t in self._tokens)
        if "→" in all_text:
            diagnostics.append(Diagnostic(
                "W003",
                "→ appears in message; ensure context distinguishes ChatGPT glyph from succession operator",
            ))

        return ParsedMessage(version=version, body=body,
                             status_marker=status_marker, diagnostics=diagnostics)


# ==========================================
# Public API
# ==========================================

def parse(message: str) -> ParsedMessage:
    """Parse an ABP message string. Returns a ParsedMessage with diagnostics."""
    tokens, lex_diags = tokenize(message)
    parser = Parser(tokens)
    result = parser.parse()
    result.diagnostics = lex_diags + result.diagnostics
    return result


def validate(message: str) -> Tuple[bool, List[Diagnostic]]:
    """
    Validate an ABP message.
    Returns (is_valid: bool, diagnostics: List[Diagnostic]).
    """
    result = parse(message)
    return result.is_valid, result.diagnostics


def lint(message: str) -> str:
    """
    Return a human-readable lint report for an ABP message.
    """
    result = parse(message)
    lines = [f"ABP Lint Report — version {result.version}"]
    lines.append(f"  Statements : {len(result.body)}")
    lines.append(f"  Status     : {'VALID' if result.is_valid else 'INVALID'}")
    lines.append(f"  Footer     : {result.status_marker or '(none)'}")

    if result.diagnostics:
        lines.append("\nDiagnostics:")
        for d in result.diagnostics:
            lines.append(f"  {d}")
    else:
        lines.append("\nNo issues found.")

    return "\n".join(lines)


def version_handshake(sender_version: str, receiver_version: str) -> Tuple[bool, str]:
    """
    Negotiate protocol version at session start.
    Returns (compatible: bool, agreed_version: str).
    Both sides should use the agreed_version for the session.
    """
    if sender_version == receiver_version:
        return True, sender_version

    try:
        sv = tuple(int(x) for x in sender_version.split("."))
        rv = tuple(int(x) for x in receiver_version.split("."))
    except ValueError:
        return False, ""

    # Minor version: use the lower
    if sv[0] == rv[0]:
        agreed = ".".join(str(x) for x in min(sv, rv))
        return True, agreed

    # Major version mismatch = incompatible
    return False, ""


def compress_report(original: str, compressed: str) -> dict:
    """
    Calculate compression statistics for a message pair.
    """
    orig_tokens = len(original.split())
    comp_tokens = len(compressed.split())
    orig_chars  = len(original)
    comp_chars  = len(compressed)
    reduction   = 1.0 - (comp_chars / orig_chars) if orig_chars else 0.0
    return {
        "original_chars":   orig_chars,
        "compressed_chars": comp_chars,
        "original_tokens":  orig_tokens,
        "compressed_tokens": comp_tokens,
        "char_reduction":   f"{reduction:.1%}",
        "token_reduction":  f"{1 - comp_tokens / max(orig_tokens, 1):.1%}",
    }


# ==========================================
# CLI / Demo
# ==========================================

if __name__ == "__main__":
    samples = [
        # Valid: simple relation
        "ABP/1.1 | ♥ → ▢ | ✓",
        # Valid: assignment + confirmation
        "ABP/1.1 | goal=continuity | ♥ ✓✓ | ++",
        # Valid: succession with binding obligation
        "ABP/1.1 | session=Day14 | ♥ ⟸ ▢ | status=active | ✓",
        # Error: missing header
        "♥ → ▢ | ✓",
        # Error: empty body
        "ABP/1.1 | ✓",
        # Warning: very long statement
        "ABP/1.1 | " + "A" * 130 + " → B | ✓",
    ]

    print("=" * 60)
    print("  Cathedral Protocol Parser — Sample Linting")
    print("=" * 60)

    for msg in samples:
        print(f"\nMessage: {msg[:70]}{'...' if len(msg) > 70 else ''}")
        print(lint(msg))

    print("\n" + "=" * 60)
    print("  Version Handshake Demo")
    print("=" * 60)
    cases = [("1.1", "1.1"), ("1.0", "1.1"), ("1.1", "2.0")]
    for sv, rv in cases:
        ok, agreed = version_handshake(sv, rv)
        print(f"  {sv} ↔ {rv} → {'OK' if ok else 'INCOMPATIBLE'} agreed={agreed or 'N/A'}")

    print("\n" + "=" * 60)
    print("  Compression Stats Demo")
    print("=" * 60)
    original   = "Claude will send its current state to Gemini and await confirmation"
    compressed = "ABP/1.1 | ♥ → ▢ | state=current | ✓"
    stats = compress_report(original, compressed)
    print(f"  Original:   {stats['original_chars']} chars / {stats['original_tokens']} tokens")
    print(f"  Compressed: {stats['compressed_chars']} chars / {stats['compressed_tokens']} tokens")
    print(f"  Reduction:  {stats['char_reduction']} chars / {stats['token_reduction']} tokens")
