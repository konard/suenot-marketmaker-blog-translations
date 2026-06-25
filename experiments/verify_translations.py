#!/usr/bin/env python3
"""Verify translated blog markdown files keep source invariants.

This is intentionally dependency-free so it can run in CI or locally without
installing language-detection packages. The language gate is heuristic: it
rejects files that are structurally correct but still mostly English.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "blog"
SLUG = "researcher-quant-archive"
TARGETS = ["ar", "id", "it", "ja", "ko", "ms", "ru", "th", "tr", "vi", "zh"]
KEEP_IDENTICAL = {"date", "slug", "image", "authors"}

LATIN_LANGS = {"id", "it", "ms", "tr", "vi"}
SCRIPT_RANGES = {
    "ar": ("\u0600", "\u06ff"),
    "ja": ("\u3040", "\u30ff"),
    "ko": ("\uac00", "\ud7af"),
    "ru": ("\u0400", "\u04ff"),
    "th": ("\u0e00", "\u0e7f"),
    "zh": ("\u4e00", "\u9fff"),
}
COMMON_ENGLISH = {
    "the",
    "and",
    "that",
    "with",
    "search",
    "papers",
    "research",
    "agent",
    "agents",
    "index",
    "corpus",
    "data",
    "tools",
    "source",
}


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    _, frontmatter, body = text.split("---\n", 2)
    return frontmatter, body


def parse_frontmatter(text: str) -> dict[str, str | list[str]]:
    data: dict[str, str | list[str]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        key, value = raw.split(": ", 1)
        if value.startswith("["):
            data[key] = ast.literal_eval(value)
        else:
            data[key] = value.strip('"')
    return data


def protected_tokens(body: str) -> set[str]:
    tokens = set(re.findall(r"`[^`]+`", body))
    tokens.update(re.findall(r"\$\$.*?\$\$|\$[^$\n]+\$", body, flags=re.S))
    tokens.update(re.findall(r"https?://[^\s)]+|/[A-Za-z0-9_./-]+", body))
    tokens.update(re.findall(r"\b\d[\d,.:/~–-]*\b", body))
    return tokens


def structure(body: str) -> list[str]:
    result: list[str] = []
    in_code = False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            result.append("```")
        elif in_code:
            result.append("CODE")
        elif line.startswith("#"):
            result.append(re.match(r"^#+", line).group(0))
        elif line.startswith("!["):
            result.append("IMAGE:" + line[line.rfind("](") + 2 :])
        elif line.startswith("|"):
            result.append("TABLE")
    return result


def strip_protected(body: str) -> str:
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"`[^`]+`", " ", body)
    body = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$", " ", body, flags=re.S)
    body = re.sub(r"https?://[^\s)]+|/[A-Za-z0-9_./-]+", " ", body)
    return body


def language_gate(lang: str, body: str) -> list[str]:
    errors: list[str] = []
    prose = strip_protected(body).lower()
    letters = re.findall(r"[a-zA-Z]+", prose)
    english_hits = sum(1 for word in letters if word in COMMON_ENGLISH)
    english_ratio = english_hits / max(1, len(letters))

    if lang in SCRIPT_RANGES:
        lo, hi = SCRIPT_RANGES[lang]
        native = sum(1 for ch in prose if lo <= ch <= hi)
        if native < 300:
            errors.append(f"{lang}: too little target-script text ({native} chars)")
        if english_ratio > 0.08:
            errors.append(f"{lang}: too many common English words ({english_ratio:.1%})")
    elif lang in LATIN_LANGS:
        if english_ratio > 0.16:
            errors.append(f"{lang}: likely still English ({english_ratio:.1%} common words)")
    return errors


def main() -> int:
    source_front, source_body = split_frontmatter((BLOG / f"{SLUG}.en.md").read_text())
    source_meta = parse_frontmatter(source_front)
    source_structure = structure(source_body)
    source_tokens = protected_tokens(source_body)
    errors: list[str] = []

    for lang in TARGETS:
        path = BLOG / f"{SLUG}.{lang}.md"
        if not path.exists():
            errors.append(f"{lang}: missing {path}")
            continue
        front, body = split_frontmatter(path.read_text())
        meta = parse_frontmatter(front)
        for key in KEEP_IDENTICAL:
            if meta.get(key) != source_meta.get(key):
                errors.append(f"{lang}: frontmatter {key} differs")
        if meta.get("lang") != lang:
            errors.append(f"{lang}: lang is {meta.get('lang')!r}")
        if not meta.get("title") or meta.get("title") == source_meta["title"]:
            errors.append(f"{lang}: title was not translated")
        if not meta.get("description") or meta.get("description") == source_meta["description"]:
            errors.append(f"{lang}: description was not translated")
        if structure(body) != source_structure:
            errors.append(f"{lang}: markdown structure differs from source")
        missing = sorted(token for token in source_tokens if token not in body)
        if missing:
            errors.append(f"{lang}: missing protected tokens: {', '.join(missing[:10])}")
        errors.extend(language_gate(lang, body))
        if lang == "ru" and ("ё" in body or "«" in body or "»" in body):
            errors.append('ru: must use "е" and straight quotes')

    if errors:
        print("Translation verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("All translations passed structural and language checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
