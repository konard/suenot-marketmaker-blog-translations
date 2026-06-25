#!/usr/bin/env python3
"""Validate translated Marketmaker blog Markdown files."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


TARGET_LANGS = ["ar", "id", "it", "ja", "ko", "ms", "ru", "th", "tr", "vi", "zh"]
IDENTICAL_FRONTMATTER = {"date", "slug", "image", "authors"}
TRANSLATED_FRONTMATTER = {"title", "description", "tags"}


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing opening frontmatter fence")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("missing closing frontmatter fence")
    raw_frontmatter = text[4:end]
    body = text[end + 5 :]
    data: dict[str, object] = {}
    for line in raw_frontmatter.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key = key.strip()
        value = value.strip()
        try:
            data[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            data[key] = value.strip('"')
    return data, body


def code_blocks(body: str) -> list[str]:
    return re.findall(r"```.*?```", body, flags=re.DOTALL)


def images(body: str) -> list[tuple[str, str]]:
    return re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", body)


def headings(body: str) -> list[str]:
    return [line.split(" ", 1)[0] for line in body.splitlines() if re.match(r"^#{1,6} ", line)]


def math_spans(body: str) -> list[str]:
    block = re.findall(r"\$\$.*?\$\$", body, flags=re.DOTALL)
    inline = re.findall(r"(?<!\$)\$[^$\n]+\$(?!\$)", body)
    return block + inline


def english_word_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    total_tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if not total_tokens:
        return 0.0
    return len(words) / len(total_tokens)


def assert_language_gate(path: Path, lang: str, body: str) -> list[str]:
    errors: list[str] = []
    ratio = english_word_ratio(body)
    thresholds = {
        "id": 0.32,
        "it": 0.35,
        "ms": 0.34,
        "tr": 0.32,
        "vi": 0.34,
    }
    if lang in thresholds and ratio > thresholds[lang]:
        errors.append(f"{path}: English word ratio {ratio:.2%} exceeds {thresholds[lang]:.0%}")
    if lang not in thresholds and ratio > 0.22:
        errors.append(f"{path}: body appears predominantly English ({ratio:.2%} ASCII word ratio)")

    script_checks = {
        "ar": r"[\u0600-\u06ff]",
        "ja": r"[\u3040-\u30ff\u4e00-\u9fff]",
        "ko": r"[\uac00-\ud7af]",
        "ru": r"[\u0400-\u04ff]",
        "th": r"[\u0e00-\u0e7f]",
        "zh": r"[\u4e00-\u9fff]",
    }
    if lang in script_checks and not re.search(script_checks[lang], body):
        errors.append(f"{path}: expected target-language script for {lang}")
    if lang == "ru":
        if "ё" in body or "Ё" in body:
            errors.append(f"{path}: Russian translation must use е, not ё")
        if "«" in body or "»" in body:
            errors.append(f'{path}: Russian translation must use straight quotes, not guillemets')
    return errors


def validate_translation(source_path: Path, lang: str) -> list[str]:
    target_path = source_path.with_name(source_path.name.replace(".en.md", f".{lang}.md"))
    errors: list[str] = []
    if not target_path.exists():
        return [f"{target_path}: missing translation"]

    try:
        source_meta, source_body = parse_frontmatter(source_path.read_text(encoding="utf-8"))
        target_meta, target_body = parse_frontmatter(target_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{target_path}: {exc}"]

    for key in IDENTICAL_FRONTMATTER:
        if target_meta.get(key) != source_meta.get(key):
            errors.append(f"{target_path}: frontmatter {key!r} must match English source")
    if target_meta.get("lang") != lang:
        errors.append(f"{target_path}: frontmatter lang must be {lang!r}")
    for key in TRANSLATED_FRONTMATTER:
        if target_meta.get(key) == source_meta.get(key):
            errors.append(f"{target_path}: frontmatter {key!r} was not translated")

    if headings(source_body) != headings(target_body):
        errors.append(f"{target_path}: heading levels/order changed")
    if code_blocks(source_body) != code_blocks(target_body):
        errors.append(f"{target_path}: fenced code blocks changed")
    if [path for _, path in images(source_body)] != [path for _, path in images(target_body)]:
        errors.append(f"{target_path}: image paths changed")
    if math_spans(source_body) != math_spans(target_body):
        errors.append(f"{target_path}: LaTeX/math spans changed")

    source_stripped = re.sub(r"```.*?```", "", source_body, flags=re.DOTALL)
    target_stripped = re.sub(r"```.*?```", "", target_body, flags=re.DOTALL)
    if source_stripped == target_stripped:
        errors.append(f"{target_path}: body is identical to English source outside code blocks")
    errors.extend(assert_language_gate(target_path, lang, target_stripped))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="blog/temporal-fusion-transformer-trading.en.md",
        help="English source Markdown file",
    )
    parser.add_argument("--langs", nargs="+", default=TARGET_LANGS)
    args = parser.parse_args()

    source_path = Path(args.source)
    all_errors: list[str] = []
    for lang in args.langs:
        all_errors.extend(validate_translation(source_path, lang))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Validated {len(args.langs)} translations for {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
