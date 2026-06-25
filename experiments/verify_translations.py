#!/usr/bin/env python3
"""Verify translated blog files preserve source invariants."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


TARGETS = ["ar", "id", "it", "ja", "ko", "ms", "ru", "th", "tr", "vi", "zh"]
SOURCE = Path("blog/researcher-quant-archive.en.md")


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter start")
    _, raw_fm, body = text.split("---\n", 2)
    data: dict[str, str] = {}
    for line in raw_fm.splitlines():
        key, value = line.split(": ", 1)
        data[key] = value
    return data, body


def extract_fenced_code(body: str) -> list[str]:
    return re.findall(r"```.*?```", body, flags=re.S)


def extract_urls(body: str) -> list[str]:
    return re.findall(r"https?://[^\s)\"]+", body)


def extract_image_paths(body: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)


def tags(value: str) -> list[str]:
    return ast.literal_eval(value)


def main() -> int:
    errors: list[str] = []
    src_fm, src_body = split_frontmatter(SOURCE.read_text())
    src_code = extract_fenced_code(src_body)
    src_urls = extract_urls(src_body)
    src_images = extract_image_paths(src_body)

    for lang in TARGETS:
        path = Path(f"blog/researcher-quant-archive.{lang}.md")
        if not path.exists():
            errors.append(f"{path}: missing")
            continue

        fm, body = split_frontmatter(path.read_text())
        for key in ["date", "slug", "image", "authors"]:
            if fm.get(key) != src_fm[key]:
                errors.append(f"{path}: frontmatter {key} differs")
        if fm.get("lang") != f'"{lang}"':
            errors.append(f"{path}: lang is {fm.get('lang')}, expected \"{lang}\"")
        if fm.get("title") == src_fm["title"]:
            errors.append(f"{path}: title is still English")
        if fm.get("description") == src_fm["description"]:
            errors.append(f"{path}: description is still English")
        if tags(fm.get("tags", "[]")) == tags(src_fm["tags"]):
            errors.append(f"{path}: tags are still identical to English")
        if extract_fenced_code(body) != src_code:
            errors.append(f"{path}: fenced code blocks differ")
        if extract_urls(body) != src_urls:
            errors.append(f"{path}: URLs differ")
        if extract_image_paths(body) != src_images:
            errors.append(f"{path}: image paths differ")
        if "Russian translation:" in body or "日本語訳:" in body or "Русский перевод:" in body:
            errors.append(f"{path}: contains translation-label placeholder text")
        if lang == "ru" and ("ё" in body or "«" in body or "»" in body):
            errors.append(f"{path}: ru typography rule violated")

    if errors:
        print("\n".join(errors))
        return 1
    print("translation invariants passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
