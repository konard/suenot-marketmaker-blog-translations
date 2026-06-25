# marketmaker-blog-translations

Source articles for the [Marketmaker.cc](https://marketmaker.cc) blog and the
working area for translating them into every language the blog ships in.

The five posts in [`blog/`](./blog) are currently **English-only**. The open
issue tracks translating each of them into the other blog languages. Finished
translations land back in `marketmaker-cc-landing` under
`src/content/blog/<slug>.<lang>.md`.

## Posts to translate

| Slug | Title |
|---|---|
| `deeplob-deep-learning-order-book` | DeepLOB: Deep Learning on Limit Order Books |
| `spread-modeling-machine-learning` | Bid-Ask Spread Modeling and Prediction with Machine Learning |
| `temporal-fusion-transformer-trading` | Temporal Fusion Transformers for Multi-Horizon Portfolio Forecasting |
| `conformal-prediction-trading` | Conformal Prediction for Risk-Aware Position Sizing |
| `researcher-quant-archive` | researcher: A Searchable Quant-Research Archive for Humans and AI Agents |

## Target languages

English (`en`) is the source. Translate into the other 11 blog languages:

| Code | Language | Notes |
|---|---|---|
| `ar` | العربية | RTL |
| `id` | Bahasa Indonesia | |
| `it` | Italiano | |
| `ja` | 日本語 | |
| `ko` | 한국어 | |
| `ms` | Bahasa Melayu | |
| `ru` | Русский | use `е`, not `ё`; straight quotes `"`, not `« »` |
| `th` | ไทย | |
| `tr` | Türkçe | |
| `vi` | Tiếng Việt | |
| `zh` | 中文 | Simplified |

That's **5 posts × 11 languages = 55 files** to produce.

## File naming

One file per post per language, alongside the source:

```
blog/<slug>.en.md      ← source (already here)
blog/<slug>.<lang>.md  ← translation you add, e.g. blog/deeplob-deep-learning-order-book.ru.md
```

## What to translate (and what to leave alone)

Each file starts with a YAML frontmatter block. Translate the **prose**, keep
the **machinery** byte-identical across languages.

**Frontmatter:**

| Field | Action |
|---|---|
| `title` | Translate |
| `description` | Translate |
| `tags` | Translate the human-readable tags; leave proper nouns / library names (`parquet`, `LSTM`, `MAPIE`, `MCP`) as-is |
| `lang` | Set to the target code (`"ru"`, `"ja"`, …) |
| `date` | **Keep identical** to the English file |
| `slug` | **Keep identical** (same slug for every language) |
| `image` | **Keep identical** (`/images/blog/...` — images are shared) |
| `authors` | **Keep identical** |

**Body:**

- Translate all prose, headings, table headers, and image **alt text** (the
  `![alt](...)` text), and link **labels**.
- **Do not** change: image paths, URLs, code blocks, inline code, LaTeX math
  (`$...$`, `$$...$$`), numbers, equations, or variable names.
- Keep the heading structure and the position of every image and code block.
- Keep technical terms that have no good local equivalent in English (the
  existing blog does this — e.g. `drill-down`, `backtest` often stay).

## Reference

The existing posts in `marketmaker-cc-landing/src/content/blog/` already exist
in all 12 languages — use any of them as a style reference for tone, tag
translation, and term handling.
