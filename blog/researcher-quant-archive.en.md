---
title: "researcher: A Searchable Quant-Research Archive for Humans and AI Agents"
description: "How we built researcher.marketmaker.cc — a unified, full-text-searchable archive of quant research (arXiv papers, GitHub repos, quant blogs, TradingView Pine scripts) that humans browse and AI agents query over MCP."
tags: ["researcher", "quant", "arxiv", "meilisearch", "mcp", "ai-agents", "research", "algotrading"]
date: "2026-06-25"
lang: "en"
image: "/images/blog/researcher-quant-archive.webp"
slug: "researcher-quant-archive"
authors: ["suenot"]
---

# researcher: A Searchable Quant-Research Archive for Humans and AI Agents

Quant research is everywhere and nowhere. The paper you need is on arXiv. The reference implementation is on GitHub. The intuition is buried in a blog post somebody wrote in 2019. The actual entry/exit logic is a Pine script on TradingView with 400 likes and no documentation. Four corpora, four search boxes, four sets of conventions, zero cross-references. When you are trying to decide whether an idea is worth a week of backtesting, that fragmentation is the real cost — not the reading, but the *finding*.

So we built our own. **[researcher.marketmaker.cc](https://researcher.marketmaker.cc)** is a curated archive and search engine for quantitative-trading research. It pulls together the material that's normally scattered across arXiv, GitHub, quant blogs, and TradingView into one place, indexes all of it for full-text search, and — this is the part we care about most — exposes the whole corpus to AI agents through a Model Context Protocol (MCP) endpoint and a public REST API. It's a research substrate that a human can browse with a keyboard and an agent can query with a tool call, backed by the exact same indexes.

This post is a tour of what's inside, how it's built, and why it sits where it does in our AI-agent stack.

## What's in the Corpus

![Four data silos — papers, code repositories, articles, and chart scripts — converging into a single index](/images/blog/researcher-quant-archive-corpus.webp)

researcher unifies four primary datasets, each with its own full-text index. The counts below are as of 2026-06-12, and they move — the arXiv pipeline runs daily and the index is rebuilt from source, so the numbers grow.

| Dataset | Source | Documents | What you search |
|---|---|---|---|
| **Papers** | arXiv q-fin (1997–2026) | ~18,647 | title, abstract, authors (filter by category) |
| **Code** | GitHub repos | ~12,957 | name, description, topics (filter by language, stars) |
| **Articles** | Quant blogs | ~4,633 | title, description (filter by source, date) |
| **Strategies** | TradingView Pine scripts | ~15,180 | title, description, tags (filter by category) |

That's a little over 51,000 documents across the four searchable indexes. The **papers** index is the largest single corpus and the one we put the most work into: it's the full arXiv quantitative-finance firehose (`q-fin.*`) going back to 1997, not a hand-picked subset. Earlier the site shipped only a few hundred curated papers; the current index is the complete q-fin corpus, with curated provenance merged in on top so a paper that was *also* referenced by a specific quant blog carries that attribution.

Beyond the four search indexes, the human-facing site layers on more: research notes and daily digests we write ourselves, a directory of quant **sites** and **authors**, a **videos** section indexing relevant YouTube channels, and a **funds** directory. The four indexes are the searchable spine; everything else is curation around it.

## Single Source of Truth

![One canonical dataset core feeding a search index and an application view in parallel — a single source of truth](/images/blog/researcher-quant-archive-single-source.webp)

The thing that quietly broke for us early on — and the thing we now design hard against — is the *count mismatch*. The homepage said one number, the search returned another, the API a third. At one point the front page advertised 719 papers while search was returning over 18,000. Nothing is more corrosive to trust in a research tool than a corpus that can't agree with itself about how big it is.

The fix was to make every surface read from one place. For the papers corpus, **Meilisearch is the source of truth**. There is no second copy of the papers living in the app bundle; the count on the homepage, the count on `/papers`, the count returned by the API, and the documents you actually search are all the same index. The corpus itself is built offline by an ingestion step that takes the curated paper set, unions it with the arXiv firehose (deduplicated by arXiv `id`, with provenance arrays merged), sorts newest-first, and writes a single ~25 MB file that the indexer consumes. That file is roughly 15,000–18,000 records and is deliberately *not* bundled into the client — a corpus that size has no business shipping to a browser.

The other three datasets (code, articles, Pine scripts) are read server-side from their JSON files and indexed from the same files, with migration to Meili-as-source the planned next step. The rule across the board: read the data on the server, never `import` a multi-megabyte dataset into the client bundle, and let the index and the displayed counts come from the same origin. Desync becomes structurally impossible.

## Search: Full-Text, Typo-Tolerant, Faceted

![Full-text, typo-tolerant, faceted search: a query beam entering an index and returning a ranked, highlighted result list](/images/blog/researcher-quant-archive-search.webp)

The search engine is [Meilisearch](https://www.meilisearch.com/), running on the same server as the app and bound to localhost — it is not publicly exposed. We use it as a **full-text search engine, not a vector store**. No embeddings, no semantic-similarity magic. For "find me the papers and repos that mention *this concept*," typo-tolerant lexical search over titles, abstracts, descriptions, authors, and tags is fast, predictable, and debuggable in a way that an embedding index is not. Each index stores the *complete* original document plus an added `_id`, so a search returns fully hydrated records the app can render directly — no second fetch to rehydrate hits.

A few details that matter in practice:

- **Typo tolerance and relevance ranking** come for free from Meilisearch. Searching `momentm` still finds momentum papers; results are ranked, not just filtered.
- **Faceted filtering.** Papers filter by arXiv `category` (`q-fin.PM`, `q-fin.TR`, …), repos by `language` and `stars`, articles by `source` and `date`, Pine scripts by `category`. The `/papers` page builds its category dropdown from the live facet distribution of the index, so the filter options always reflect what's actually in the corpus.
- **camelCase splitting.** Meilisearch tokenizes on whitespace and punctuation but not on camelCase. That means a repo literally named `TradingAgents` would be a single token, unreachable by the natural query "trading agents." During indexing we derive a `name_split` field — `TradingAgents` → `TradingAgents Trading Agents`, `ai-hedge-fund` → `ai-hedge-fund ai hedge fund` — and add it to the searchable attributes. The original token is kept first so exact-name matches still rank highest, and the derived field is never returned to clients. It's a small thing that makes the difference between finding the flagship repo and not.
- **Browse = newest-first.** An empty query isn't an error; it's the browse path. On `/papers` an empty query sorts by `published` descending, so the page doubles as a reverse-chronological feed of the latest q-fin research.

- **A side index for aggregates.** Some numbers Meilisearch can't compute cheaply at query time — total stars across all repos, total Python files, total notebooks. Rather than scan the whole corpus on every page load, the indexer writes those sums once, at index time, into a small `researcher_meta` index holding one document per dataset. The `stats` endpoint reads them straight back. Counts that change only on reindex are computed only on reindex.
- **A pagination ceiling that's actually usable.** The papers index raises Meilisearch's `maxTotalHits` to 50,000 and marks `published` sortable, so you can page deep into an ~18k-document corpus and sort the whole thing newest-first — not just the first page of relevance hits.

The indexer is idempotent: it creates each index if absent, re-applies settings, and upserts every document in batches of 2,000 keyed by `_id` (papers derive theirs from the arXiv id, repos and articles from a hash of the URL). Because the whole index is reconstructible from the source file, there's no backup to manage — a rollback is just a reindex. Re-running it is safe by construction.

## Agent-Accessible: MCP and a Public API

![Autonomous AI-agent nodes connecting through an MCP and public API surface into a central research corpus](/images/blog/researcher-quant-archive-mcp-api.webp)

Here's the part that ties researcher to the rest of what we do. The corpus isn't just a website with a search box — it's a **tool an AI agent can call**.

### The MCP endpoint

researcher exposes a Model Context Protocol server at **`/api/mcp`** over Streamable HTTP. Any MCP-compatible agent — Claude, a custom agent in our own stack, anything that speaks the protocol — can connect and call read-only tools against the live corpus. There are 13 tools, grouped by dataset, following a consistent `search` / `get` / `list` shape:

| Group | Tools |
|---|---|
| Papers | `search_papers`, `get_paper`, `list_papers` |
| Code | `search_repos`, `get_repo`, `list_repos` |
| Articles | `search_articles`, `get_article`, `list_articles_by_site` |
| Strategies | `search_pine`, `get_pine_script`, `list_pine` |
| Knowledge | `knowledge_query` (graceful stub, reserved for a future graph layer) |

The tool schemas are written for an agent's benefit, not a human's. `search_papers`, for example, advertises itself as typo-tolerant relevance-ranked search over title, abstract, and authors, with an optional `category` filter (e.g. `q-fin.PM`) and a result limit — and tells the agent to call `get_paper` for the full abstract once it's narrowed things down. `search` returns compact, snippet-sized hits so an agent can scan many results cheaply; `get` returns the full record once it has picked one. That two-step shape keeps an agent's context window from drowning in abstracts it doesn't need.

Concretely, an agent investigating, say, optimal execution can run `search_papers("optimal execution", category: "q-fin.TR")` to get a ranked shortlist of titles and snippets, `search_repos("optimal execution", language: "Python")` to find implementations sorted by relevance and filterable by stars, and `search_pine("VWAP")` to see how the same idea shows up as a published TradingView strategy — three tool calls against three corpora that were, an hour ago, three different websites. Then a single `get_paper` pulls the full abstract for the one that looked promising. The agent never leaves the protocol, and every result is a real, hydrated record rather than a search-result stub it has to go re-fetch.

### The public REST API

For non-MCP consumers there's a parallel REST surface under `/api/v1/`: `papers`, `repos`, `articles`, `pine`, and a `stats` aggregate. It speaks plain JSON with `q`, `category`, `limit`, and `offset` parameters, returns the true total and the facet distribution alongside each page, and is CORS-enabled. `GET /api/v1/papers?q=optimal+execution&category=q-fin.TR` is a one-liner from anywhere. The same endpoint drives the site's own `/papers` page — the browser is just another API client.

### Failing honestly

A search backend that lies is worse than one that's down. We took a deliberate stance on what happens when Meilisearch is unreachable. The data layer *throws* on failure rather than silently returning empty results — and callers decide how to handle it. For datasets that still keep an in-memory copy, the tools fall back to a plain `.filter()` over that copy, so the site stays up. For papers, where Meilisearch *is* the source of truth and there is no second copy, the tools and the API return an explicit error (the API responds `503`) rather than serving stale or partial data. Each search call has a short timeout so a hung index can't stall a tool. The principle: degrade loudly, never quietly hand back wrong answers.

## How the Data Gets In

![Ingestion pipeline: external sources flowing through extraction and normalization stages into a unified index](/images/blog/researcher-quant-archive-pipeline.webp)

The corpus is fed by a pipeline of scrapers, all running against free, public sources.

- **Papers** come from the arXiv Atom API. A harvester pulls the full `q-fin` corpus into JSONL, a build step unions it with the curated set (dedup by arXiv id, provenance merged), and the result is handed to the indexer. The harvester also has an "enrich these specific ids" mode for seeding from external reading lists.
- **Code** is a crawl of quant-relevant GitHub repositories, captured with the metadata that matters for filtering — stars, forks, primary language, topics, and counts of Python files and notebooks.
- **Articles** are scraped from quant blogs and aggregators, with the better ones mirrored locally so they survive link rot. The homepage flags which articles we've saved a local copy of.
- **Strategies** are TradingView Pine scripts with their metadata — author, category, tags, likes, and whether the listing includes code, a chart, or analysis.
- **Videos** index relevant YouTube channels so the talks and walkthroughs are discoverable alongside the written material.

Reindexing in production runs over an SSH tunnel to the localhost-bound Meilisearch, because the engine is never exposed to the internet. The whole loop — harvest, build, deploy, index — is designed to be re-run idempotently, which is exactly what the daily cron does.

## Access and Hosting

![Secure, authenticated access to a hosted service: an auth gate in front of a luminous server stack](/images/blog/researcher-quant-archive-access.webp)

researcher runs on our Server 1 as a small Docker Compose stack: a Next.js container behind Traefik and a Meilisearch container bound to localhost. The Next.js app reads its datasets server-side and talks to Meilisearch over the internal network.

Access is gated through **[auth.marketmaker.cc](https://auth.marketmaker.cc)**, our shared identity service. Tokens are RS256 JWTs verified against the auth service's JWKS — every authorization decision checks the signature (with strict issuer and algorithm checks, failing closed if the key endpoint is unreachable), and the unverified decode path is used only for cosmetic UI like showing your email in the navbar. The auth service issues per-service roles; on researcher the `admin` role gates the internal admin area (where we run and monitor scrapers), and the public homepage needs no token at all. It's the same auth fabric that fronts our other internal tools, so one login carries across the ecosystem.

## Where It Fits in the Marketmaker Stack

![Where the research archive fits in the Marketmaker stack: a central substrate feeding AI agents and human researchers](/images/blog/researcher-quant-archive-ecosystem.webp)

researcher is infrastructure, not a destination. The point isn't the website — it's that we now have a *queryable* view of the field that both people and agents share.

For us as humans, it's where a lot of this very blog comes from. When we review a tool like [VectorBT](/blog/vectorbt-overview) or dissect a framework like [TradingAgents](/blog/tradingagents-multi-agent-framework) or [Fincept Terminal](/blog/fincept-terminal-review), the starting point is often a search across researcher: what papers does this build on, what other repos solve the same problem, who's written about it. The archive is the funnel; the blog posts are what falls out of it.

For our **AI agents**, it's something more structural. A research substrate that's reachable over MCP means an agent doing strategy work doesn't have to scrape arXiv live, juggle four different APIs, or guess at what's out there — it calls `search_papers`, `search_repos`, `search_pine` against a corpus that's already unified, deduplicated, and indexed. That's the same direction as our command-and-operate (cmdop) and agent tooling: give agents typed, read-only, well-documented tools over real data, fail loudly when the backend is unavailable, and let one shared backend serve the human UI and the machine interface from identical indexes. The human browses and the agent queries — but they're looking at the same archive, and that's the whole idea.

## Conclusion

researcher started as a fix for a small, annoying problem — that quant research is scattered across four places that don't talk to each other — and turned into something we lean on daily. Roughly 51,000 documents across papers, code, articles, and strategies, all behind one full-text search engine, all reachable both by a human with a browser and by an agent with an MCP client. It's intentionally unglamorous: full-text search, not embeddings; a single source of truth, not a clever cache; tools that throw honest errors, not ones that paper over outages.

If you're building agents for trading research, the lesson generalizes past our particular corpus: the highest-leverage thing you can give an agent is not a bigger model but a clean, unified, queryable view of the data it needs — exposed through the same indexes the humans trust. That's what researcher is.
