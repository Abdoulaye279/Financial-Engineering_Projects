# LLM analysis of Lehman Brothers' last 10-Q: full context vs RAG

Gemini extracts bankruptcy warning signs from Lehman Brothers' 10-Q of 10 July 2008, two months before the bankruptcy, with an exact supporting quote for each. Two ways of feeding the filing are compared on cost, coverage and quote reliability.

## Key results

| | Full filing | RAG |
|---|---|---|
| Tokens sent per question | 109,582 | ~15,000 (estimated) |
| Warning signs found | 10 | 7 |
| Quotes found word for word in the filing | 7 / 10 | 2 / 7 |
| Invented numbers (checked by hand in the PDF) | 0 | 0 |

Asking for exact quotes is not enough: the model turns table rows into "quotes", cuts sentences with "..." or rewrites dates. The numbers are right, the quotes are not, and only an automatic check catches it.

## Data

| Item | Value |
|---|---|
| Filing | Lehman Brothers Holdings, form 10-Q, filed 2008-07-10 (quarter ended 31 May 2008) |
| Source | SEC EDGAR through `edgartools` (CIK 806085) |
| Size | 535,654 characters |

## Methodology

| Step | Full context (`part1_full_context`) | RAG (`part2_rag`) |
|---|---|---|
| Input | Whole filing in one prompt | 77 chunks of 8,192 characters (1,024 overlap), embedded with `gemini-embedding-001`, stored in Qdrant |
| Retrieval | none | 5 queries (leverage, mortgage exposure, liquidity, losses, Repo 105), 5 chunks each, 10 distinct chunks kept, rank by rank |
| Model | `gemini-3-flash-preview`, temperature 0 | same |
| Output | Structured (pydantic): type, metric, value, severity 1-10, exact quote | same |
| Check | Each quote searched in the source text, ignoring spaces and case | Each quote searched in the retrieved context |

Then every quote that failed was looked up by hand in the PDF of the 10-Q.

## Results

**Full filing**: $2.8bn quarterly loss, $4.0bn of negative valuation adjustments, $3.1bn of potential collateral calls, negative Capital Markets revenues, $2.76bn subprime and $29.4bn commercial mortgage exposure, $37.9bn of Level 3 assets, 19% cut in total assets, $4bn share issue in June 2008, own-credit gain from widening spreads.

**RAG**: $2.77bn loss, $17.9bn of cash used in operations, -$3.0bn Fixed Income revenues, $2.9bn collateral if downgraded, leverage ratio 24.3x, $23.9bn high-yield positions, $60.8bn mortgage and asset-backed securities. Missed the June share issue and the Level 3 assets, probably absent from the 10 chunks.

**Failed quotes**

| | Full filing (3 failures) | RAG (5 failures) |
|---|---|---|
| Table row rewritten as a sentence | 2 | 4 |
| Real sentence cut with "..." | (same 2) | 1 |
| Wording changed ("At May 31, 2008" for "At the end of the second quarter of 2008") | 1 | 0 |

All the figures in the failed quotes are in the filing: $2,755m subprime, $29.4bn commercial mortgages, $37.9bn Level 3, $(2,774)m net income, 24.34x leverage, $23,856m high yield, $60.8bn mortgage and asset-backed securities, $2.9bn collateral.

## Limitations

- One filing and one model; results vary with the model version.
- RAG token count estimated from the indexing log (~1,500 tokens per chunk), not measured.
- 10 chunks retrieved: coverage depends on the five hand-written queries.
- The quote check is exact matching; a quote cut with "..." fails even when both pieces are real.

## Reproducing

Run in Colab from the badge of each notebook. Needs a Gemini API key (Colab secret `GEMINI_API_KEY`, or `GOOGLE_API_KEY` locally) and an email in `set_identity` for SEC access. The free tier works; RAG indexing takes ~4 min because of the 30k tokens per minute limit.

| Notebook | Content | Runtime |
|---|---|---|
| `part1_full_context` | Whole filing in one prompt, quote check | ~1 min |
| `part2_rag` | Chunking, Qdrant, retrieval, quote check | ~5 min |
