# filings-qa

Ask questions about US companies' annual reports (10-K filings) and get answers where
**every claim cites its source**, plus an evaluation harness that measures how far those
answers can be trusted.

In finance, a fluent answer with a wrong number is worse than no answer. So this project
is built around one question: *how do you know the answer is right?*

## How it works

```
question
   │
   ▼
 plan ────► gather ────► answer ────► verify ────► answer + citations + issues
 (LLM)      (tools)      (LLM)        (code)
                           ▲             │
                           └── retry ────┘  once, with the problems found
```

| Step | What happens |
|---|---|
| **plan** | The LLM identifies the company, fiscal year, which reported figures are needed, and what to search for in the text. Unknown companies are dropped, not guessed. |
| **gather** | Two tools. **Reported facts**: numbers from the SEC's XBRL data (what the company itself tagged). **Filing search**: passages from the 10-K (Business, Risk Factors, MD&A, Market Risk, Financial Statements) via PostgreSQL + pgvector. |
| **answer** | The LLM answers from that evidence only, as short claims that each cite evidence ids, or refuses if the evidence is not enough. |
| **verify** | Deterministic checks, no LLM involved: every claim is cited; every citation was actually provided; **every number in a claim appears in the sources that claim cites** (handling "391.0 billion" vs "391,035" in a table in millions). On failure the model gets one retry with the list of problems; if it still fails, the answer is returned *with its issues attached* rather than quietly accepted. |

Built with Python, LangGraph, PostgreSQL + pgvector, FastAPI, and Claude (Anthropic API).
Embeddings run locally (`BAAI/bge-small-en-v1.5`), so ingestion has no per-call cost.

## Evaluation

The evaluation is the point of the project, so it is designed to be hard to fool:

* **Numeric questions** (generated): ground truth comes from XBRL, i.e. the value the
  company reported, not from another model. Scored on *correct* (the right number within
  0.5%) and *grounded* (the claim cites the fact it came from).
* **Text questions** (hand-labelled): I read the relevant 10-K section, record which item
  holds the answer and the key points a correct answer must contain, then grade each answer
  by hand as `correct`, `partial`, `wrong` or `invented`.
* **Unanswerable questions**: future values, companies not ingested, facts a 10-K does
  not contain. The right behaviour is to refuse.

Every rate is reported with a **Wilson 95% confidence interval**, because with a few dozen
questions "92%" might really be anywhere from about 80% to 97%.

Metrics reported: numeric accuracy, grounded numeric answers, human-graded text accuracy,
answers citing the expected 10-K item, correct refusals, false refusals, answers passing
all verification checks, and answers containing an unsupported number.

### Results

_Not published yet: the first full run is in progress. This section will show the metrics
table produced by `filings-qa eval`, with the question set and model version used._

## Run it

```bash
docker compose up -d                       # PostgreSQL + pgvector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local-embeddings,dev]"
cp .env.example .env                       # set SEC_USER_AGENT and ANTHROPIC_API_KEY

filings-qa ingest AAPL MSFT NVDA --years 2  # download, chunk, embed, store
filings-qa ask "What was Apple's total revenue for fiscal year 2024?"
filings-qa build-gold AAPL MSFT NVDA        # numeric questions with XBRL ground truth
filings-qa eval                             # writes eval/results/report_*.md
uvicorn filings_qa.api:app --reload         # POST /ask {"question": "..."}
```

SEC asks automated clients to send a User-Agent with a contact email and stay under
10 requests per second; the client does both and caches every download in `data/`.

## Tests

```bash
pytest                                     # offline: scripted LLM, in-memory store
TEST_DATABASE_URL=postgresql://filings:filings@localhost:5432/filings pytest  # + pgvector
```

The offline suite covers parsing (including skipping the table of contents), XBRL
selection (ignoring prior-year comparatives and quarterly values inside a 10-K), the
number-grounding checks, the agent graph including the retry path and refusals, scoring,
and the confidence intervals. CI runs everything, including a real pgvector database.

## Known limitations

* Fiscal year is taken as the calendar year in which the reporting period ends.
* Item detection uses heading patterns; unusual filing layouts can merge or miss sections.
* The number check confirms a number *appears* in the cited source, not that it is used in
  the right sense (e.g. a revenue figure quoted as profit). The human-graded text set and
  the "grounded" metric exist to catch that.
* Only a handful of companies and the most recent filings are indexed.

## License

MIT
