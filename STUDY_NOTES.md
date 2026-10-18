# RBAC RAG Chatbot — Study Notes

Learning log for building a Role-Based-Access-Control RAG chatbot for "FinSolve Technologies"
using LangChain + Chroma + Groq. Updated after every implementation stage.

**Stack decided:**
- Framework: LangChain (LCEL)
- Vector DB: Chroma (embedded, local, metadata filtering)
- Embeddings: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, no API key)
- LLM: Groq Cloud (`ChatGroq`)
- Interface: CLI
- Guardrails: hand-written regex/rules engine

---

## Progress Checklist

- [x] Stage 0 — Environment & dependencies
- [x] Stage 1 — RAG primer (concepts)
- [x] Stage 2 — Ingestion pipeline (`ingest.py`)
- [x] Stage 3 — RBAC layer (`rbac.py`)
- [x] Stage 4 — RBAC-filtered retrieval + Groq generation chain (`rag_chain.py`)
- [x] Stage 5 — Guardrails (`guardrails/scope.py`, `guardrails/pii.py`)
- [x] Stage 6 — CLI chatbot entrypoint (`chatbot.py`)
- [x] Stage 7 — Manual end-to-end verification across roles
- [x] Stage 8 — Groq API key checkpoint (create key, add to `.env`)
- [x] Stage 9 — Web UI (FastAPI backend + custom frontend)
- [x] Stage 10 — Azure deployment (Container Apps) — see `DEPLOYMENT.md`

---

## Stage 0 — Environment & Dependencies

Installed into `RAG_project.venv`:

| Package | Role |
|---|---|
| `langchain` | Core orchestration — prompt templates, LCEL (`\|` pipe composition) |
| `langchain-community` | Document loaders (Markdown, CSV, etc.) |
| `langchain-groq` | `ChatGroq` wrapper — talks to Groq Cloud's API |
| `langchain-chroma` | LangChain ↔ Chroma integration |
| `chromadb` | Embedded vector database — stores chunks + embeddings, supports metadata filtering |
| `langchain-huggingface` | Wraps local HuggingFace embedding models for LangChain |
| `sentence-transformers` | Runs the actual embedding model locally on CPU (no API key) |
| `python-dotenv` | Loads secrets (`GROQ_API_KEY`) from a `.env` file |
| `pandas` | Parses `hr_data.csv` row-by-row for the custom HR loader |

Installed with LangChain 1.3.x (the newer "LCEL-first" major version) — legacy chain classes now
live in a separate `langchain-classic` package; we're intentionally not using those and building
chains with LCEL directly instead.

---

## Stage 1 — How RAG Actually Works

**The problem RAG solves.** An LLM only knows what was in its training data — it has never seen
FinSolve's HR spreadsheet or Q3 marketing report. RAG (Retrieval-Augmented Generation) fixes this
*without retraining the model*: at question time, we search our own documents for the relevant
passages and paste them into the prompt as context. The LLM then answers "closed-book" from that
pasted text, not from its own memory.

The pipeline has two phases:

### A) Ingestion (offline, done once / whenever data changes)
```
raw files → chunking → embedding → store in vector DB
```
1. **Chunking** — split each document into small pieces (a few paragraphs each). Why not embed a
   whole 10-page report as one block? Retrieval works by similarity of the *whole chunk* to the
   question — if a chunk mixes five unrelated topics, its "average meaning" matches nothing well.
   Small, focused chunks retrieve precisely.
2. **Embedding** — run each chunk through an embedding model that converts text into a vector (a
   list of ~384 numbers, for our model). Semantically similar text produces vectors that are close
   together in that 384-dimensional space — e.g. "employee salary" and "compensation figures" land
   near each other even though they share no words.
3. **Storing** — save each chunk's vector + original text + metadata (source file, department) into
   Chroma. This is what makes step B fast: instead of comparing a question to every document, Chroma
   finds nearest-neighbor vectors efficiently.

### B) Query time (every user question)
```
question → embed question → similarity search in vector DB → top-k chunks →
stuff chunks + question into a prompt → LLM generates answer
```
1. Embed the user's question with the *same* embedding model used at ingestion.
2. Ask Chroma: "give me the k chunks whose vectors are closest to this question's vector."
3. Build a prompt: *"Answer using only this context: [retrieved chunks]. Question: [question]"*
4. Send that prompt to the Groq-hosted LLM, which generates the final answer grounded in the
   retrieved text.

### Where RBAC plugs in
Step B.2 (`similarity search`) is exactly where we insert a **metadata filter** — e.g. "only search
chunks where `department` is in `[hr, general]`." Chroma then physically excludes non-permitted
chunks from the search space, so retrieval can never surface a document a role isn't allowed to see.
This is why every chunk gets tagged with `department` metadata during ingestion — it's the hook RBAC
hangs on.

### Where guardrails plug in
- **Scope guardrail**: looks at the similarity *scores* from step B.2 — if even the best match is a
  poor match, the question is probably off-topic, so generation is skipped entirely.
- **PII guardrail**: runs after step B.4, scanning the generated answer text before it's shown to
  the user (defense in depth, in case RBAC filtering is ever bypassed or the model is tricked via
  prompt injection).

---

## Stage 2 — Ingestion Pipeline

File: `ingest.py`. No Groq key needed here — embeddings run locally.

**The problem with one splitting strategy:** the source `.md` files aren't uniformly structured.
Most use proper `#`/`##`/`###`/`####` headers, but two files (`financial_summary.md`,
`marketing_report_2024.md`) only have a title and plain-text section labels with no `#` markup.
So a **two-pass split** is used:

1. `MarkdownHeaderTextSplitter` — splits on header levels 1-4, and for each resulting piece
   records which headers it fell under (e.g. `Header 2: "Q2..."`, `Header 3: "Cash Flow
   Analysis"`). A file with no headers just comes back as one big piece — handled by pass 2.
2. `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=150) — runs on pass 1's
   output and further splits anything still too long. It's a no-op on pieces already small enough.

**Self-contained chunks:** the header breadcrumb (e.g. `Q2 - April to June 2024 > Cash Flow
Analysis`) is prepended into the chunk's *text*, not just left in metadata — a chunk retrieved in
isolation (which is what happens at query time) should be able to say what it's about without
needing the rest of the document.

**hr_data.csv gets a different treatment entirely:** embeddings work on natural language, not raw
comma-separated fields, so each row is turned into one readable sentence per employee (e.g.
*"Aadhya Patel (employee ID FINEMP1000) works as a Sales Manager... Annual salary:
1,332,478.37..."*) — one Document per employee, no further splitting needed.

**Every chunk gets tagged** `metadata = {"department": <folder name>, "source": <filename>}`
(HR rows also get `employee_id`). `department` is the field RBAC filtering keys off in Stage 4 —
department names are taken directly from the `data/` subfolder names (`engineering`, `finance`,
`general`, `hr`, `marketing`), so ingestion and the access-control matrix stay in sync automatically.

**Idempotency:** the script deletes and rebuilds `chroma_db/` on every run rather than appending,
so re-running it after editing source docs never leaves stale/duplicate chunks behind.

**Result of first run:** 331 chunks total — engineering=83, finance=32, general=56, hr=100,
marketing=60 — embedded with local `sentence-transformers/all-MiniLM-L6-v2` and persisted to
`chroma_db/`.

---

## Stage 3 — RBAC Layer

File: `rbac.py`. Deliberately just data + one lookup function, not scattered logic — the whole
access policy should be auditable by reading one dict.

**`ROLE_DEPARTMENTS`** — the access matrix:

| Role         | Allowed departments                                  |
|--------------|-------------------------------------------------------|
| finance      | finance, marketing, general                          |
| marketing    | marketing, general                                    |
| hr           | hr, general                                            |
| engineering  | engineering, general                                   |
| c_level      | engineering, finance, general, hr, marketing (all)     |
| employee     | general                                                 |

**`DEMO_USERS`** — one hardcoded user per role, standing in for real auth (SSO/JWT in a real
deployment). Everything from `ROLE_DEPARTMENTS` onward doesn't care how the role was determined.

**`get_allowed_departments(role)`** — the single function every retrieval call goes through. If
the storage shape of the matrix ever changes, only this function needs to change.

---

## Stage 4 — Retrieval + Generation Chain

File: `rag_chain.py`. Split into two functions on purpose:

**`retrieve_with_scores(question, role, k)`** — needs no Groq key (local embeddings only). Calls
`rbac.get_allowed_departments(role)` and passes it straight into Chroma's search as
`filter={"department": {"$in": allowed_departments}}`. This is the actual enforcement point: the
filter is applied *inside* the vector search, so non-permitted chunks are structurally excluded,
not filtered after the fact. Returns `(Document, score)` pairs — score is cosine similarity, 0-1,
higher = more relevant (meaningful because of the cosine collection config from Stage 2).

**`generate_answer(question, docs)`** — needs `GROQ_API_KEY` (lazy-loaded so importing this module
doesn't fail without a key; only calling this function does). Composed as an **LCEL chain**:
```
prompt_template | ChatGroq | StrOutputParser()
```
`|` pipes each stage's output into the next: prompt fills `{context}`/`{question}` → `ChatGroq`
sends it to Groq Cloud → `StrOutputParser` extracts the plain string from the response object. The
system prompt explicitly restricts the model to the supplied context ("if the context doesn't
contain the answer, say so — do not use outside knowledge"), which is what keeps answers grounded.

Default model: `llama-3.3-70b-versatile` (overridable via `GROQ_MODEL` env var).

**Sanity-checked retrieval (no Groq key required) before moving on:**
- `hr` and `marketing` roles asked the same payroll question both got the same *general* handbook
  policy chunk (that content is department=`general`, so it's shared) — but `marketing` never saw
  the `hr`-tagged individual employee salary rows that a more specific query would surface. Confirms
  RBAC filtering is structural, not just "the model chose not to answer."
- In-scope queries scored ~0.57-0.64 relevance; an off-topic query ("write me a poem about the
  ocean") scored ~0.10-0.17 — a clear gap to build the Stage 5 scope guardrail's threshold on
  (settled on 0.35 as the cutoff).

---

## Stage 5 — Guardrails

Files: `guardrails/scope.py`, `guardrails/pii.py`.

**Scope guardrail** — trivial given Stage 4's findings: take the best score from
`retrieve_with_scores()`, and if it's below `0.35`, return a fixed refusal without calling the LLM
at all. No LLM call needed = also saves latency/cost on junk questions.

**PII guardrail — the key design decision was what *not* to redact.** A naive "redact any
number/currency" rule would break the product: `financial_summary.md` legitimately contains `$30M`
type figures the `finance` role is *supposed* to see. Blanket redaction can't distinguish "aggregate
company financial figure" from "one person's private salary." So detection is scoped to patterns
that are inherently personal:
- email addresses, phone numbers, employee IDs (`FINEMP####` scheme)
- salary figures **only when explicitly labeled** (`"salary: 1,332,478.37"`), not any dollar amount

**It's a defense-in-depth layer, not the primary control.** RBAC filtering in Stage 4 is the real
gate — a non-HR/non-C-level role should never even retrieve HR chunks. `redact_pii(text, role)` is
a no-op for `hr`/`c_level` (they're authorized to see this data; redacting it for them would be
wrong, not safer) and only does real work as a safety net for edge cases like prompt injection or a
future retrieval bug.

**Verified in isolation:**
- `marketing` role: email/phone/employee-ID/salary all redacted from a sample HR sentence.
- `hr` role: same sentence passes through unchanged (exempt).
- `finance` role: `"Vendor Services totaled $30M..."` passes through **untouched** — confirms the
  salary pattern's label requirement prevents over-redaction of legitimate financial figures.
- Scope check: empty results → out-of-scope; low scores (~0.1-0.2) → out-of-scope; high scores
  (~0.5-0.6) → in-scope.

---

## Stage 6 — CLI Chatbot

File: `chatbot.py`. Pure wiring, no new concepts — this is the loop that ties every prior stage
together per question:

```
retrieve_with_scores(question, role)      # Stage 4, RBAC-filtered
    -> is_out_of_scope(results)?          # Stage 5 guardrail -- short-circuit, no LLM call
    -> generate_answer(question, docs)    # Stage 4, needs GROQ_API_KEY
    -> redact_pii(answer, role)           # Stage 5 guardrail, safety net
    -> print answer + cited sources
```

`choose_user()` lists `rbac.DEMO_USERS` and lets you pick one by number — that becomes the `role`
threaded through the whole loop. Sources are printed from the retrieved chunks' `metadata["source"]`
so you can always see which documents an answer was grounded in (and, implicitly, verify it never
cites a document outside your role's allowed departments).

**Smoke-tested the full pipeline before adding a Groq key:**
- In-scope question ("What is the HR leave policy?") passed the scope guardrail, then failed
  *gracefully* at the generation step with a clear "add GROQ_API_KEY to .env" message instead of a
  stack trace.
- Out-of-scope question ("write me a poem about the ocean") was caught by the scope guardrail and
  never reached the generation step at all.

---

## Stage 7 — End-to-End Verification (live Groq generation)

Ran real (question, role) pairs through the full pipeline with a live Groq key.

| Test | Result |
|---|---|
| `hr` asks for an employee's salary + leave info | Answered correctly from `hr_data.csv` (salary, leave balance). Retrieved chunks: `department=hr` only. |
| `marketing` asks the *same* salary question | Correctly answered "I don't have that information" — retrieval only surfaced `general` handbook content, **zero `hr` chunks reached the model**. This is the core RBAC guarantee working: the model never saw the data to refuse it manually, it structurally wasn't given the data. |
| `finance` asks about marketing vendor-expense figures | Answered correctly ($30M vendor services, 40% marketing-related → $12M), pulling from **both** `finance` and `marketing` department chunks — matches the original spec ("finance gets financial reports **and** marketing expenses"). |
| `c_level` asks single-topic questions across engineering / HR / finance | Engineering and finance answered correctly from their respective departments; HR leave-policy question retrieved the right *document* (`general`) but missed the specific content chunk — a retrieval-recall limitation (see below), not an access-control failure. |
| `employee` asks an out-of-scope question ("What's the capital of France?") | Caught by the scope guardrail before any LLM call — fixed refusal message returned. |
| Automated leak check on every test | For every role/question pair, `departments_in_retrieved_chunks - role's_allowed_departments` was empty. No RBAC leak observed in any test. |

**Observed limitation (not a bug, a real RAG characteristic worth knowing):** a single compound
question spanning multiple topics (e.g. "summarize engineering architecture, HR leave policy, and
Q4 marketing highlights" in one message) can retrieve unevenly — with `k=4`, all 4 slots got
consumed by the most semantically dominant sub-topic (marketing), starving the other two topics
entirely. This is a known weakness of single-embedding similarity search on multi-part questions,
not an RBAC issue (c_level *was* allowed all of it; retrieval just didn't fetch it). Real fixes if
this mattered in production: raise `k`, or decompose compound questions into sub-queries before
retrieval (multi-query retrieval) — left as a possible future improvement, out of scope for this
learning build.

**Conclusion:** RBAC enforcement, the scope guardrail, and the PII guardrail all behave as designed
end-to-end, including the specific access rules requested (finance ↔ marketing-expense visibility,
HR/payroll restricted to HR + C-level, off-topic questions rejected without an LLM call).

---

## Stage 9 — Web UI (FastAPI + custom frontend)

Files: `app.py` (backend), `web/index.html` (frontend). Reuses the **entire** existing pipeline
unchanged — `rag_chain`, `rbac`, `guardrails` are imported as-is; the web layer adds nothing to the
RAG/RBAC/guardrail logic.

**Backend (`app.py`, FastAPI):**
- `GET /api/users` — public, lists demo users for the login screen.
- `POST /api/login` — validates the user, issues an opaque session token, stores `token → username`
  in an in-memory dict, returns the user + their allowed departments.
- `POST /api/chat` — resolves role **from the token server-side**, then runs the same
  retrieve → scope-guard → generate → PII-redact flow the CLI uses.
- `POST /api/logout`, plus static serving of `web/index.html` at `/`.

**Key security decision:** the browser never sends its own `role`. Authorization is derived from the
server-side session token on every request. If the client could put `role: "c_level"` in the request
body, RBAC would be trivially bypassable — **authz must be decided by the server, never trusted from
the client.** Verified this: a request with a bogus token gets `401`, not an answer.

**Frontend (`web/index.html`, single self-contained file):** login screen with per-role colored
user cards → chat view with the logged-in user's role badge + allowed-department chips, message
bubbles, source chips under each answer, distinct amber styling for guardrail/out-of-scope replies,
a typing indicator during Groq calls, role-aware suggested prompts, light/dark theme toggle, and
logout. No external assets/CDNs — works offline, served straight from FastAPI.

**End-to-end verified over HTTP:**
- `finance` login → marketing-expense question answered with sources. ✅
- `marketing` login → salary question denied, no HR sources leaked. ✅
- off-topic question → `out_of_scope: true`. ✅
- bogus session token → `401`. ✅

**Run it:**
```
RAG_project.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```
then open http://127.0.0.1:8000

---

## Stage 10 — Azure Deployment (Container Apps)

Full command-by-command runbook lives in `DEPLOYMENT.md`. Key ideas:

**Everything ships as one Docker image** (`Dockerfile`): code + deps + embedding-model weights +
a pre-built `chroma_db/` (ingestion runs *during the image build*, via `RUN python ingest.py`). So
the container starts with its vector store ready and needs no runtime downloads. The only runtime
input is `GROQ_API_KEY`, injected as an Azure secret.

**No local Docker needed** — `az acr build .` uploads the build context and builds the image on an
Azure Container Registry build agent, then Azure Container Apps runs it with a free HTTPS URL.

**Two app-specific gotchas baked into the process:**
- **CPU-only torch**: the `Dockerfile` installs torch from the CPU wheel index first, because the
  default torch wheel bundles CUDA and is multiple GB (no GPU in Container Apps).
- **Single replica pin** (`--min-replicas 1 --max-replicas 1`): sessions live in an in-memory dict
  in `app.py`, so multiple replicas would each hold separate session tables and break logins. The
  documented hardening path (Redis/JWT sessions + Entra ID auth) is what lifts this restriction.

`.dockerignore` keeps `.env` and the multi-GB `RAG_project.venv/` out of the build context — and
deliberately does **not** exclude `*.md`, since `data/` documents are markdown that ingestion needs.

---

## Deferred (explicitly out of scope for now)
- Deployment: containerizing, hosting, real authentication (SSO/JWT) in place of `DEMO_USERS`.
- Retrieval-recall improvements for compound/multi-topic questions (see Stage 7 note above).
