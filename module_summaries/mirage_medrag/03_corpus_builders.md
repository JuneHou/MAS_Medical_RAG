# Module Summary: MedRAG Corpus Builders — Per-Corpus Indexing & Preprocessing Pipeline

**Source directory**: `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/`  
**Summary scope**: All corpus builder scripts, custom MIMIC mortality track, UMLS track, and the KARE adoption guide.  
**Do NOT read the giant JSON corpora** — all sizes cited are from directory listings and README statistics.

---

## Table of Contents

1. [Shared Conventions](#1-shared-conventions)
2. [Standard Corpus Builders](#2-standard-corpus-builders)
   - [PubMed](#21-pubmed)
   - [StatPearls](#22-statpearls)
   - [Textbooks](#23-textbooks)
   - [Wikipedia](#24-wikipedia)
3. [UMLS Custom Track](#3-umls-custom-track)
4. [MIMIC Mortality Custom Track — Core Custom Addition](#4-mimic-mortality-custom-track)
   - [Rationale](#41-rationale)
   - [mimic_mor.py — Corpus Builder](#42-mimic_morpy--corpus-builder)
   - [mimic_mor_faiss.py — FAISS Similarity Framework](#43-mimic_mor_faisspy--faiss-similarity-framework)
   - [MIMIC_MORTALITY_README.md — Authoritative Reference](#44-mimic_mortality_readmemd--authoritative-reference)
5. [KARE Adoption Guide](#5-kare-adoption-guide)
6. [mimic_mor.jsonl Record Format — Sample](#6-mimic_morjsonl-record-format--sample)
7. [On-Disk Corpus Structure](#7-on-disk-corpus-structure)
8. [Corpus Size Reference Table](#8-corpus-size-reference-table)

---

## 1. Shared Conventions

Every corpus builder follows the same output contract regardless of source:

- **Output unit**: JSONL files, one JSON object per line, in `corpus/<source>/chunk/`.
- **Required fields on every record**:
  - `id` — unique string key (e.g., `PMID:12345`, `MIMIC_MOR_P95057_V0`, `UMLS_R0_L0_C22`)
  - `title` — section/article/patient title
  - `content` — the text chunk
  - `contents` — `concat(title, content)`: title + ". " + content (or title + " " + content if title already ends in `.?!`)
- **Whitespace normalization**: `re.sub("\s+", " ", text)` applied everywhere.
- **Token/chunk target**: ~1,000 characters per chunk (200-character overlap) using `RecursiveCharacterTextSplitter` for sources that need splitting. MIMIC uses a different 5,000-token target (see section 4).
- **The `concat()` helper** (defined identically in every file, `mimic_mor.py:11-15`, `pubmed.py:10-14`, `statpearls.py:10-14`, `textbooks.py:10-14`, `wikipedia.py:10-14`, `umls.py:10-14`) handles the punctuation-aware join.

After chunking, the standard MedRAG retrieval pipeline (in `utils.py`, not covered here) encodes each chunk with a chosen encoder model (MedCPT-Article-Encoder, Contriever, or SPECTER) and builds a FAISS index at `corpus/<source>/index/<org>/<model>/faiss.index`.

---

## 2. Standard Corpus Builders

### 2.1 PubMed

**File**: `pubmed.py` (2.2 KB)  
**Input**: Gzip-compressed NLM XML baseline files from `corpus/pubmed/baseline/*.xml.gz` — the annual PubMed baseline snapshot.  
**Parsing**: A line-by-line XML scanner (no DOM parsing) extracts `<PMID>`, `<ArticleTitle>`, and `<AbstractText>` tags, skipping articles with blank abstracts. Each article becomes exactly one record; no chunking is applied because PubMed abstracts are naturally short (~395–1,897 characters, mean ~1,309 chars from notebook stats).  
**ID scheme**: `PMID:<pmid_string>`.  
**Output**: One JSONL file per source `.xml.gz` file in `corpus/pubmed/chunk/`, e.g., `pubmed23n0001.jsonl`. The notebook measured 23,898,701 snippets across 1,166 chunk files.  
**Key line**: `pubmed.py:57` — `json.dumps({"id": "PMID:"+str(ids[i]), "title": titles[i], "content": abstracts[i], "contents": concat(titles[i], abstracts[i])})`.

### 2.2 StatPearls

**File**: `statpearls.py` (4.6 KB)  
**Input**: NXML (NLM XML) article files from `corpus/statpearls/statpearls_NBK430685/` — StatPearls is NCBI's continuously updated clinical reference, distributed as a book collection.  
**Parsing**: Full XML tree parsing via `xml.etree.ElementTree`. The `extract()` function walks `<sec>` elements, tracking section title and sub-title (detected as a `<p><bold>` pattern). Paragraphs (`<p>`) are emitted as chunks; **smart merging** combines short consecutive paragraphs: if a paragraph is under 200 characters and together with the previous chunk would remain under 1,000 characters, it is merged into the prior record in place rather than creating a new one (`statpearls.py:59-63`). List items are handled similarly. No `RecursiveCharacterTextSplitter` is used — StatPearls sections are naturally granular.  
**ID scheme**: `<filename_stem>_<sequential_int>` (e.g., `NBK430685_1_0`).  
**Title scheme**: hierarchical `"ArticleTitle -- SectionTitle -- SubTitle"` path.  
**Output**: One JSONL per `.nxml` article in `corpus/statpearls/chunk/`. The notebook counted 352,155 snippets across 9,625 files.

### 2.3 Textbooks

**File**: `textbooks.py` (1.3 KB)  
**Input**: Plain-text `.txt` files from `corpus/textbooks/en/` — a curated set of 18 medical textbooks.  
**Parsing**: The entire file is read as a single string and split by `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` (`textbooks.py:19`). No section-aware parsing; the splitter tries paragraph, sentence, and character boundaries in order.  
**ID scheme**: `<filename_stem>_<chunk_index>`.  
**Title**: filename stem (the textbook name without `.txt`).  
**Output**: One JSONL per textbook in `corpus/textbooks/chunk/`. The notebook counted 125,847 snippets across 18 files.

### 2.4 Wikipedia

**File**: `wikipedia.py` (1.9 KB)  
**Input**: The `wikipedia/20220301.en` Hugging Face dataset, loaded via `datasets.load_dataset()` with `cache_dir="./corpus/wikipedia"` (`wikipedia.py:19`). This is the March 2022 English Wikipedia snapshot.  
**Parsing**: Each article's `text` field is split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`. Articles are streamed one at a time; chunks are batched into files of 10,000 articles each (`batch_size = 10000`, `wikipedia.py:26`).  
**ID scheme**: `<article_id>_<chunk_index>` (Wikipedia article numeric ID).  
**Output**: Files named `wiki20220301en<zero_padded_batch>.jsonl` in `corpus/wikipedia/chunk/`. The notebook counted 29,913,202 snippets across 646 files.  
**Note**: `hf_cache/` at `corpus/../hf_cache/` stores the Hugging Face dataset download cache for Wikipedia (and potentially other HF-sourced datasets). It is not a FAISS cache — it is the raw HF arrow/parquet cache that `datasets` writes automatically.

---

## 3. UMLS Custom Track

**File**: `umls.py` (5.5 KB)  
**README**: `UMLS_README.md` (~7 KB)

### What It Indexes

UMLS (Unified Medical Language System) is processed not from the raw UMLS terminology but from a **graph community detection output**. The input is a 416 MB `communities.json` file at `/data/wang/junh/datasets/umls2text/leiden_only_output/communities/communities.json`, produced by running the Leiden community detection algorithm on the UMLS knowledge graph. Each community record contains:
- `run` (integer 0–34), `level`, `community_id`
- `triples` — raw knowledge graph triples
- `summary` — an LLM-generated textual summary of the community's medical concepts and relationships

This makes the UMLS corpus qualitatively different from the others: instead of literature or educational text, it contains **structured knowledge summarized into prose**, describing relationships among UMLS concepts.

### Processing (umls.py)

1. Load all 35,797 communities from `communities.json`.
2. Filter invalid summaries: blank strings or those containing `"error"` or `"too large"` (3.3% of communities, i.e., 1,177 records skipped) — `umls.py:77-79`.
3. Group by `run` for batch organization — produces 35 JSONL files, one per run.
4. For each valid community, normalize whitespace, then check length:
   - If `len(summary) > 2000`: split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` (`umls.py:29`). Approximately 46.9% of summaries exceed 1,000 chars and need chunking.
   - If shorter: emit as a single record.
5. Hierarchical title format: `"UMLS -- Run {run} -- Level {level} -- Community {comm_id}"` (`umls.py:85`).
6. ID format: `UMLS_R{run}_L{level}_C{comm_id}` (single chunk) or `UMLS_R{run}_L{level}_C{comm_id}_{i}` (multi-chunk) — `umls.py:99,109`.

### Statistics (from UMLS_README.md)

| Metric | Value |
|---|---|
| Total communities | 35,797 |
| Valid communities | 34,620 (96.7%) |
| Communities requiring chunking | 16,235 (46.9%) |
| Total chunks created | 62,340 |
| Output files | 35 JSONL files in `corpus/umls/chunk/` |
| Total size | 123 MB |

### UMLS_README.md — Key Points

The README explains that UMLS communities complement text-based corpora with **structured knowledge**: each community summary describes semantically related medical concepts and their relationships (parent/child, co-occurrence, causal). The corpus was added to provide concept-level retrieval that pure text search misses. The README describes the three-stage indexing flow:
1. `embed()` in `utils.py` loads JSONL chunks, encodes each with the chosen encoder, and saves `.npy` embedding files to `corpus/umls/index/<retriever>/embedding/`.
2. `construct_index()` concatenates `.npy` files and builds a FAISS flat index (`IndexFlatIP` for MedCPT, `IndexFlatL2` for SPECTER), saving `faiss.index` and a `metadatas.jsonl` side-file mapping FAISS row indices back to source JSONL file and line number.
3. At query time, `get_relevant_documents()` encodes the query with the Query-Encoder variant and performs a top-k FAISS search.

The UMLS corpus is registered in `utils.py` under corpus alias `"UMLS"` and is also bundled into the `"MedCorpUMLS"` composite alias (`["pubmed", "textbooks", "statpearls", "wikipedia", "umls"]`).

---

## 4. MIMIC Mortality Custom Track

**Files**: `mimic_mor.py` (13 KB) and `mimic_mor_faiss.py` (22.9 KB)  
**README**: `MIMIC_MORTALITY_README.md` (~12 KB)  
**This is the most important custom addition in this repository.**

### 4.1 Rationale

All five standard MedRAG corpora (PubMed, StatPearls, Textbooks, Wikipedia, UMLS) contain general medical knowledge: research abstracts, clinical guidelines, educational text, or concept summaries. None of them contains **real patient cases with verified clinical outcomes**. The MIMIC mortality track fills this gap by indexing de-identified MIMIC-III EHR records as retrievable documents, enabling a downstream system (KARE / Debate) to answer questions like "find patients with similar clinical presentations and report their mortality outcomes."

The specific task being indexed is **next-visit mortality prediction**: given a patient's cumulative clinical history up to visit N, predict whether they will die in visit N+1. This framing comes from PyHealth's `mortality_prediction_mimic3_fn` task, which processes raw MIMIC-III discharge records (using `HOSPITAL_EXPIRE_FLAG`) and produces structured training/validation sets.

Data provenance chain:
```
MIMIC-III raw data
  -> PyHealth (mortality_prediction_mimic3_fn)
    -> KARE (cumulative visit aggregation, vocabulary standardization)
      -> MedRAG format (mimic_mor.py)
```

Vocabularies used: CCSCM (CCS-Multi-Level for conditions), CCSPROC (CCS procedures), ATC-3 (Anatomical Therapeutic Chemical for medications). These are the same vocabularies KARE uses internally.

### 4.2 mimic_mor.py — Corpus Builder

**Input files** (`mimic_mor.py:150-151`):
```
/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_train.json
/data/wang/junh/datasets/KARE/ehr_data/mimic3_mortality_samples_val.json
```

Each record in these JSON files represents a single cumulative patient snapshot:
- `patient_id`: format `{original_patient_id}_{visit_sequence}` — e.g., `"95057_1"` meaning patient 95057, cumulative snapshot after visit 1.
- `visit_id`: the visit index.
- `conditions`, `procedures`, `drugs`: lists-of-lists of standardized codes/names.
- `label`: 0 (survival) or 1 (mortality) for the **next** visit.

**Processing steps**:

1. Load both JSON files and group records by `patient_id` into a `defaultdict(list)`.
2. For each unique patient, call `chunk_patient_visits(visits, target_tokens=5000)` (`mimic_mor.py:195`).
3. The chunker:
   - Sorts visits by `visit_id` (ascending, preserving temporal order).
   - Estimates total token count using the 3-chars-per-token heuristic: `estimate_tokens(text) = len(text) / 3.0` (`mimic_mor.py:18-19`).
   - If total tokens ≤ 5,000: all visits become **one chunk** (ID suffix `_V<range>` or `_V<single>`).
   - If total tokens > 5,000: greedily partition visits into chunks, each staying under the 5,000-token budget. Chunk IDs are suffixed `_chunk_N`.
4. For each chunk, `format_medical_data(conditions, procedures, drugs)` (`mimic_mor.py:21-49`) flattens the nested lists and produces pipe-separated prose: `"Medical Conditions: ... | Medical Procedures: ... | Medications: ..."`.
5. Each visit within a chunk becomes: `"Visit {visit_id}: {formatted_data} | Mortality Risk: High/Low mortality risk"`.
6. Visits are joined with `" || "` separator. A temporal context prefix is prepended: `"Temporal Context: {N} visit(s) spanning visits {range}"`.
7. The `label` for a multi-visit chunk is the **last visit's label** (`mimic_mor.py:211`).

**Output** (`mimic_mor.py:154-165`): a single file `corpus/mimic_mor/chunk/mimic_mor.jsonl`.

**Key constants** (`mimic_mor.py:195`):
- `target_tokens = 5000` — chunking threshold
- Token estimate: `len(text) / 3.0` characters per token (`mimic_mor.py:18-19`)

**ID and title conventions**:
- Single visit: `id = "MIMIC_MOR_P{patient_id}_V{visit_id}"`, title = `"MIMIC-III Mortality Patient {patient_id} -- Visit {visit_id}"`
- Multi-visit (single chunk): `id = "MIMIC_MOR_P{patient_id}_V{start}-{end}"`, title includes visit count and range
- Multi-chunk patient: `id = "MIMIC_MOR_P{patient_id}_chunk_{N}"`, title includes `"(chunk_N of M)"`

**Statistics** (from README):
- Input: 8,721 patient visit records from 4,948 unique patients (train + val combined)
- Label distribution: 93.4% label-0 (low mortality risk), 6.6% label-1 (high mortality risk)
- 70.2% of patients have only a single visit; 29.8% have multiple visits (max 27)
- ~5% of patients exceed 5,000 tokens and require chunking
- Output: ~5,200 total documents, ~15 MB

### 4.3 mimic_mor_faiss.py — FAISS Similarity Framework

This file is a **design document and advisor discussion aid**, not the production builder. It contrasts two approaches and documents what would need to be adopted from KARE to implement a FAISS-based precomputed similarity corpus.

**Key constants** (`mimic_mor_faiss.py:27-30`):
```python
MAX_K = 2               # Similar patients per label class (positive/negative)
TASK = 'mortality'
DATASET = 'mimic3'
MAX_CONTEXT_LENGTH = 20000   # Character limit inherited from KARE
MAX_TOKEN_LIMIT = 5000       # MedRAG chunking target
```

**Approach 1 — Real-time retrieval** (already implemented as `mimic_mor.py`): Patients are indexed as JSONL chunks; at query time, MedRAG's standard text retrieval finds similar patients via embedding similarity. Simple, no precomputation, but purely text-semantic matching with no explicit label-aware filtering.

**Approach 2 — Precomputed FAISS similarity** (KARE-style, `mimic_mor_faiss.py`): Would precompute pairwise patient similarities offline, then annotate each patient's corpus record with a list of its top-K most similar patients (separated by label class: `positive` = same mortality label, `negative` = opposite label). The FAISS index used is `faiss.IndexFlatIP` (inner product, cosine similarity after L2 normalization) (`mimic_mor_faiss.py:280`). Embedding dimension: placeholder 384 (sentence-transformers/all-MiniLM-L6-v2), to be replaced with KARE's actual model. The similarity search retrieves 100 nearest neighbors, then filters down to 50 valid non-self patients, then separates by label class and keeps top `MAX_K` from each class (`mimic_mor_faiss.py:308-333`).

**Status of Approach 2**: Placeholder / not production-ready. The embedding generator uses random vectors (`mimic_mor_faiss.py:216`) and context generation calls a KARE function (`transform_patient_data` from `base_context.py`) that is not yet integrated. The precomputed similarity cache would be saved to `similarity_cache/patient_similarity_cache_mimic3_mortality.json` (`mimic_mor_faiss.py:367`).

**The `similarity_cache/` directory** at `src/data/similarity_cache/` is the intended destination for these precomputed per-patient nearest-neighbor JSON files — not FAISS index files themselves, but the JSON output of the FAISS search results so they can be quickly looked up at inference time without re-running the FAISS query.

**The `hf_cache/` directory** at `src/data/hf_cache/` stores Hugging Face model and dataset download caches (primarily Wikipedia and any HF-hosted encoder models). The notebook confirms both directories exist.

**Recommendation embedded in the file**: Short-term use Approach 1 for immediate integration; long-term adopt KARE's embedding pipeline for Approach 2 to enable label-aware clinical similarity retrieval.

### 4.4 MIMIC_MORTALITY_README.md — Authoritative Reference

The README is the primary reference document for the MIMIC mortality corpus. Key content:

**Pipeline description** (README lines 278–283):
```
MIMIC-III Raw Data  ->  PyHealth (mortality_prediction_mimic3_fn)
  ->  KARE (cumulative visit aggregation + vocabulary mapping)
    ->  MedRAG (chunking + JSONL emission = mimic_mor.py)
```

**Chunking decision rationale**: A token-length analysis of all patients found that 95% stay under ~5,094 tokens, so 5,000 tokens was chosen as the chunk boundary — preserving 95% of patients as single documents while only splitting the most complex cases.

**Token length statistics** for combined visits per patient (README lines 56–71):
- Min: 71 tokens, Max: 158,713 tokens, Mean: 1,575 tokens, Median: 604 tokens
- 95th percentile: 5,094 tokens; 99th percentile: 14,457 tokens
- Patients requiring chunking (>5K tokens): 278 (5.0%)

**Known limitations** (README lines 284–288):
- **Label artifacts**: 67% of patients with changing labels show impossible 1→0 transitions because PyHealth's "next-visit prediction" logic labels visit N based on what happens at N+1, creating apparent label reversals for patients with re-admissions.
- Vocabulary mapping to CCSCM/CCSPROC/ATC loses clinical nuance.
- Selection bias: only patients with sufficient EHR data quality pass PyHealth processing.

**Integration with utils.py** (README lines 162–172): The corpus is registered as:
```python
"MIMIC-MOR": ["mimic_mor"]
"MedCorpAll": ["pubmed", "textbooks", "statpearls", "wikipedia", "umls", "mimic_mor"]
```

**Clinical use cases identified**:
- Similar case retrieval for ICU/mortality risk assessment
- Evidence-based risk factor analysis across historical patient cohorts
- Temporal disease progression analysis
- Case-based reasoning in the downstream debate/multi-agent system

---

## 5. KARE Adoption Guide

**File**: `KARE_ADOPTION_GUIDE.md` (~4.5 KB)

### Plain-English Summary

The KARE project (`/data/wang/junh/githubs/Debate/KARE/`) is a multi-agent debate system for clinical outcome prediction. KARE already has its own FAISS-based patient similarity retrieval pipeline (`sim_patient_ret_faiss.py`) that retrieves similar historical patients for in-context learning. The MedRAG corpus builders in this repository extend MedRAG's retrieval layer so KARE's agents can also retrieve relevant **medical literature and knowledge** (PubMed abstracts, clinical guidelines, UMLS concept relationships) alongside the patient cases KARE already retrieves internally.

The adoption guide documents two integration paths:

**Path 1 (real-time, already done)**: `mimic_mor.py` converts KARE's pre-processed MIMIC data into standard MedRAG JSONL format. KARE's agent calls MedRAG's `Retriever` (with `corpus_name="MIMIC-MOR"` or `"MedCorpAll"`), which retrieves relevant patient chunks via embedding similarity — the same retrieval interface as for PubMed or StatPearls. This requires no modification to KARE's code.

**Path 2 (precomputed FAISS, planned)**: Adopt KARE's own `sim_patient_ret_faiss.py` logic — specifically its FAISS `IndexFlatIP` index building, label-aware positive/negative neighbor separation, and context-length filtering — into the MedRAG corpus builder to precompute patient-to-patient similarity relationships as metadata on corpus records. The guide identifies four KARE components that would need to be integrated:

| KARE component | Purpose | Status |
|---|---|---|
| `base_context.py::transform_patient_data()` | Converts raw patient records to text context | Not yet integrated |
| Embedding model (likely sentence-transformers) | Generates patient embeddings for FAISS | Not yet identified exactly |
| Resource files: `CCSCM.csv`, `CCSPROC.csv`, `ATC.csv` | Vocabulary mappings | Used in KARE; paths need updating |
| `sim_patient_ret_faiss.py` FAISS logic | Index building and similarity search | Documented in `mimic_mor_faiss.py` |

**What can be directly copy-pasted from KARE** (per guide sections "Direct Adoption"):
- `is_context_valid()` — filters out patients with contexts >20,000 characters
- FAISS index building block: `faiss.normalize_L2()` + `faiss.IndexFlatIP` + `index.add()`
- Similarity search loop: `index.search(embedding, 100)` then filter self, then split by label
- Label-aware top-K selection: sort positive and negative neighbors by cosine score, keep `MAX_K=2` from each class

**Current status**:
- Approach 1 (real-time): Complete and usable from KARE.
- Approach 2 (FAISS): Framework documented; needs `transform_patient_data()` and KARE's embedding model to be production-ready.

**Recommended path**: Use Approach 1 immediately for MedRAG integration and KARE consumption. Implement Approach 2 when research requires semantic clinical similarity matching beyond what keyword/embedding retrieval of free-text chunks provides.

---

## 6. mimic_mor.jsonl Record Format — Sample

**File**: `corpus/mimic_mor/chunk/mimic_mor.jsonl` (produced by `mimic_mor.py`)  
**Note**: There is also a `src/data/mimic_mor.jsonl` (31 MB, produced by `mimic_mor_faiss.py`'s `build_realtime_corpus()` in a slightly different format — see note at end of section).

### Sample records from `corpus/mimic_mor/chunk/mimic_mor.jsonl` (first 3 lines):

**Record 1 — single visit, label 0 (low mortality risk)**:
```json
{
  "id": "MIMIC_MOR_P10004_V0",
  "title": "MIMIC-III Mortality Patient 10004 -- Visit 0",
  "content": "Temporal Context: 1 visit(s) spanning visits 0 || Visit 0: Medical Conditions: other fractures, intracranial injury, epilepsy; convulsions, pleurisy; pneumothorax; pulmonary collapse, coronary atherosclerosis and other heart disease, diabetes mellitus without complication, e codes: fall, other gastrointestinal disorders | Medical Procedures: spinal fusion, laminectomy; excision intervertebral disc | Medications: i.v. solution additives, potassium supplements, anxiolytics, opioid analgesics, ... | Mortality Risk: Low mortality risk",
  "contents": "MIMIC-III Mortality Patient 10004 -- Visit 0. Temporal Context: ...",
  "label": 0
}
```

**Record 2 — single visit, label 1 (high mortality risk)**:
```json
{
  "id": "MIMIC_MOR_P1004_V0",
  "title": "MIMIC-III Mortality Patient 1004 -- Visit 0",
  "content": "Temporal Context: 1 visit(s) spanning visits 0 || Visit 0: Medical Conditions: other and ill-defined cerebrovascular disease, respiratory failure; insufficiency; arrest (adult), complications of surgical procedures or medical care, aortic and peripheral arterial embolism or thrombosis, ... | Medical Procedures: other or procedures on vessels other than head and neck, embolectomy and endarterectomy of lower limbs, ... | Medications: cardiac stimulants excl. cardiac glycosides, ... | Mortality Risk: High mortality risk",
  "contents": "MIMIC-III Mortality Patient 1004 -- Visit 0. Temporal Context: ...",
  "label": 1
}
```

**Record 5 — multi-visit patient (2 visits, label 0)**:
```json
{
  "id": "MIMIC_MOR_P1006_V0-1",
  "title": "MIMIC-III Mortality Patient 1006 -- 2 Visits (0-1)",
  "content": "Temporal Context: 2 visit(s) spanning visits 0-1 || Visit 0: Medical Conditions: respiratory failure..., cancer of bronchus; lung... | ... || Visit 1: Medical Conditions: cancer of bronchus; lung, cardiac dysrhythmias, ... | Medical Procedures: incision of pleura; thoracentesis; chest drainage, exploratory laparotomy, ... | Medications: opioid analgesics, ...",
  "contents": "MIMIC-III Mortality Patient 1006 -- 2 Visits (0-1). Temporal Context: ...",
  "label": 0
}
```

**Schema summary**:

| Field | Type | Description |
|---|---|---|
| `id` | string | `MIMIC_MOR_P{patient_id}_V{visit_range}` or `_chunk_{N}` |
| `title` | string | Human-readable patient + visit description |
| `content` | string | Temporal context prefix + pipe-delimited visit data joined with `\|\|` |
| `contents` | string | `concat(title, content)` — title + ". " + content |
| `label` | int | 0 = low mortality risk, 1 = high mortality risk (last visit's label) |

**Note on `src/data/mimic_mor.jsonl`**: This 31 MB file in the `src/data/` directory was produced by `mimic_mor_faiss.py`'s `build_realtime_corpus()` function from a different input file (`pateint_mimic3_mortality.json`). Its schema is slightly different — IDs use `mimic_mor_{patient_id}_chunk_{N}` format and content uses newline-separated visit formatting without the pipe-delimited structure. The canonical production JSONL for MedRAG retrieval is the one in `corpus/mimic_mor/chunk/mimic_mor.jsonl`.

---

## 7. On-Disk Corpus Structure

The corpus directory at `src/data/corpus/` holds both the processed chunk files and the FAISS index artifacts for each source. The canonical layout (documented in READMEs and confirmed by reading known-good paths):

```
src/data/corpus/
├── MedCorp_id2text.json          (~61 GB — combined PubMed+Textbooks+StatPearls+Wikipedia flat lookup)
├── MedCorp2_id2text.json         (~61 GB — variant or updated version of MedCorp)
├── PubMed_id2text.json           (~36 GB — PubMed-only flat lookup)
├── Wikipedia_id2text.json        (~25 GB — Wikipedia-only flat lookup)
├── StatPearls_id2text.json       (~231 MB)
├── Textbooks_id2text.json        (~112 MB)
├── UMLS_id2text.json             (~67 MB)
│
├── pubmed/
│   ├── baseline/                 # Raw .xml.gz downloads (NLM baseline)
│   │   └── pubmed23n0001.xml.gz  # (and ~1165 more)
│   └── chunk/
│       └── pubmed23n0001.jsonl   # (and ~1165 more, 23.9M snippets total)
│       └── ...
│
├── wikipedia/
│   ├── (HF dataset arrow cache files)
│   └── chunk/
│       ├── wiki20220301en000.jsonl
│       └── ... (646 files, 29.9M snippets total)
│
├── textbooks/
│   ├── en/
│   │   └── <textbook>.txt        # (18 plain-text textbooks)
│   └── chunk/
│       └── <textbook>.jsonl      # (18 files, 125K snippets total)
│
├── statpearls/
│   ├── statpearls_NBK430685/
│   │   └── *.nxml                # (9,625 StatPearls articles)
│   └── chunk/
│       └── *.jsonl               # (9,625 files, 352K snippets total)
│
├── umls/
│   └── chunk/
│       ├── umls_run00.jsonl      # (3,976 chunks, 6.7 MB)
│       ├── umls_run01.jsonl      # (2,711 chunks, 5.1 MB)
│       ├── ...
│       └── umls_run34.jsonl      # (1,424 chunks, 3.0 MB)
│                                 # Total: 35 files, 62,340 chunks, 123 MB
│
└── mimic_mor/
    └── chunk/
        └── mimic_mor.jsonl       # (~5,200 patient documents, ~15 MB)
```

**Index sub-directory layout** (same structure for all sources, documented in UMLS_README.md and MIMIC_MORTALITY_README.md):

```
corpus/<source>/
└── index/
    ├── ncbi/
    │   └── MedCPT-Article-Encoder/
    │       ├── embedding/
    │       │   └── <chunk_file>.npy     # one .npy per JSONL chunk file
    │       ├── faiss.index              # concatenated FAISS flat index
    │       └── metadatas.jsonl          # row-index -> (source_file, line_num) mapping
    ├── facebook/
    │   └── contriever/
    │       └── ... (same layout)
    └── allenai/
        └── specter/
            └── ... (same layout)
```

**FAISS index types** (from UMLS_README.md and mimic_mor_faiss.py):
- MedCPT: `faiss.IndexFlatIP` (inner product = cosine similarity after L2 normalization)
- SPECTER: `faiss.IndexFlatL2` (Euclidean distance)
- Contriever: `faiss.IndexFlatL2`

**Auxiliary directories**:
- `src/data/hf_cache/` — Hugging Face download cache (Wikipedia dataset arrow files, HF model weights). Written by `datasets.load_dataset()` and `transformers` automatically.
- `src/data/similarity_cache/` — Intended destination for precomputed patient-to-patient FAISS similarity JSON files (from `mimic_mor_faiss.py`'s Approach 2). Exists on disk but is currently empty or contains placeholder data, as Approach 2 is not production-ready.

---

## 8. Corpus Size Reference Table

| Corpus | Source Type | Chunk Count | Raw Size | Avg. Snippet Length |
|---|---|---|---|---|
| PubMed | Literature abstracts (NLM XML) | 23,898,701 | ~36 GB (flat JSON) | ~1,309 chars |
| Wikipedia | General encyclopedia (HF dataset) | 29,913,202 | ~25 GB (flat JSON) | ~682 chars |
| Textbooks | Medical textbooks (plain text) | 125,847 | ~112 MB (flat JSON) | ~777 chars |
| StatPearls | Clinical reference (NXML) | 352,155 | ~231 MB (flat JSON) | ~516 chars |
| UMLS | KG community summaries (LLM-generated) | 62,340 | 123 MB (chunks) | ~952 chars (median) |
| MIMIC-MOR | Patient EHR records (MIMIC-III) | ~5,200 | ~15 MB | variable (multi-visit) |

**Composite aliases** registered in `utils.py`:
- `"MedText"`: textbooks + statpearls
- `"MedCorp"`: pubmed + textbooks + statpearls + wikipedia
- `"MedCorpUMLS"`: pubmed + textbooks + statpearls + wikipedia + umls
- `"MIMIC-MOR"`: mimic_mor only
- `"MedCorpAll"`: pubmed + textbooks + statpearls + wikipedia + umls + mimic_mor

---

*Generated 2026-05-27 by Claude Code summarization agent. Source files read but not modified.*
