# Paper Outline — DAIH Workshop (8 pages)

> Working title: **"Who Does the Reasoning? A Factorial Study of Open-Weight Multi-Agent
> Debate for Clinical Risk Prediction."**
>
> Status convention throughout the paper: a cell or claim that depends on an experiment still
> **running or missing** is marked **`-`**. Replace with the number once the run lands.

---

## The argument in one paragraph

LLM multi-agent "debate + RAG" pipelines are widely proposed for clinical prediction, but two
questions are unanswered: (1) **does the debate structure earn its cost** over plain single-agent
RAG, and (2) **which agent role actually drives the gains** — and can that role be filled by an
open-weight model? We answer both with a controlled factorial over a 4-call pipeline (two
contrastive label-blind analysts → retrieval → integrator). We find the **integrator** is the
decisive position: swapping only it from a frontier-scale model to a local open-weight model
(Cond D′) lifts recall +50pp at equal discrimination. We then map where this holds on a full
deployment-scale grid (mortality/readmission × MIMIC-III/IV) and show — honestly — that the gain
is largely a **better operating point**, not better discrimination (AUC is near-identical across
integrators). Net: a fully open-weight debate pipeline matches a frontier-mixed one, and the
debate structure itself beats single-agent RAG.

**Four empirical pillars** (in narrative order):
1. **Debate ablation** — does the contrastive 2-analyst structure beat single-agent RAG? *(NEW; Cond H′)*
2. **Factorial attribution** — which of the 3 swappable positions matters? *(n=100, done)*
3. **Clinical generalization matrix** — where does it hold at full scale? *(mortality done; readmission running)*
4. **QA breadth** — does it transfer beyond mortality, to medical QA? *(MIRAGE, scheduled on ARC)*

---

## Section-by-section structure & page budget

| § | Section | Pages | What fills it | Pillar |
|---|---|---|---|---|
| — | **Abstract** | 0.15 | The one-paragraph argument above, condensed | all |
| 1 | **Introduction** | 1.0 | Clinical risk prediction with LLMs; the "debate+RAG" trend; the two unanswered questions; our contributions (bulleted) | — |
| 2 | **Related Work** | 0.75 | KARE (our base); multi-agent debate (MAD, MedAgents, ReConcile); clinical EHR SOTA (ColaCare/AdaCare/ConCare — §2.6 ref table); medical RAG (MedRAG / MIRAGE) | — |
| 3 | **Method** | 1.25 | The 4-call pipeline (2 contrastive label-blind analysts → query-gen+retrieval → integrator); the **factorial design** (3 swappable positions, cached reuse); condition catalog A′–G + H′; why **AUC** is the primary metric on imbalanced cells | — |
| 4 | **Experimental Setup** | 0.75 | Datasets (MIMIC-III/IV mortality+readmission test sets; MIRAGE QA); models (gpt-oss-120b via API, Qwen2.5-7B local); retrieval (MedRAG MedCorp2 / MedCPT); base rates; prompt regime (KARE biased mortality / neutral readmission) | — |
| 5 | **Results** | 2.75 | Four subsections below | 1–4 |
| 5.1 | — Debate vs single-agent | 0.6 | Cond H′ vs A′ across 4 cells; CoT row isolates RAG from debate | **1** |
| 5.2 | — Factorial attribution | 0.7 | n=100 table A′–G; the integrator-swap +50pp recall; model-family invariance vs GPT-4o | **2** |
| 5.3 | — Generalization matrix | 0.9 | Full-set A′/D′ × {mort,readm} × {mimic3,mimic4}; Acc/Rec/F1/AUC/majority; the discrimination-vs-threshold finding | **3** |
| 5.4 | — QA breadth (MIRAGE) | 0.55 | Debate+RAG / +CoT gap-fill across 5 QA datasets, 3 models | **4** |
| 6 | **Discussion** | 0.75 | (a) Threshold vs discrimination — what the AUC parity really means; (b) open-weight sufficiency = the practical payoff; (c) calibration as the likely mechanism (forward-pointer, not a full study); (d) when debate helps vs when single-agent RAG suffices | — |
| 7 | **Limitations** | 0.35 | Single base model per family; MIMIC-only clinical; AUC modest in absolute terms; no prospective eval; cost not yet a controlled axis | — |
| 8 | **Conclusion** | 0.25 | Restate: integrator is the lever; open-weight is sufficient; debate beats single-agent | — |
| — | **References** | 0.5 | — | — |

**Total ≈ 8.0 pages.**

---

## Figures & tables (planned)

| ID | Type | Content | Status |
|---|---|---|---|
| Fig 1 | diagram | The 4-call pipeline with the 3 swappable positions highlighted | to draw |
| Tab 1 | table | **Factorial n=100** (Conds A′–G: Acc/Rec/F1) | ✅ data ready |
| Tab 2 | table | **Generalization matrix** (A′/D′ × 4 cells: Acc/Rec/F1/AUC/maj) | ⚠️ mortality ready; readmission `-` |
| Tab 3 | table | **Debate ablation** (H′ vs A′, 4 cells) | `-` H′ not yet run |
| Tab 4 | table | **MIRAGE** debate+RAG/+CoT gap-fill grid | `-` ARC scheduled |
| Fig 2 | plot | (optional/Discussion) reliability curves A′ vs D′ integrator — calibration mechanism | `-` if we add calibration |

---

## Open decisions for the author

- **Lead pillar.** Outline currently opens Results with the *debate ablation* (pillar 1) as the
  foundation, then factorial as the punchline. Alternative: lead with the factorial (strongest
  number) and use the ablation as justification. Pick one for the Intro's contribution ordering.
- **Calibration.** Currently a forward-pointer in Discussion, not a pillar. If a reviewer would
  demand the *why*, promote it to a 5th short subsection (free — re-analysis of saved probs).
- **Cost table.** Not in the budget above. If the workshop angle is efficiency, add a small
  calls/$/latency table to §5.1 or §6 (~0.25 pg) — would push total to ~8.25, trim elsewhere.
