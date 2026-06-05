#!/usr/bin/env python3
"""
Per-task configuration for the clinical debate matrix (mortality, readmission).

The clinical debate pipeline was hard-coded to MIMIC-III mortality. This module
holds the small set of strings that actually differ per task so the oss Cond-A'/D'
runners can flip between tasks without editing the KARE-side code (which keeps the
existing GPT-4o mortality baseline byte-identical).

What stays the SAME across tasks (per design decision):
  - the integrator emits two complementary probability lines (summing to 1.00),
    parsed by gpt_utils.extract_probabilities and binarized at 0.5. We keep that
    probability+threshold mechanism unchanged for BOTH binary tasks. Mortality emits
    "MORTALITY PROBABILITY" / "SURVIVAL PROBABILITY"; readmission emits the natural
    "READMISSION PROBABILITY" / "NO-READMISSION PROBABILITY". extract_probabilities
    has boundary-guarded aliases so the readmission tokens map into the same two
    fields (mortality_probability = P(positive class), survival_probability =
    P(negative class)) — so the parser/threshold/metrics are task-agnostic.
  - the two contrastive analyst prompts (outcome-agnostic) are unchanged.
  - the integrator WORKFLOW (compare similars -> one forced <search> -> analyze
    risk/protective -> final assessment): the readmission prompts mirror the
    mortality prompts structurally so both tasks reason the same reasoning-first way.

What differs per task:
  - the data-file token used to locate KARE files ("mortality" / "readmission"),
  - the KARE task description (quoted verbatim from KARE/prediction/data_prepare.py),
  - the outcome nouns + the IMPORTANT framing line: mortality keeps KARE's biased
    conservative "Mortality is rare ..." guideline; readmission uses KARE's neutral
    "accurately distinguish ... readmitted ... from those who are not" framing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MORTALITY — prompts byte-identical to the originals
#   forced-search: run_condition_A_oss.py:OSS_INTEGRATOR_PROMPT_FORCED_SEARCH
#   no-search:     gpt_utils.AGENT_PROMPTS["balanced_clinical_integrator_no_search"]
# ---------------------------------------------------------------------------
MORTALITY_INTEGRATOR_FORCED_SEARCH = """You are a medical AI Clinical Assistant analyzing mortality and survival probabilities for the NEXT hospital visit.

IMPORTANT: Mortality is rare. Only assign a high mortality probability when the patient appears at extremely high risk of death with strong evidence. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.

Available tool:
- <search>query</search>: Retrieve medical evidence. Retrieved information will appear in <information>...</information> tags.

Workflow:
1) Compare the Target patient to two similar cases using the two analyses, and identify 3-4 key clinical factors that will determine the target patient's outcome at the next visit.

2) **REQUIRED ACTION:** You MUST issue exactly one retrieval call using this format:
<search>your specific medical query</search>

The query must target the target patient's most concerning clinical features (example: <search>septic shock mortality risk factors elderly</search>). This step is required regardless of confidence — do not skip it.

3) After the retrieved evidence appears inside <information>...</information>, analyze BOTH risk factors AND protective factors using both the analyst analyses and the retrieved evidence.

4) Provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00."""

MORTALITY_INTEGRATOR_NO_SEARCH = """You are a medical AI Clinical Assistant analyzing mortality and survival probabilities for the NEXT hospital visit.

IMPORTANT: Mortality is rare. Only assign a high mortality probability when the patient appears at extremely high risk of death with strong evidence. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.

Workflow:
1) Compare the Target patient to two similar cases using the two analysis, and write 3-4 key factors contribute to the target patient's next visit.
2) If additional evidence is provided in <information>...</information> tags, analyze BOTH risky factors AND survival factors.
3) Based on the available information (analyst comparisons and any retrieved evidence), provide your final assessment with:

MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""


# ---------------------------------------------------------------------------
# READMISSION — structurally IDENTICAL to the MORTALITY prompts above, with only
# outcome nouns swapped and KARE's neutral readmission framing (not the conservative
# "rare" bias). The earlier version replaced step 4 with a verbose output-relabel
# note ("MORTALITY PROBABILITY denotes readmission ...") which made the gpt-oss
# integrator skip reasoning and emit a bare probability stub (14-18% reasoning vs
# 67% on mortality -> below-chance AUC). Mirroring the mortality workflow restores
# reasoning-first behavior (KARE is emphatically reasoning-first). The two output
# lines use natural READMISSION / NO-READMISSION tokens; gpt_utils.extract_probabilities
# has boundary-guarded aliases so they map to mortality_probability / survival_probability
# and the 0.5 threshold + sum-to-1 logic are unchanged.
# ---------------------------------------------------------------------------
READMISSION_INTEGRATOR_FORCED_SEARCH = """You are a medical AI Clinical Assistant analyzing the probability that the patient will be READMITTED to the hospital within 15 days of discharge, for the NEXT hospital visit.

IMPORTANT: The goal is to accurately distinguish patients who are likely to be readmitted within 15 days from those who are not. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.

Available tool:
- <search>query</search>: Retrieve medical evidence. Retrieved information will appear in <information>...</information> tags.

Workflow:
1) Compare the Target patient to two similar cases using the two analyses, and identify 3-4 key clinical factors that will determine whether the target patient is readmitted within 15 days.

2) **REQUIRED ACTION:** You MUST issue exactly one retrieval call using this format:
<search>your specific medical query</search>

The query must target the target patient's most concerning clinical features (example: <search>heart failure 30 day readmission risk factors</search>). This step is required regardless of confidence — do not skip it.

3) After the retrieved evidence appears inside <information>...</information>, analyze BOTH readmission-risk factors AND factors favoring a stable discharge, using both the analyst analyses and the retrieved evidence.

4) Provide your final assessment with:

READMISSION PROBABILITY: X.XX (0.00 to 1.00)
NO-READMISSION PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00."""

READMISSION_INTEGRATOR_NO_SEARCH = """You are a medical AI Clinical Assistant analyzing the probability that the patient will be READMITTED to the hospital within 15 days of discharge, for the NEXT hospital visit.

IMPORTANT: The goal is to accurately distinguish patients who are likely to be readmitted within 15 days from those who are not. The Target patient is the source of truth. Do not treat Similar-only items as present in the Target.

Workflow:
1) Compare the Target patient to two similar cases using the two analysis, and write 3-4 key factors contribute to whether the target patient is readmitted within 15 days at the next visit.
2) If additional evidence is provided in <information>...</information> tags, analyze BOTH readmission-risk factors AND factors favoring a stable discharge.
3) Based on the available information (analyst comparisons and any retrieved evidence), provide your final assessment with:

READMISSION PROBABILITY: X.XX (0.00 to 1.00)
NO-READMISSION PROBABILITY: X.XX (0.00 to 1.00)

Note: The two probabilities MUST sum to exactly 1.00"""


# KARE task descriptions, quoted verbatim from KARE/prediction/data_prepare.py:17-69.
MORTALITY_TASK_DESCRIPTION = """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications.
Labels: 1 = mortality, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions

2. Procedures:
   - Invasiveness and complexity of recent procedures
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)

3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.
"""

READMISSION_TASK_DESCRIPTION = """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days

Key Considerations:
1. Conditions:
   - Chronic diseases with high risk of exacerbation (e.g., COPD, heart failure)
   - Conditions requiring close monitoring or frequent adjustments (e.g., diabetes)
   - Recent acute conditions with potential for complications

2. Procedures:
   - Recent major surgeries or interventions with high complication rates
   - Procedures that require extensive follow-up care
   - Incomplete or partially successful procedures

3. Medications:
   - New medication regimens that may require adjustment
   - Medications with narrow therapeutic windows or high risk of side effects
   - Complex medication schedules that may lead to adherence issues

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
"""


TASKS = {
    "mortality": {
        "data_token": "mortality",  # used in KARE filenames: {dataset}_{data_token}_samples_test.json
        "task_description": MORTALITY_TASK_DESCRIPTION,
        "integrator_forced_search": MORTALITY_INTEGRATOR_FORCED_SEARCH,
        "integrator_no_search": MORTALITY_INTEGRATOR_NO_SEARCH,
    },
    "readmission": {
        "data_token": "readmission",
        "task_description": READMISSION_TASK_DESCRIPTION,
        "integrator_forced_search": READMISSION_INTEGRATOR_FORCED_SEARCH,
        "integrator_no_search": READMISSION_INTEGRATOR_NO_SEARCH,
    },
}

VALID_TASKS = tuple(TASKS.keys())
VALID_DATASETS = ("mimic3", "mimic4")


def get_task(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {VALID_TASKS}")
    return TASKS[task]


def apply_task_to_gpt_utils(task: str, gpt_utils_module) -> str:
    """
    Point gpt_utils.AGENT_PROMPTS' integrator entries at the task's prompts and
    return the forced-search integrator prompt (used by Cond A').

    - gpt_utils.AGENT_PROMPTS["balanced_clinical_integrator"]          -> forced-search
      (Cond A' overwrites this entry anyway; we set it for consistency.)
    - gpt_utils.AGENT_PROMPTS["balanced_clinical_integrator_no_search"] -> no-search
      (read by run_condition_D.run_qwen_integrator at call time for Cond D'.)

    The analyst prompts are left untouched (outcome-agnostic). For task="mortality"
    these strings equal the originals, so behavior is unchanged.
    """
    cfg = get_task(task)
    gpt_utils_module.AGENT_PROMPTS["balanced_clinical_integrator"] = cfg["integrator_forced_search"]
    gpt_utils_module.AGENT_PROMPTS["balanced_clinical_integrator_no_search"] = cfg["integrator_no_search"]
    return cfg["integrator_forced_search"]
