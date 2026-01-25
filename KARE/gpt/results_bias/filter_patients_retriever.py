#!/usr/bin/env python3
"""
Filter patient IDs based on retrieval/query problem criteria.

R1: "Retrieval swap breaks a correct GPT pipeline"
    - A correct, but B wrong

R2: "Qwen baseline positive, but B collapses to survival"
    - y=1, QB=1 (Qwen condition B prediction=1), but B=0 (GPT condition B prediction=0)

R3: "Qwen retrieval bundle looks suspicious"
    - Low query-context concept coverage
    - High doc redundancy
    - Low doc relevance (using embedding similarity scores)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(condition_path):
    """Load results.json from a condition folder."""
    with open(Path(condition_path) / "results.json", "r") as f:
        return json.load(f)


def load_log_file(log_path):
    """Load a patient log file."""
    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return None


def analyze_query_quality(query, patient_context_length_proxy=100):
    """
    Analyze query quality based on concept coverage.
    Returns a score (0-1, lower is worse).
    """
    if not query:
        return 0.0
    
    # Simple heuristic: query length and diversity
    query_terms = query.lower().split()
    unique_terms = set(query_terms)
    
    # Penalize very short queries (< 3 unique terms is suspicious)
    if len(unique_terms) < 3:
        return 0.3
    elif len(unique_terms) < 5:
        return 0.5
    else:
        return 0.8


def analyze_doc_redundancy(docs):
    """
    Analyze document redundancy based on content overlap.
    Returns redundancy score (0-1, higher is more redundant).
    """
    if not docs or len(docs) < 2:
        return 0.0
    
    # Extract meaningful content
    contents = []
    for doc in docs:
        if isinstance(doc.get("content"), dict):
            text = doc["content"].get("content", "")
        else:
            text = str(doc.get("content", ""))
        contents.append(text.lower())
    
    # Simple overlap metric: count of shared words
    redundancy_scores = []
    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            words_i = set(contents[i].split())
            words_j = set(contents[j].split())
            
            if len(words_i) == 0 or len(words_j) == 0:
                continue
            
            # Jaccard similarity
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            
            if union > 0:
                similarity = intersection / union
                redundancy_scores.append(similarity)
    
    if not redundancy_scores:
        return 0.0
    
    # Return average redundancy
    return np.mean(redundancy_scores)


def analyze_doc_relevance(docs):
    """
    Analyze document relevance based on retrieval scores.
    Returns average relevance score and identifies low-relevance docs.
    """
    if not docs:
        return 0.0, 0
    
    scores = []
    for doc in docs:
        score = doc.get("score", 0.0)
        scores.append(score)
    
    avg_score = np.mean(scores) if scores else 0.0
    
    # Count docs with very low scores (< 0 indicates poor relevance in some systems)
    low_relevance_count = sum(1 for s in scores if s < 0)
    
    return avg_score, low_relevance_count


def main():
    # Base path
    base_path = Path("/data/wang/junh/githubs/Debate/KARE/gpt/results")
    
    # Load all results
    print("Loading results...")
    results_A = load_results(base_path / "condition_A_gpt_4o")
    results_B = load_results(base_path / "condition_B_gpt_4o")
    results_C = load_results(base_path / "condition_C_gpt_4o")
    results_D = load_results(base_path / "condition_D_qwen")
    
    # Load Qwen baseline (single-agent RAG)
    qwen_baseline_path = Path("/data/wang/junh/githubs/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8")
    # Qwen baseline uses different JSON filename
    with open(qwen_baseline_path / "kare_debate_mortality_results.json", "r") as f:
        results_qwen_baseline = json.load(f)
    
    # Create dictionaries keyed by sample_id for easy lookup
    data_A = {r["sample_id"]: r for r in results_A["results"]}
    data_B = {r["sample_id"]: r for r in results_B["results"]}
    data_C = {r["sample_id"]: r for r in results_C["results"]}
    data_D = {r["sample_id"]: r for r in results_D["results"]}
    # Qwen baseline uses different key names: patient_id instead of sample_id, ground_truth instead of label
    data_qwen_baseline = {
        r["patient_id"]: {"sample_id": r["patient_id"], "label": r["ground_truth"], "prediction": r["prediction"]}
        for r in results_qwen_baseline["results"]
    }
    
    # Get all sample IDs
    all_sample_ids = set(data_A.keys())
    
    # R1: "Retrieval swap breaks a correct GPT pipeline"
    # A correct, but B wrong
    r1_patient_ids = []
    r1_details = []
    
    for sample_id in sorted(all_sample_ids):
        a = data_A[sample_id]
        b = data_B[sample_id]
        
        label = a["label"]
        pred_a = a["prediction"]
        pred_b = b["prediction"]
        
        # A correct, but B wrong
        if pred_a == label and pred_b != label:
            r1_patient_ids.append(sample_id)
            
            # Determine the type of error
            if label == 0 and pred_b == 1:
                error_type = "false_alarm"
            elif label == 1 and pred_b == 0:
                error_type = "missed_positive"
            else:
                error_type = "other"
            
            r1_details.append(f"{sample_id}\t(y={label}, A={pred_a}, B={pred_b}) - {error_type}")
    
    print(f"\nR1 - Retrieval swap breaks correct GPT pipeline: {len(r1_patient_ids)} cases")
    
    # R2: "Qwen baseline positive, but B collapses to survival"
    # y=1, Qwen_baseline=1 (single-agent RAG), but B=0
    r2_patient_ids = []
    r2_details = []
    
    for sample_id in sorted(all_sample_ids):
        b = data_B[sample_id]
        qb = data_qwen_baseline.get(sample_id)
        
        if not qb:  # Skip if baseline doesn't have this sample
            continue
        
        label = b["label"]
        pred_b = b["prediction"]
        pred_qb = qb["prediction"]
        
        # y=1, Qwen baseline=1, but GPT B=0
        if label == 1 and pred_qb == 1 and pred_b == 0:
            r2_patient_ids.append(sample_id)
            r2_details.append(f"{sample_id}\t(y={label}, Qwen_baseline={pred_qb}, GPT_B={pred_b})")
    
    print(f"R2 - Qwen baseline positive, but B collapses to survival: {len(r2_patient_ids)} cases")
    
    # R3: "Qwen retrieval bundle looks suspicious"
    # Analyze query quality, doc redundancy, and doc relevance from log files
    print("\nAnalyzing retrieval quality from log files...")
    r3_patient_ids = []
    r3_details = []
    
    # We'll analyze condition C (GPT integrator + Qwen analysts + Qwen retrieval)
    # and condition D (Qwen full system) logs
    for condition_name, condition_path in [("C_gpt", base_path / "condition_C_gpt_4o"),
                                            ("D_qwen", base_path / "condition_D_qwen")]:
        logs_path = condition_path / "logs"
        
        if not logs_path.exists():
            print(f"  Warning: {logs_path} not found, skipping R3 for {condition_name}")
            continue
        
        for sample_id in sorted(all_sample_ids):
            log_file = logs_path / f"{sample_id}.json"
            
            if not log_file.exists():
                continue
            
            log_data = load_log_file(log_file)
            if not log_data:
                continue
            
            # Skip if retriever wasn't called
            if not log_data.get("called_retriever", False):
                continue
            
            # Extract query and docs
            query = log_data.get("gpt_query") or log_data.get("query", "")
            docs = log_data.get("gpt_docs") or log_data.get("docs", [])
            
            if not query or not docs:
                continue
            
            # Analyze retrieval quality
            query_quality = analyze_query_quality(query)
            doc_redundancy = analyze_doc_redundancy(docs)
            avg_relevance, low_relevance_count = analyze_doc_relevance(docs)
            
            # Define thresholds for "suspicious"
            # - Low query quality (< 0.5)
            # - High redundancy (> 0.3)
            # - Low relevance (avg < 50 OR > 3 docs with negative scores)
            is_suspicious = False
            reasons = []
            
            if query_quality < 0.5:
                is_suspicious = True
                reasons.append(f"low_query_quality={query_quality:.2f}")
            
            if doc_redundancy > 0.3:
                is_suspicious = True
                reasons.append(f"high_redundancy={doc_redundancy:.2f}")
            
            if avg_relevance < 50 or low_relevance_count > 3:
                is_suspicious = True
                reasons.append(f"low_relevance(avg={avg_relevance:.1f},neg_count={low_relevance_count})")
            
            if is_suspicious:
                detail_key = f"{sample_id}_{condition_name}"
                if detail_key not in [d.split('\t')[0] for d in r3_details]:
                    r3_patient_ids.append(sample_id)
                    r3_details.append(
                        f"{sample_id}\t[{condition_name}] query='{query[:50]}...' | {', '.join(reasons)}"
                    )
    
    # Remove duplicates while preserving order
    r3_patient_ids = list(dict.fromkeys(r3_patient_ids))
    
    print(f"R3 - Suspicious retrieval bundles: {len(r3_patient_ids)} cases")
    
    # Save results
    output_path = base_path / "filtered_patient_ids"
    output_path.mkdir(exist_ok=True)
    
    # R1
    with open(output_path / "R1_retrieval_swap_breaks_pipeline.txt", "w") as f:
        f.write("# R1: Retrieval swap breaks a correct GPT pipeline\n")
        f.write("# A correct, but B wrong\n")
        f.write(f"# Total cases: {len(r1_patient_ids)}\n\n")
        for detail in r1_details:
            f.write(detail + "\n")
    
    # R1 - just patient IDs
    with open(output_path / "R1_patient_ids_only.txt", "w") as f:
        for pid in r1_patient_ids:
            f.write(pid + "\n")
    
    # R2
    with open(output_path / "R2_qwen_positive_gpt_collapse.txt", "w") as f:
        f.write("# R2: Qwen baseline positive, but B collapses to survival\n")
        f.write("# y=1, Qwen_baseline=1 (single-agent RAG), but GPT_B=0\n")
        f.write("# Qwen baseline: single-agent RAG system\n")
        f.write(f"# Total cases: {len(r2_patient_ids)}\n\n")
        for detail in r2_details:
            f.write(detail + "\n")
    
    # R2 - just patient IDs
    with open(output_path / "R2_patient_ids_only.txt", "w") as f:
        for pid in r2_patient_ids:
            f.write(pid + "\n")
    
    # R3
    with open(output_path / "R3_suspicious_retrieval.txt", "w") as f:
        f.write("# R3: Qwen retrieval bundle looks suspicious\n")
        f.write("# Metrics: low query quality, high doc redundancy, low doc relevance\n")
        f.write(f"# Total cases: {len(r3_patient_ids)}\n\n")
        for detail in r3_details:
            f.write(detail + "\n")
    
    # R3 - just patient IDs
    with open(output_path / "R3_patient_ids_only.txt", "w") as f:
        for pid in r3_patient_ids:
            f.write(pid + "\n")
    
    # Summary
    with open(output_path / "SUMMARY.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PATIENT ID FILTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"R1 - Retrieval swap breaks correct GPT pipeline: {len(r1_patient_ids)} cases\n")
        f.write("    Definition: A correct, but B wrong\n")
        f.write("    Files: R1_retrieval_swap_breaks_pipeline.txt, R1_patient_ids_only.txt\n\n")
        
        f.write(f"R2 - Qwen baseline positive, but B collapses to survival: {len(r2_patient_ids)} cases\n")
        f.write("    Definition: y=1, Qwen_baseline=1 (single-agent RAG), but GPT_B=0\n")
        f.write("    Files: R2_qwen_positive_gpt_collapse.txt, R2_patient_ids_only.txt\n\n")
        
        f.write(f"R3 - Suspicious retrieval bundles: {len(r3_patient_ids)} cases\n")
        f.write("    Definition: Low query quality OR high doc redundancy OR low doc relevance\n")
        f.write("    Thresholds: query_quality<0.5, redundancy>0.3, avg_relevance<50 or >3 neg docs\n")
        f.write("    Files: R3_suspicious_retrieval.txt, R3_patient_ids_only.txt\n\n")
        
        # Breakdown by error type for R1
        false_alarms = [d for d in r1_details if "false_alarm" in d]
        missed_positives = [d for d in r1_details if "missed_positive" in d]
        
        f.write("\nR1 Breakdown:\n")
        f.write(f"  - False alarms (y=0, A=0 correct, B=1 wrong): {len(false_alarms)}\n")
        f.write(f"  - Missed positives (y=1, A=1 correct, B=0 wrong): {len(missed_positives)}\n")
        
        # Overlap analysis
        overlap_r1_r2 = set(r1_patient_ids) & set(r2_patient_ids)
        overlap_r1_r3 = set(r1_patient_ids) & set(r3_patient_ids)
        overlap_r2_r3 = set(r2_patient_ids) & set(r3_patient_ids)
        overlap_all = set(r1_patient_ids) & set(r2_patient_ids) & set(r3_patient_ids)
        
        f.write(f"\nOverlap Analysis:\n")
        f.write(f"  R1 ∩ R2: {len(overlap_r1_r2)} cases")
        if overlap_r1_r2:
            f.write(f" - {', '.join(sorted(overlap_r1_r2))}")
        f.write("\n")
        
        f.write(f"  R1 ∩ R3: {len(overlap_r1_r3)} cases")
        if overlap_r1_r3:
            f.write(f" - {', '.join(sorted(overlap_r1_r3))}")
        f.write("\n")
        
        f.write(f"  R2 ∩ R3: {len(overlap_r2_r3)} cases")
        if overlap_r2_r3:
            f.write(f" - {', '.join(sorted(overlap_r2_r3))}")
        f.write("\n")
        
        f.write(f"  R1 ∩ R2 ∩ R3: {len(overlap_all)} cases")
        if overlap_all:
            f.write(f" - {', '.join(sorted(overlap_all))}")
        f.write("\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nFiles created:")
    print(f"  - R1_retrieval_swap_breaks_pipeline.txt")
    print(f"  - R1_patient_ids_only.txt")
    print(f"  - R2_qwen_positive_gpt_collapse.txt")
    print(f"  - R2_patient_ids_only.txt")
    print(f"  - R3_suspicious_retrieval.txt")
    print(f"  - R3_patient_ids_only.txt")
    print(f"  - SUMMARY.txt")
    
    # Print some examples
    print("\n" + "=" * 80)
    print("EXAMPLES FROM R1:")
    print("=" * 80)
    for detail in r1_details[:5]:
        print(detail)
    if len(r1_details) > 5:
        print(f"... and {len(r1_details) - 5} more")
    
    print("\n" + "=" * 80)
    print("EXAMPLES FROM R2:")
    print("=" * 80)
    for detail in r2_details[:5]:
        print(detail)
    if len(r2_details) > 5:
        print(f"... and {len(r2_details) - 5} more")
    
    print("\n" + "=" * 80)
    print("EXAMPLES FROM R3:")
    print("=" * 80)
    for detail in r3_details[:5]:
        print(detail)
    if len(r3_details) > 5:
        print(f"... and {len(r3_details) - 5} more")


if __name__ == "__main__":
    main()
