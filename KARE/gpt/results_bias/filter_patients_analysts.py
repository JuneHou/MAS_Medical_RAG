#!/usr/bin/env python3
"""
Filter patient IDs based on analyst-summary problems.

A1: "Analyst swap breaks a correct GPT pipeline"
    - A correct, but C wrong
    - A uses GPT analysts, C uses Qwen analysts (retrieval & integrator both GPT)

A2: "Qwen baseline positive, but C collapses to survival"
    - y=1, D=1 (Qwen baseline), but C=0
    - Tests if Qwen analysts cause issues even with GPT retrieval + integrator

A3: "Qwen analysts vs GPT analysts disagreement cases"
    - A ≠ C (predictions differ regardless of correctness)
"""

import json
from pathlib import Path


def load_results(condition_path):
    """Load results.json from a condition folder."""
    with open(Path(condition_path) / "results.json", "r") as f:
        return json.load(f)


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
    
    # A1: "Analyst swap breaks a correct GPT pipeline"
    # A correct, but C wrong
    # A: GPT analysts + GPT retrieval + GPT integrator
    # C: Qwen analysts + GPT retrieval + GPT integrator
    a1_patient_ids = []
    a1_details = []
    
    for sample_id in sorted(all_sample_ids):
        a = data_A[sample_id]
        c = data_C[sample_id]
        
        label = a["label"]
        pred_a = a["prediction"]
        pred_c = c["prediction"]
        
        # A correct, but C wrong
        if pred_a == label and pred_c != label:
            a1_patient_ids.append(sample_id)
            
            # Determine the type of error
            if label == 0 and pred_c == 1:
                error_type = "false_alarm"
            elif label == 1 and pred_c == 0:
                error_type = "missed_positive"
            else:
                error_type = "other"
            
            a1_details.append(f"{sample_id}\t(y={label}, A={pred_a}, C={pred_c}) - {error_type}")
    
    print(f"\nA1 - Analyst swap breaks correct GPT pipeline: {len(a1_patient_ids)} cases")
    
    # A2: "Qwen baseline positive, but C collapses to survival"
    # y=1, Qwen_baseline=1 (single-agent RAG), but C=0
    a2_patient_ids = []
    a2_details = []
    
    for sample_id in sorted(all_sample_ids):
        c = data_C[sample_id]
        qb = data_qwen_baseline.get(sample_id)
        
        if not qb:  # Skip if baseline doesn't have this sample
            continue
        
        label = c["label"]
        pred_c = c["prediction"]
        pred_qb = qb["prediction"]
        
        # y=1, Qwen baseline=1, but C=0
        if label == 1 and pred_qb == 1 and pred_c == 0:
            a2_patient_ids.append(sample_id)
            a2_details.append(f"{sample_id}\t(y={label}, Qwen_baseline={pred_qb}, C={pred_c})")
    
    print(f"A2 - Qwen baseline positive, but C collapses to survival: {len(a2_patient_ids)} cases")
    
    # A3: "Qwen analysts vs GPT analysts disagreement cases"
    # A ≠ C (predictions differ regardless of correctness)
    a3_patient_ids = []
    a3_details = []
    
    for sample_id in sorted(all_sample_ids):
        a = data_A[sample_id]
        c = data_C[sample_id]
        
        label = a["label"]
        pred_a = a["prediction"]
        pred_c = c["prediction"]
        
        # A ≠ C
        if pred_a != pred_c:
            a3_patient_ids.append(sample_id)
            
            # Add correctness info
            a_correct = "✓" if pred_a == label else "✗"
            c_correct = "✓" if pred_c == label else "✗"
            
            a3_details.append(
                f"{sample_id}\t(y={label}, A={pred_a}{a_correct}, C={pred_c}{c_correct})"
            )
    
    print(f"A3 - GPT vs Qwen analysts disagreement: {len(a3_patient_ids)} cases")
    
    # Save results
    output_path = base_path / "filtered_patient_ids"
    output_path.mkdir(exist_ok=True)
    
    # A1
    with open(output_path / "A1_analyst_swap_breaks_pipeline.txt", "w") as f:
        f.write("# A1: Analyst swap breaks a correct GPT pipeline\n")
        f.write("# A correct, but C wrong\n")
        f.write("# A: GPT analysts + GPT retrieval + GPT integrator\n")
        f.write("# C: Qwen analysts + GPT retrieval + GPT integrator\n")
        f.write(f"# Total cases: {len(a1_patient_ids)}\n\n")
        for detail in a1_details:
            f.write(detail + "\n")
    
    # A1 - just patient IDs
    with open(output_path / "A1_patient_ids_only.txt", "w") as f:
        for pid in a1_patient_ids:
            f.write(pid + "\n")
    
    # A2
    with open(output_path / "A2_qwen_baseline_positive_C_collapse.txt", "w") as f:
        f.write("# A2: Qwen baseline positive, but C collapses to survival\n")
        f.write("# y=1, Qwen_baseline=1 (single-agent RAG), but C=0\n")
        f.write("# Tests if Qwen analysts cause issues with GPT retrieval + integrator\n")
        f.write(f"# Total cases: {len(a2_patient_ids)}\n\n")
        for detail in a2_details:
            f.write(detail + "\n")
    
    # A2 - just patient IDs
    with open(output_path / "A2_patient_ids_only.txt", "w") as f:
        for pid in a2_patient_ids:
            f.write(pid + "\n")
    
    # A3
    with open(output_path / "A3_analyst_disagreement.txt", "w") as f:
        f.write("# A3: Qwen analysts vs GPT analysts disagreement cases\n")
        f.write("# A ≠ C (predictions differ)\n")
        f.write("# ✓ = correct prediction, ✗ = incorrect prediction\n")
        f.write(f"# Total cases: {len(a3_patient_ids)}\n\n")
        for detail in a3_details:
            f.write(detail + "\n")
    
    # A3 - just patient IDs
    with open(output_path / "A3_patient_ids_only.txt", "w") as f:
        for pid in a3_patient_ids:
            f.write(pid + "\n")
    
    # Summary
    with open(output_path / "ANALYST_SUMMARY.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ANALYST-FOCUSED PATIENT ID FILTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Condition Recap:\n")
        f.write("  A: GPT analysts + GPT retrieval + GPT integrator\n")
        f.write("  C: Qwen analysts + GPT retrieval + GPT integrator\n")
        f.write("  D: Qwen analysts + Qwen retrieval + Qwen integrator (multi-agent)\n")
        f.write("  Qwen baseline: Single-agent RAG system (not D)\n\n")
        
        f.write(f"A1 - Analyst swap breaks correct GPT pipeline: {len(a1_patient_ids)} cases\n")
        f.write("    Definition: A correct, but C wrong\n")
        f.write("    Files: A1_analyst_swap_breaks_pipeline.txt, A1_patient_ids_only.txt\n\n")
        
        f.write(f"A2 - Qwen baseline positive, but C collapses to survival: {len(a2_patient_ids)} cases\n")
        f.write("    Definition: y=1, Qwen_baseline=1 (single-agent RAG), but C=0\n")
        f.write("    Files: A2_qwen_baseline_positive_C_collapse.txt, A2_patient_ids_only.txt\n\n")
        
        f.write(f"A3 - GPT vs Qwen analysts disagreement: {len(a3_patient_ids)} cases\n")
        f.write("    Definition: A ≠ C (any disagreement)\n")
        f.write("    Files: A3_analyst_disagreement.txt, A3_patient_ids_only.txt\n\n")
        
        # Breakdown by error type for A1
        false_alarms = [d for d in a1_details if "false_alarm" in d]
        missed_positives = [d for d in a1_details if "missed_positive" in d]
        
        f.write("\nA1 Breakdown:\n")
        f.write(f"  - False alarms (y=0, A=0✓, C=1✗): {len(false_alarms)}\n")
        f.write(f"  - Missed positives (y=1, A=1✓, C=0✗): {len(missed_positives)}\n")
        
        # A3 correctness breakdown
        both_wrong = []
        a_correct_c_wrong = []
        c_correct_a_wrong = []
        
        for sample_id in a3_patient_ids:
            a = data_A[sample_id]
            c = data_C[sample_id]
            label = a["label"]
            pred_a = a["prediction"]
            pred_c = c["prediction"]
            
            a_correct = pred_a == label
            c_correct = pred_c == label
            
            if not a_correct and not c_correct:
                both_wrong.append(sample_id)
            elif a_correct and not c_correct:
                a_correct_c_wrong.append(sample_id)
            elif not a_correct and c_correct:
                c_correct_a_wrong.append(sample_id)
        
        f.write("\nA3 Breakdown:\n")
        f.write(f"  - A correct, C wrong: {len(a_correct_c_wrong)} (same as A1)\n")
        f.write(f"  - C correct, A wrong: {len(c_correct_a_wrong)} (Qwen analysts improve)\n")
        f.write(f"  - Both wrong (different errors): {len(both_wrong)}\n")
        
        # Overlap analysis
        overlap_a1_a2 = set(a1_patient_ids) & set(a2_patient_ids)
        overlap_a1_a3 = set(a1_patient_ids) & set(a3_patient_ids)
        overlap_a2_a3 = set(a2_patient_ids) & set(a3_patient_ids)
        overlap_all = set(a1_patient_ids) & set(a2_patient_ids) & set(a3_patient_ids)
        
        f.write(f"\nOverlap Analysis:\n")
        f.write(f"  A1 ∩ A2: {len(overlap_a1_a2)} cases")
        if overlap_a1_a2:
            f.write(f" - {', '.join(sorted(overlap_a1_a2))}")
        f.write("\n")
        
        f.write(f"  A1 ∩ A3: {len(overlap_a1_a3)} cases")
        if overlap_a1_a3:
            f.write(f" - {', '.join(sorted(overlap_a1_a3))}")
        f.write("\n")
        
        f.write(f"  A2 ∩ A3: {len(overlap_a2_a3)} cases")
        if overlap_a2_a3:
            f.write(f" - {', '.join(sorted(overlap_a2_a3))}")
        f.write("\n")
        
        f.write(f"  A1 ∩ A2 ∩ A3: {len(overlap_all)} cases")
        if overlap_all:
            f.write(f" - {', '.join(sorted(overlap_all))}")
        f.write("\n")
        
        # Note on A1 vs A3
        f.write("\nNote: A1 cases are a subset of A3 (A1 requires A correct, A3 just requires A≠C)\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nFiles created:")
    print(f"  - A1_analyst_swap_breaks_pipeline.txt")
    print(f"  - A1_patient_ids_only.txt")
    print(f"  - A2_qwen_baseline_positive_C_collapse.txt")
    print(f"  - A2_patient_ids_only.txt")
    print(f"  - A3_analyst_disagreement.txt")
    print(f"  - A3_patient_ids_only.txt")
    print(f"  - ANALYST_SUMMARY.txt")
    
    # Print some examples
    print("\n" + "=" * 80)
    print("EXAMPLES FROM A1:")
    print("=" * 80)
    for detail in a1_details[:5]:
        print(detail)
    if len(a1_details) > 5:
        print(f"... and {len(a1_details) - 5} more")
    
    print("\n" + "=" * 80)
    print("EXAMPLES FROM A2:")
    print("=" * 80)
    for detail in a2_details[:5]:
        print(detail)
    if len(a2_details) > 5:
        print(f"... and {len(a2_details) - 5} more")
    
    print("\n" + "=" * 80)
    print("EXAMPLES FROM A3:")
    print("=" * 80)
    for detail in a3_details[:10]:
        print(detail)
    if len(a3_details) > 10:
        print(f"... and {len(a3_details) - 10} more")
    
    # Show A3 breakdown by correctness
    print("\n" + "=" * 80)
    print("A3 CORRECTNESS BREAKDOWN:")
    print("=" * 80)
    print(f"Cases where Qwen analysts improved (C correct, A wrong): {len(c_correct_a_wrong)}")
    if c_correct_a_wrong:
        print("  Examples:", ", ".join(c_correct_a_wrong[:5]))
        if len(c_correct_a_wrong) > 5:
            print(f"  ... and {len(c_correct_a_wrong) - 5} more")


if __name__ == "__main__":
    main()
