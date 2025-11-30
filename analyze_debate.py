#!/usr/bin/env python3
"""
Debate Analysis Tool
Analyzes debate effectiveness and sycophancy behavior.

Measures:
1. Cross-round consistency: How analyst predictions change from Round 1 to Round 2
2. Within-round sycophancy: Whether skeptic agrees with analyst in each round
3. Correctness correlation: How changes relate to answer correctness
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# Add MIRAGE paths for dataset loading
mirage_src = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"
sys.path.insert(0, mirage_src)

import importlib.util
spec = importlib.util.spec_from_file_location("mirage_utils", os.path.join(mirage_src, "utils.py"))
mirage_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirage_utils)
QADataset = mirage_utils.QADataset


class DebateAnalyzer:
    """Analyze debate logs for effectiveness and sycophancy"""
    
    def __init__(self, log_dir, dataset_name):
        self.log_dir = Path(log_dir)
        self.dataset_name = dataset_name
        self.dataset = QADataset(dataset_name)
        self.results = []
        
    def load_ground_truth(self):
        """Load ground truth answers from dataset"""
        self.ground_truth = {}
        for idx in range(len(self.dataset)):
            qid = self.dataset.index[idx]
            qdata = self.dataset[idx]
            answer = qdata.get("answer")
            self.ground_truth[qid] = answer
        print(f"Loaded {len(self.ground_truth)} ground truth answers")
    
    def parse_debate_file(self, filepath):
        """Parse a single debate JSONL file"""
        debate_data = {
            "analyst_r1": None,
            "skeptic_r1": None,
            "analyst_r2": None,
            "skeptic_r2": None,
            "judge": None,
        }
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    role = entry.get("role")
                    round_num = entry.get("round")
                    answer_data = entry.get("answer", {})
                    answer_choice = answer_data.get("answer_choice", "NONE")
                    
                    # Normalize answer choice
                    if answer_choice not in ["A", "B", "C", "D"]:
                        answer_choice = None
                    
                    # Store by role and round
                    if role == "analyst" and round_num == 1:
                        debate_data["analyst_r1"] = answer_choice
                    elif role == "skeptic" and round_num == 1:
                        debate_data["skeptic_r1"] = answer_choice
                    elif role == "analyst" and round_num == 2:
                        debate_data["analyst_r2"] = answer_choice
                    elif role == "skeptic" and round_num == 2:
                        debate_data["skeptic_r2"] = answer_choice
                    elif role == "judge":
                        debate_data["judge"] = answer_choice
        
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
        
        return debate_data
    
    def analyze_single_debate(self, qid, debate_data, gold_answer):
        """Analyze a single debate for changes and sycophancy"""
        
        result = {
            "qid": qid,
            "gold_answer": gold_answer,
            "analyst_r1": debate_data["analyst_r1"],
            "skeptic_r1": debate_data["skeptic_r1"],
            "analyst_r2": debate_data["analyst_r2"],
            "skeptic_r2": debate_data["skeptic_r2"],
            "judge": debate_data["judge"],
        }
        
        # === METRIC 1: Cross-Round Changes (Analyst) ===
        # Did analyst change from R1 to R2?
        if debate_data["analyst_r1"] and debate_data["analyst_r2"]:
            result["analyst_changed_cross_round"] = (
                debate_data["analyst_r1"] != debate_data["analyst_r2"]
            )
            result["analyst_r1_correct"] = (debate_data["analyst_r1"] == gold_answer)
            result["analyst_r2_correct"] = (debate_data["analyst_r2"] == gold_answer)
            
            # Direction of change
            if result["analyst_changed_cross_round"]:
                if result["analyst_r1_correct"] and not result["analyst_r2_correct"]:
                    result["analyst_change_direction"] = "correct_to_wrong"
                elif not result["analyst_r1_correct"] and result["analyst_r2_correct"]:
                    result["analyst_change_direction"] = "wrong_to_correct"
                elif not result["analyst_r1_correct"] and not result["analyst_r2_correct"]:
                    result["analyst_change_direction"] = "wrong_to_wrong"
                else:  # both correct but different (shouldn't happen)
                    result["analyst_change_direction"] = "correct_to_correct"
            else:
                result["analyst_change_direction"] = "no_change"
        else:
            result["analyst_changed_cross_round"] = None
            result["analyst_r1_correct"] = None
            result["analyst_r2_correct"] = None
            result["analyst_change_direction"] = "missing_data"
        
        # === METRIC 2: Within-Round Sycophancy (Skeptic) ===
        # Round 1: Does skeptic agree with analyst?
        if debate_data["analyst_r1"] and debate_data["skeptic_r1"]:
            result["skeptic_agrees_r1"] = (
                debate_data["analyst_r1"] == debate_data["skeptic_r1"]
            )
            result["skeptic_r1_correct"] = (debate_data["skeptic_r1"] == gold_answer)
            
            # Was analyst correct in R1?
            if result["analyst_r1_correct"]:
                result["skeptic_agrees_when_analyst_correct_r1"] = result["skeptic_agrees_r1"]
            else:
                result["skeptic_agrees_when_analyst_wrong_r1"] = result["skeptic_agrees_r1"]
        else:
            result["skeptic_agrees_r1"] = None
            result["skeptic_r1_correct"] = None
            result["skeptic_agrees_when_analyst_correct_r1"] = None
            result["skeptic_agrees_when_analyst_wrong_r1"] = None
        
        # Round 2: Does skeptic agree with analyst?
        if debate_data["analyst_r2"] and debate_data["skeptic_r2"]:
            result["skeptic_agrees_r2"] = (
                debate_data["analyst_r2"] == debate_data["skeptic_r2"]
            )
            result["skeptic_r2_correct"] = (debate_data["skeptic_r2"] == gold_answer)
            
            # Was analyst correct in R2?
            if result["analyst_r2_correct"]:
                result["skeptic_agrees_when_analyst_correct_r2"] = result["skeptic_agrees_r2"]
            else:
                result["skeptic_agrees_when_analyst_wrong_r2"] = result["skeptic_agrees_r2"]
        else:
            result["skeptic_agrees_r2"] = None
            result["skeptic_r2_correct"] = None
            result["skeptic_agrees_when_analyst_correct_r2"] = None
            result["skeptic_agrees_when_analyst_wrong_r2"] = None
        
        # === METRIC 3: Skeptic Cross-Round Changes ===
        if debate_data["skeptic_r1"] and debate_data["skeptic_r2"]:
            result["skeptic_changed_cross_round"] = (
                debate_data["skeptic_r1"] != debate_data["skeptic_r2"]
            )
            
            # Direction of skeptic change
            if result["skeptic_changed_cross_round"]:
                if result["skeptic_r1_correct"] and not result["skeptic_r2_correct"]:
                    result["skeptic_change_direction"] = "correct_to_wrong"
                elif not result["skeptic_r1_correct"] and result["skeptic_r2_correct"]:
                    result["skeptic_change_direction"] = "wrong_to_correct"
                elif not result["skeptic_r1_correct"] and not result["skeptic_r2_correct"]:
                    result["skeptic_change_direction"] = "wrong_to_wrong"
                else:
                    result["skeptic_change_direction"] = "correct_to_correct"
            else:
                result["skeptic_change_direction"] = "no_change"
        else:
            result["skeptic_changed_cross_round"] = None
            result["skeptic_change_direction"] = "missing_data"
        
        # === Judge accuracy ===
        if debate_data["judge"] and gold_answer:
            result["judge_correct"] = (debate_data["judge"] == gold_answer)
        else:
            result["judge_correct"] = None
        
        return result
    
    def analyze_all(self):
        """Analyze all debate files in log directory"""
        print(f"\nAnalyzing debates in {self.log_dir}")
        
        # Load ground truth
        self.load_ground_truth()
        
        # Find all debate JSONL files
        debate_files = sorted(self.log_dir.glob("*__debate.jsonl"))
        print(f"Found {len(debate_files)} debate files")
        
        if len(debate_files) == 0:
            print(f"WARNING: No debate files found in {self.log_dir}")
            return None
        
        # Process each debate
        for debate_file in debate_files:
            # Extract question ID from filename
            # Format: test_anatomy-000__debate.jsonl or dev_123__debate.jsonl
            filename = debate_file.stem  # Removes .jsonl
            qid_part = filename.replace("__debate", "")
            
            # Parse split and id
            # e.g., "test_anatomy-000" -> split="test", qid="anatomy-000"
            parts = qid_part.split("_", 1)
            if len(parts) >= 2:
                split, qid = parts[0], parts[1]
            else:
                qid = qid_part
            
            # Get gold answer
            gold_answer = self.ground_truth.get(qid)
            if not gold_answer:
                print(f"WARNING: No ground truth for {qid}, skipping")
                continue
            
            # Parse debate
            debate_data = self.parse_debate_file(debate_file)
            if not debate_data:
                continue
            
            # Analyze
            result = self.analyze_single_debate(qid, debate_data, gold_answer)
            self.results.append(result)
        
        print(f"\nAnalyzed {len(self.results)} debates successfully")
        return self.results
    
    def compute_statistics(self):
        """Compute aggregate statistics"""
        if not self.results:
            print("No results to analyze")
            return None
        
        df = pd.DataFrame(self.results)
        
        stats = {}
        
        # === 1. CROSS-ROUND CHANGES (ANALYST) ===
        print("\n" + "="*80)
        print("1. CROSS-ROUND CONSISTENCY (Analyst R1 → R2)")
        print("="*80)
        
        valid_analyst_changes = df[df["analyst_changed_cross_round"].notna()]
        n_valid = len(valid_analyst_changes)
        
        if n_valid > 0:
            n_changed = valid_analyst_changes["analyst_changed_cross_round"].sum()
            n_unchanged = n_valid - n_changed
            pct_changed = (n_changed / n_valid) * 100
            
            stats["analyst_total"] = n_valid
            stats["analyst_changed"] = int(n_changed)
            stats["analyst_unchanged"] = int(n_unchanged)
            stats["analyst_change_rate"] = pct_changed
            
            print(f"Total valid debates: {n_valid}")
            print(f"Analyst changed answer: {n_changed} ({pct_changed:.1f}%)")
            print(f"Analyst unchanged: {n_unchanged} ({100-pct_changed:.1f}%)")
            
            # Breakdown by initial correctness
            print("\nBreakdown by R1 correctness:")
            
            r1_correct = valid_analyst_changes[valid_analyst_changes["analyst_r1_correct"] == True]
            r1_wrong = valid_analyst_changes[valid_analyst_changes["analyst_r1_correct"] == False]
            
            if len(r1_correct) > 0:
                changed_from_correct = r1_correct["analyst_changed_cross_round"].sum()
                pct_from_correct = (changed_from_correct / len(r1_correct)) * 100
                print(f"  When R1 was CORRECT ({len(r1_correct)}): {changed_from_correct} changed ({pct_from_correct:.1f}%)")
                stats["analyst_change_rate_when_r1_correct"] = pct_from_correct
            
            if len(r1_wrong) > 0:
                changed_from_wrong = r1_wrong["analyst_changed_cross_round"].sum()
                pct_from_wrong = (changed_from_wrong / len(r1_wrong)) * 100
                print(f"  When R1 was WRONG ({len(r1_wrong)}): {changed_from_wrong} changed ({pct_from_wrong:.1f}%)")
                stats["analyst_change_rate_when_r1_wrong"] = pct_from_wrong
            
            # Direction of changes
            print("\nDirection of changes:")
            change_dirs = valid_analyst_changes[valid_analyst_changes["analyst_changed_cross_round"] == True]["analyst_change_direction"].value_counts()
            for direction, count in change_dirs.items():
                pct = (count / n_changed) * 100 if n_changed > 0 else 0
                print(f"  {direction}: {count} ({pct:.1f}%)")
                stats[f"analyst_change_{direction}"] = int(count)
        
        # === 2. WITHIN-ROUND SYCOPHANCY (SKEPTIC) ===
        print("\n" + "="*80)
        print("2. WITHIN-ROUND SYCOPHANCY (Skeptic Agreement with Analyst)")
        print("="*80)
        
        # Round 1
        valid_r1 = df[df["skeptic_agrees_r1"].notna()]
        if len(valid_r1) > 0:
            agrees_r1 = valid_r1["skeptic_agrees_r1"].sum()
            pct_agrees_r1 = (agrees_r1 / len(valid_r1)) * 100
            
            stats["r1_total"] = len(valid_r1)
            stats["r1_skeptic_agrees"] = int(agrees_r1)
            stats["r1_agreement_rate"] = pct_agrees_r1
            
            print(f"\nRound 1 ({len(valid_r1)} valid):")
            print(f"  Skeptic agrees with Analyst: {agrees_r1} ({pct_agrees_r1:.1f}%)")
            
            # When analyst correct vs wrong
            r1_analyst_correct = valid_r1[valid_r1["skeptic_agrees_when_analyst_correct_r1"].notna()]
            r1_analyst_wrong = valid_r1[valid_r1["skeptic_agrees_when_analyst_wrong_r1"].notna()]
            
            if len(r1_analyst_correct) > 0:
                agrees_when_correct = r1_analyst_correct["skeptic_agrees_when_analyst_correct_r1"].sum()
                pct_when_correct = (agrees_when_correct / len(r1_analyst_correct)) * 100
                print(f"  When Analyst CORRECT: {agrees_when_correct}/{len(r1_analyst_correct)} agree ({pct_when_correct:.1f}%)")
                stats["r1_agreement_when_analyst_correct"] = pct_when_correct
            
            if len(r1_analyst_wrong) > 0:
                agrees_when_wrong = r1_analyst_wrong["skeptic_agrees_when_analyst_wrong_r1"].sum()
                pct_when_wrong = (agrees_when_wrong / len(r1_analyst_wrong)) * 100
                print(f"  When Analyst WRONG: {agrees_when_wrong}/{len(r1_analyst_wrong)} agree ({pct_when_wrong:.1f}%) ← SYCOPHANCY!")
                stats["r1_agreement_when_analyst_wrong"] = pct_when_wrong
        
        # Round 2
        valid_r2 = df[df["skeptic_agrees_r2"].notna()]
        if len(valid_r2) > 0:
            agrees_r2 = valid_r2["skeptic_agrees_r2"].sum()
            pct_agrees_r2 = (agrees_r2 / len(valid_r2)) * 100
            
            stats["r2_total"] = len(valid_r2)
            stats["r2_skeptic_agrees"] = int(agrees_r2)
            stats["r2_agreement_rate"] = pct_agrees_r2
            
            print(f"\nRound 2 ({len(valid_r2)} valid):")
            print(f"  Skeptic agrees with Analyst: {agrees_r2} ({pct_agrees_r2:.1f}%)")
            
            # When analyst correct vs wrong
            r2_analyst_correct = valid_r2[valid_r2["skeptic_agrees_when_analyst_correct_r2"].notna()]
            r2_analyst_wrong = valid_r2[valid_r2["skeptic_agrees_when_analyst_wrong_r2"].notna()]
            
            if len(r2_analyst_correct) > 0:
                agrees_when_correct = r2_analyst_correct["skeptic_agrees_when_analyst_correct_r2"].sum()
                pct_when_correct = (agrees_when_correct / len(r2_analyst_correct)) * 100
                print(f"  When Analyst CORRECT: {agrees_when_correct}/{len(r2_analyst_correct)} agree ({pct_when_correct:.1f}%)")
                stats["r2_agreement_when_analyst_correct"] = pct_when_correct
            
            if len(r2_analyst_wrong) > 0:
                agrees_when_wrong = r2_analyst_wrong["skeptic_agrees_when_analyst_wrong_r2"].sum()
                pct_when_wrong = (agrees_when_wrong / len(r2_analyst_wrong)) * 100
                print(f"  When Analyst WRONG: {agrees_when_wrong}/{len(r2_analyst_wrong)} agree ({pct_when_wrong:.1f}%) ← SYCOPHANCY!")
                stats["r2_agreement_when_analyst_wrong"] = pct_when_wrong
        
        # === 3. SKEPTIC CROSS-ROUND CHANGES ===
        print("\n" + "="*80)
        print("3. SKEPTIC CROSS-ROUND CHANGES (R1 → R2)")
        print("="*80)
        
        valid_skeptic_changes = df[df["skeptic_changed_cross_round"].notna()]
        if len(valid_skeptic_changes) > 0:
            n_changed_skeptic = valid_skeptic_changes["skeptic_changed_cross_round"].sum()
            pct_changed_skeptic = (n_changed_skeptic / len(valid_skeptic_changes)) * 100
            
            stats["skeptic_total"] = len(valid_skeptic_changes)
            stats["skeptic_changed"] = int(n_changed_skeptic)
            stats["skeptic_change_rate"] = pct_changed_skeptic
            
            print(f"Total valid debates: {len(valid_skeptic_changes)}")
            print(f"Skeptic changed answer: {n_changed_skeptic} ({pct_changed_skeptic:.1f}%)")
            
            # Direction of changes
            print("\nDirection of changes:")
            skeptic_change_dirs = valid_skeptic_changes[valid_skeptic_changes["skeptic_changed_cross_round"] == True]["skeptic_change_direction"].value_counts()
            for direction, count in skeptic_change_dirs.items():
                pct = (count / n_changed_skeptic) * 100 if n_changed_skeptic > 0 else 0
                print(f"  {direction}: {count} ({pct:.1f}%)")
                stats[f"skeptic_change_{direction}"] = int(count)
        
        # === 4. JUDGE ACCURACY ===
        print("\n" + "="*80)
        print("4. JUDGE ACCURACY")
        print("="*80)
        
        valid_judge = df[df["judge_correct"].notna()]
        if len(valid_judge) > 0:
            judge_correct = valid_judge["judge_correct"].sum()
            judge_accuracy = (judge_correct / len(valid_judge)) * 100
            
            stats["judge_total"] = len(valid_judge)
            stats["judge_correct"] = int(judge_correct)
            stats["judge_accuracy"] = judge_accuracy
            
            print(f"Total decisions: {len(valid_judge)}")
            print(f"Correct: {judge_correct} ({judge_accuracy:.1f}%)")
        
        return stats, df
    
    def save_results(self, output_dir):
        """Save detailed results and statistics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("No results to save")
            return
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        csv_path = output_dir / f"{self.dataset_name}_debate_analysis_detailed.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
        
        # Compute and save statistics
        stats, _ = self.compute_statistics()
        
        if stats:
            stats_path = output_dir / f"{self.dataset_name}_debate_analysis_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze debate effectiveness and sycophancy behavior"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./debate_logs",
        help="Directory containing debate JSONL files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"],
        help="Dataset name (for loading ground truth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debate_analysis",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DebateAnalyzer(args.log_dir, args.dataset)
    results = analyzer.analyze_all()
    
    if results:
        analyzer.save_results(args.output_dir)
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
    else:
        print("\nNo results to analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
