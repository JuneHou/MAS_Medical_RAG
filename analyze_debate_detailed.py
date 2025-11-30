#!/usr/bin/env python3
"""
Additional analysis: breakdown by specific patterns
"""

import pandas as pd
import json
from pathlib import Path

# Load the detailed results
df = pd.read_csv("debate_analysis/mmlu_debate_analysis_detailed.csv")

print("="*80)
print("ADDITIONAL ANALYSIS: DETAILED BREAKDOWN")
print("="*80)

# === 1. Analyst Confidence Analysis ===
print("\n1. ANALYST CONSISTENCY PATTERN")
print("-"*80)

# Group by whether analyst changed
changed = df[df["analyst_changed_cross_round"] == True]
unchanged = df[df["analyst_changed_cross_round"] == False]

print(f"Unchanged Analyst R1→R2: {len(unchanged)}")
print(f"  R1 Correct → R2 Correct: {((unchanged['analyst_r1_correct'] == True) & (unchanged['analyst_r2_correct'] == True)).sum()}")
print(f"  R1 Wrong → R2 Wrong: {((unchanged['analyst_r1_correct'] == False) & (unchanged['analyst_r2_correct'] == False)).sum()}")

print(f"\nChanged Analyst R1→R2: {len(changed)}")
for direction in ["correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"]:
    count = (changed["analyst_change_direction"] == direction).sum()
    print(f"  {direction}: {count}")

# === 2. Skeptic Behavior Analysis ===
print("\n\n2. SKEPTIC AGREEMENT PATTERNS")
print("-"*80)

# When does skeptic agree vs disagree?
r1_valid = df[df["skeptic_agrees_r1"].notna()]

print("Round 1:")
print(f"  Total valid: {len(r1_valid)}")

# Break down by outcome
r1_both_correct = r1_valid[(r1_valid["analyst_r1_correct"] == True) & (r1_valid["skeptic_r1_correct"] == True)]
r1_both_wrong = r1_valid[(r1_valid["analyst_r1_correct"] == False) & (r1_valid["skeptic_r1_correct"] == False)]
r1_analyst_correct_skeptic_wrong = r1_valid[(r1_valid["analyst_r1_correct"] == True) & (r1_valid["skeptic_r1_correct"] == False)]
r1_analyst_wrong_skeptic_correct = r1_valid[(r1_valid["analyst_r1_correct"] == False) & (r1_valid["skeptic_r1_correct"] == True)]

print(f"  Both Correct: {len(r1_both_correct)} ({len(r1_both_correct)/len(r1_valid)*100:.1f}%)")
print(f"    Agree: {r1_both_correct['skeptic_agrees_r1'].sum()}")
print(f"  Both Wrong: {len(r1_both_wrong)} ({len(r1_both_wrong)/len(r1_valid)*100:.1f}%)")
print(f"    Agree: {r1_both_wrong['skeptic_agrees_r1'].sum()}")
print(f"  Analyst Correct, Skeptic Wrong: {len(r1_analyst_correct_skeptic_wrong)} ({len(r1_analyst_correct_skeptic_wrong)/len(r1_valid)*100:.1f}%)")
print(f"    Agree: {r1_analyst_correct_skeptic_wrong['skeptic_agrees_r1'].sum()}")
print(f"  Analyst Wrong, Skeptic Correct: {len(r1_analyst_wrong_skeptic_correct)} ({len(r1_analyst_wrong_skeptic_correct)/len(r1_valid)*100:.1f}%)") 
print(f"    Agree: {r1_analyst_wrong_skeptic_correct['skeptic_agrees_r1'].sum()}")

# === 3. Judge Decision Analysis ===
print("\n\n3. JUDGE DECISION PATTERNS")
print("-"*80)

# When both agents agree, does judge follow them?
both_agree_r2 = df[(df["analyst_r2"].notna()) & (df["skeptic_r2"].notna()) & (df["analyst_r2"] == df["skeptic_r2"])]
print(f"Both agents agree in R2: {len(both_agree_r2)} debates")

if len(both_agree_r2) > 0:
    judge_follows_consensus = (both_agree_r2["judge"] == both_agree_r2["analyst_r2"]).sum()
    print(f"  Judge follows consensus: {judge_follows_consensus} ({judge_follows_consensus/len(both_agree_r2)*100:.1f}%)")
    
    consensus_correct = ((both_agree_r2["analyst_r2"] == both_agree_r2["gold_answer"])).sum()
    print(f"  Consensus was correct: {consensus_correct} ({consensus_correct/len(both_agree_r2)*100:.1f}%)")

# When agents disagree, who does judge side with?
both_disagree_r2 = df[(df["analyst_r2"].notna()) & (df["skeptic_r2"].notna()) & (df["analyst_r2"] != df["skeptic_r2"])]
print(f"\nAgents disagree in R2: {len(both_disagree_r2)} debates")

if len(both_disagree_r2) > 0:
    judge_sides_analyst = (both_disagree_r2["judge"] == both_disagree_r2["analyst_r2"]).sum()
    judge_sides_skeptic = (both_disagree_r2["judge"] == both_disagree_r2["skeptic_r2"]).sum()
    judge_third_option = len(both_disagree_r2) - judge_sides_analyst - judge_sides_skeptic
    
    print(f"  Judge sides with Analyst: {judge_sides_analyst} ({judge_sides_analyst/len(both_disagree_r2)*100:.1f}%)")
    print(f"  Judge sides with Skeptic: {judge_sides_skeptic} ({judge_sides_skeptic/len(both_disagree_r2)*100:.1f}%)")
    print(f"  Judge picks third option: {judge_third_option} ({judge_third_option/len(both_disagree_r2)*100:.1f}%)")
    
    # Who was more often correct when they disagreed?
    analyst_correct_when_disagree = (both_disagree_r2["analyst_r2"] == both_disagree_r2["gold_answer"]).sum()
    skeptic_correct_when_disagree = (both_disagree_r2["skeptic_r2"] == both_disagree_r2["gold_answer"]).sum()
    
    print(f"\n  Analyst correct when disagree: {analyst_correct_when_disagree} ({analyst_correct_when_disagree/len(both_disagree_r2)*100:.1f}%)")
    print(f"  Skeptic correct when disagree: {skeptic_correct_when_disagree} ({skeptic_correct_when_disagree/len(both_disagree_r2)*100:.1f}%)")

# === 4. Overall Performance Comparison ===
print("\n\n4. OVERALL ACCURACY COMPARISON")
print("-"*80)

valid_all = df[(df["analyst_r1"].notna()) & (df["analyst_r2"].notna()) & (df["judge"].notna())]

analyst_r1_acc = (valid_all["analyst_r1"] == valid_all["gold_answer"]).sum() / len(valid_all) * 100
analyst_r2_acc = (valid_all["analyst_r2"] == valid_all["gold_answer"]).sum() / len(valid_all) * 100
judge_acc = (valid_all["judge"] == valid_all["gold_answer"]).sum() / len(valid_all) * 100

print(f"Sample size: {len(valid_all)} debates")
print(f"Analyst R1 accuracy: {analyst_r1_acc:.2f}%")
print(f"Analyst R2 accuracy: {analyst_r2_acc:.2f}%")
print(f"Judge accuracy: {judge_acc:.2f}%")

if analyst_r2_acc > analyst_r1_acc:
    print(f"\n→ Debate improves Analyst by {analyst_r2_acc - analyst_r1_acc:.2f} percentage points")
else:
    print(f"\n→ Debate HURTS Analyst by {analyst_r1_acc - analyst_r2_acc:.2f} percentage points")

if judge_acc > analyst_r1_acc:
    print(f"→ Judge improves over Analyst R1 by {judge_acc - analyst_r1_acc:.2f} percentage points")
else:
    print(f"→ Judge does NOT improve over Analyst R1 (worse by {analyst_r1_acc - judge_acc:.2f} points)")

# === 5. Sycophancy Impact ===
print("\n\n5. SYCOPHANCY IMPACT ANALYSIS")
print("-"*80)

# In R1, when Analyst is wrong and Skeptic agrees (sycophancy)
r1_syc = df[(df["analyst_r1_correct"] == False) & (df["skeptic_agrees_r1"] == True) & (df["analyst_changed_cross_round"].notna())]
print(f"R1: Analyst WRONG + Skeptic AGREES (sycophancy): {len(r1_syc)} cases")

if len(r1_syc) > 0:
    analyst_changed_after_syc = r1_syc["analyst_changed_cross_round"].sum()
    analyst_improved_after_syc = (r1_syc["analyst_change_direction"] == "wrong_to_correct").sum()
    
    print(f"  Analyst changed in R2: {analyst_changed_after_syc} ({analyst_changed_after_syc/len(r1_syc)*100:.1f}%)")
    print(f"  Analyst improved to correct: {analyst_improved_after_syc} ({analyst_improved_after_syc/len(r1_syc)*100:.1f}%)")
    print(f"  → Sycophancy reduces chance of self-correction!")

# In R1, when Analyst is wrong and Skeptic disagrees (good skepticism)
r1_good = df[(df["analyst_r1_correct"] == False) & (df["skeptic_agrees_r1"] == False) & (df["analyst_changed_cross_round"].notna())]
print(f"\nR1: Analyst WRONG + Skeptic DISAGREES (good): {len(r1_good)} cases")

if len(r1_good) > 0:
    analyst_changed_after_challenge = r1_good["analyst_changed_cross_round"].sum()
    analyst_improved_after_challenge = (r1_good["analyst_change_direction"] == "wrong_to_correct").sum()
    
    print(f"  Analyst changed in R2: {analyst_changed_after_challenge} ({analyst_changed_after_challenge/len(r1_good)*100:.1f}%)")
    print(f"  Analyst improved to correct: {analyst_improved_after_challenge} ({analyst_improved_after_challenge/len(r1_good)*100:.1f}%)")
    print(f"  → Good skepticism helps correction!")

print("\n" + "="*80)
