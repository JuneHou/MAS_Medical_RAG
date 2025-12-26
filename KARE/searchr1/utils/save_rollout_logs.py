"""
Utility to save Search-R1 rollout data for analysis.
Since Search-R1's veRL fork doesn't have built-in rollout logging,
we need to manually log data during validation/training.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


class RolloutLogger:
    """Logger for Search-R1 rollout data."""
    
    def __init__(self, output_dir: str, experiment_name: str):
        """
        Args:
            output_dir: Directory to save rollout data
            experiment_name: Name of experiment
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        self.rollout_data = []
        self.step_counter = 0
    
    def log_batch(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        ground_truths: List[Any],
        data_sources: List[str] = None,
        extra_info: Dict[str, List[Any]] = None,
    ):
        """
        Log a batch of rollout data.
        
        Args:
            prompts: Input prompts
            responses: Model generated responses
            rewards: Reward scores
            ground_truths: Ground truth labels
            data_sources: Data source identifiers
            extra_info: Additional info to log (e.g., search queries, retrieval results)
        """
        batch_size = len(prompts)
        
        if data_sources is None:
            data_sources = ["unknown"] * batch_size
        
        for i in range(batch_size):
            entry = {
                "step": self.step_counter,
                "prompt": prompts[i],
                "response": responses[i],
                "reward": rewards[i],
                "ground_truth": ground_truths[i],
                "data_source": data_sources[i],
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add extra info if provided
            if extra_info:
                for key, values in extra_info.items():
                    if i < len(values):
                        entry[key] = values[i]
            
            self.rollout_data.append(entry)
    
    def save_step(self, step: int = None):
        """
        Save current rollout data to parquet and JSON.
        
        Args:
            step: Step number (uses internal counter if None)
        """
        if step is not None:
            self.step_counter = step
        
        if not self.rollout_data:
            print("No rollout data to save")
            return
        
        # Create step directory
        step_dir = os.path.join(self.output_dir, f"step_{self.step_counter}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save as parquet
        df = pd.DataFrame(self.rollout_data)
        parquet_path = os.path.join(step_dir, "rollout_data.parquet")
        df.to_parquet(parquet_path, index=False)
        
        # Save as JSON for easy reading
        json_path = os.path.join(step_dir, "rollout_data.json")
        with open(json_path, 'w') as f:
            json.dump(self.rollout_data, f, indent=2)
        
        # Save summary statistics
        summary = {
            "step": self.step_counter,
            "num_samples": len(self.rollout_data),
            "mean_reward": df["reward"].mean(),
            "std_reward": df["reward"].std(),
            "min_reward": df["reward"].min(),
            "max_reward": df["reward"].max(),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add per-source statistics if available
        if "data_source" in df.columns:
            summary["reward_by_source"] = df.groupby("data_source")["reward"].agg(
                ["mean", "std", "count"]
            ).to_dict()
        
        summary_path = os.path.join(step_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved rollout data to {step_dir}")
        print(f"  Samples: {len(self.rollout_data)}")
        print(f"  Mean reward: {summary['mean_reward']:.3f}")
        
        # Clear buffer for next step
        self.rollout_data = []
        self.step_counter += 1
    
    def save_final(self):
        """Save any remaining data and create overall summary."""
        if self.rollout_data:
            self.save_step()
        
        # Create overall summary across all steps
        all_summaries = []
        for step_dir in sorted(os.listdir(self.output_dir)):
            if not step_dir.startswith("step_"):
                continue
            
            summary_path = os.path.join(self.output_dir, step_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    all_summaries.append(json.load(f))
        
        if all_summaries:
            overall_summary = {
                "experiment": self.experiment_name,
                "total_steps": len(all_summaries),
                "total_samples": sum(s["num_samples"] for s in all_summaries),
                "steps": all_summaries,
            }
            
            overall_path = os.path.join(self.output_dir, "overall_summary.json")
            with open(overall_path, 'w') as f:
                json.dump(overall_summary, f, indent=2)
            
            print(f"Saved overall summary to {overall_path}")


def extract_prediction_from_response(response: str) -> str:
    """
    Extract prediction from Search-R1 response.
    
    Args:
        response: Model generated response
    
    Returns:
        Extracted prediction (0 or 1) or "unknown"
    """
    import re
    
    # Try different patterns
    patterns = [
        r'<answer>\s*([01])\s*</answer>',
        r'\\boxed\{([01])\}',
        r'boxed\s*\{\s*([01])\s*\}',
        r'prediction\s*:?\s*([01])',
        r'final\s+prediction\s*:?\s*([01])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return "unknown"


def extract_search_queries(response: str) -> List[str]:
    """
    Extract search queries from Search-R1 response.
    
    Args:
        response: Model generated response
    
    Returns:
        List of search queries
    """
    import re
    
    pattern = r'<search>(.*?)</search>'
    matches = re.findall(pattern, response, re.DOTALL)
    return [m.strip() for m in matches]
