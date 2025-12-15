"""
Parse debate logs with hierarchical fallback logic.

This script extracts agent responses from debate log files, implementing a two-tier
screening strategy:
1. Primary directory: arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8
2. Fallback directory: fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8

Screening criteria:
- Check if EXTRACTED PREDICTION is "none" or None
- Check if agent response is too short (< 500 chars)
- If primary fails validation, try fallback directory
"""

import os
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DebateResponses:
    """Container for parsed debate responses."""
    target_patient_analyst: str
    mortality_risk_assessor: str
    protective_factor_analyst: str
    is_complete: bool
    source_dir: str  # Track which directory was used


class DebateLogParser:
    """Parser for debate log files with hierarchical fallback."""
    
    # Agent names to extract
    AGENTS = [
        "TARGET_PATIENT_ANALYST",
        "MORTALITY_RISK_ASSESSOR", 
        "PROTECTIVE_FACTOR_ANALYST"
    ]
    
    # Minimum response length threshold
    MIN_RESPONSE_LENGTH = 500
    
    def __init__(
        self,
        primary_dir: str,
        fallback_dir: str
    ):
        """
        Initialize parser with primary and fallback directories.
        
        Args:
            primary_dir: Primary log directory path
            fallback_dir: Fallback log directory path
        """
        self.primary_dir = os.path.join(primary_dir, "debate_logs")
        self.fallback_dir = os.path.join(fallback_dir, "debate_logs")
        
        if not os.path.exists(self.primary_dir):
            raise ValueError(f"Primary directory not found: {self.primary_dir}")
        if not os.path.exists(self.fallback_dir):
            raise ValueError(f"Fallback directory not found: {self.fallback_dir}")
    
    def parse_log_file(self, filepath: str) -> Tuple[Dict[str, str], bool]:
        """
        Parse a single debate log file.
        
        Args:
            filepath: Path to log file
            
        Returns:
            Tuple of (agent_responses dict, is_valid bool)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return {}, False
        
        agent_responses = {}
        
        # Extract each agent's response
        for agent_name in self.AGENTS:
            # Pattern to match agent response section - looking for the Full response content
            # up to the next line of ====== (which marks end of section)
            pattern = rf"RAW RESPONSE from {agent_name}\s*=+\s*Response type:.*?Response length:.*?Full response:\s*(.*?)\s*=+\s*\d{{4}}-\d{{2}}-\d{{2}}"
            
            match = re.search(pattern, content, re.DOTALL)
            if match:
                response = match.group(1).strip()
                agent_responses[agent_name] = response
            else:
                # Agent response not found
                agent_responses[agent_name] = ""
        
        # Validate completeness
        is_valid = self._validate_responses(agent_responses, content)
        
        return agent_responses, is_valid
    
    def _validate_responses(self, agent_responses: Dict[str, str], content: str) -> bool:
        """
        Validate that all agent responses meet quality criteria.
        
        Args:
            agent_responses: Dictionary of agent responses
            content: Full log file content
            
        Returns:
            True if valid, False otherwise
        """
        # Check TARGET_PATIENT_ANALYST has substantial content and valid prediction (0 or 1)
        if "TARGET_PATIENT_ANALYST" not in agent_responses:
            return False
            
        target_response = agent_responses["TARGET_PATIENT_ANALYST"]
        if not target_response or len(target_response) < self.MIN_RESPONSE_LENGTH:
            return False
        
        # Must have extracted prediction of 0 or 1 (look in full content after TARGET_PATIENT_ANALYST)
        # Find the section after TARGET_PATIENT_ANALYST response
        target_section_pattern = r"RAW RESPONSE from TARGET_PATIENT_ANALYST.*?EXTRACTED PREDICTION:\s*([01])"
        target_match = re.search(target_section_pattern, content, re.DOTALL)
        if not target_match:
            return False
        
        # Check other two agents have reasonable length (they don't need predictions)
        for agent_name in ["MORTALITY_RISK_ASSESSOR", "PROTECTIVE_FACTOR_ANALYST"]:
            if agent_name not in agent_responses:
                return False
            response = agent_responses[agent_name]
            # More lenient for these agents - just need some content
            if not response or len(response) < 200:
                return False
        
        return True
    
    def get_debate_responses(
        self,
        patient_id: str,
        visit_id: str
    ) -> Optional[DebateResponses]:
        """
        Get debate responses with hierarchical fallback.
        
        Args:
            patient_id: Patient ID
            visit_id: Visit ID
            
        Returns:
            DebateResponses object or None if both sources fail
        """
        log_filename = f"debate_responses_{patient_id}_{visit_id}.log"
        
        # Try primary directory first
        primary_path = os.path.join(self.primary_dir, log_filename)
        if os.path.exists(primary_path):
            agent_responses, is_valid = self.parse_log_file(primary_path)
            
            if is_valid:
                return DebateResponses(
                    target_patient_analyst=agent_responses["TARGET_PATIENT_ANALYST"],
                    mortality_risk_assessor=agent_responses["MORTALITY_RISK_ASSESSOR"],
                    protective_factor_analyst=agent_responses["PROTECTIVE_FACTOR_ANALYST"],
                    is_complete=True,
                    source_dir="primary"
                )
        
        # Fallback to secondary directory
        fallback_path = os.path.join(self.fallback_dir, log_filename)
        if os.path.exists(fallback_path):
            agent_responses, is_valid = self.parse_log_file(fallback_path)
            
            if is_valid:
                return DebateResponses(
                    target_patient_analyst=agent_responses["TARGET_PATIENT_ANALYST"],
                    mortality_risk_assessor=agent_responses["MORTALITY_RISK_ASSESSOR"],
                    protective_factor_analyst=agent_responses["PROTECTIVE_FACTOR_ANALYST"],
                    is_complete=True,
                    source_dir="fallback"
                )
            else:
                # Fallback also failed validation
                print(f"WARNING: Both primary and fallback logs failed validation for {patient_id}_{visit_id}")
                return None
        
        # Neither directory has the file
        print(f"ERROR: No log file found for {patient_id}_{visit_id}")
        return None
    
    def get_available_samples(self) -> list[Tuple[str, str]]:
        """
        Get list of available (patient_id, visit_id) tuples from primary directory.
        
        Returns:
            List of (patient_id, visit_id) tuples
        """
        samples = []
        
        for filename in os.listdir(self.primary_dir):
            if filename.startswith("debate_responses_") and filename.endswith(".log"):
                # Extract patient_id and visit_id
                parts = filename.replace("debate_responses_", "").replace(".log", "").split("_")
                if len(parts) == 2:
                    patient_id, visit_id = parts
                    samples.append((patient_id, visit_id))
        
        return sorted(samples)


def test_parser():
    """Test the parser on a single example."""
    primary_dir = "/data/wang/junh/githubs/Debate/KARE/results/arc_rag_mor_Qwen_Qwen2.5_7B_Instruct_int_Qwen_Qwen2.5_32B_Instruct_8_8"
    fallback_dir = "/data/wang/junh/githubs/Debate/KARE/results/fallback_rag_mor_Qwen_Qwen2.5_7B_Instruct_8_8"
    
    parser = DebateLogParser(primary_dir, fallback_dir)
    
    # Debug: manually test primary file
    primary_path = os.path.join(parser.primary_dir, "debate_responses_34_0.log")
    print(f"Testing primary path: {primary_path}")
    print(f"File exists: {os.path.exists(primary_path)}")
    
    if os.path.exists(primary_path):
        agent_responses, is_valid = parser.parse_log_file(primary_path)
        print(f"\nAgent responses extracted: {list(agent_responses.keys())}")
        for agent, response in agent_responses.items():
            print(f"{agent}: {len(response)} chars")
        print(f"Is valid: {is_valid}")
        
        # Check for prediction pattern
        with open(primary_path, 'r') as f:
            content = f.read()
        prediction_pattern = r"EXTRACTED PREDICTION:\s*([01])"
        match = re.search(prediction_pattern, content)
        print(f"Prediction match found: {match is not None}")
        if match:
            print(f"Prediction value: {match.group(1)}")
    
    # Test on patient 34_0
    responses = parser.get_debate_responses("34", "0")
    
    if responses:
        print(f"\nSuccessfully parsed debate responses from {responses.source_dir} directory")
        print(f"\nTARGET_PATIENT_ANALYST (length={len(responses.target_patient_analyst)}):")
        print(responses.target_patient_analyst[:200] + "...")
    else:
        print("\nFailed to parse debate responses")
    
    # Get total available samples
    samples = parser.get_available_samples()
    print(f"\nTotal available samples: {len(samples)}")


if __name__ == "__main__":
    test_parser()
