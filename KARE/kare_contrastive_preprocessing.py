#!/usr/bin/env python3
"""
KARE Contrastive Preprocessing for Label-Blind Similar Patient Analysis
Implements shared/unique formatting to keep analysts label-blind.
"""

import re
from typing import Dict, List, Tuple, Optional


def normalize_concept(line: str) -> str:
    """
    Normalize a concept line by removing numbering and metadata annotations.
    
    Input examples:
      '1. pneumonia (new)'
      '2. acute and unspecified renal failure (continued from previous visit)'
    Output:
      'pneumonia'
      'acute and unspecified renal failure'
    
    Args:
        line: Formatted concept line from patient context
        
    Returns:
        Normalized concept string
    """
    # Remove numbering prefix like "1. "
    s = re.sub(r'^\s*\d+\.\s*', '', line)
    # Remove trailing "(new)" or "(continued ...)" metadata
    s = re.sub(r'\s*\(.*?\)\s*$', '', s)
    return s.strip()


def parse_patient_context(text: str) -> Dict[int, Dict[str, List[str]]]:
    """
    Parse formatted patient context text into structured visit data.
    
    Returns:
        visits[visit_num]['ICD'] = [concept_str, ...]
        visits[visit_num]['Procedure'] = [concept_str, ...]
        visits[visit_num]['Medication'] = [concept_str, ...]
    
    Args:
        text: Formatted patient context string
        
    Returns:
        Dictionary mapping visit numbers to sector data
    """
    visits = {}
    current_visit = None
    current_sector = None
    
    for raw_line in text.splitlines():
        line = raw_line.strip()
        
        # Detect visit header: "Visit 0:", "Visit 1:", etc.
        visit_match = re.match(r'^Visit\s+(\d+):', line)
        if visit_match:
            current_visit = int(visit_match.group(1))
            visits[current_visit] = {"ICD": [], "Procedure": [], "Medication": []}
            current_sector = None
            continue
        
        # Detect sector headers
        if line == "Conditions:":
            current_sector = "ICD"
            continue
        if line == "Procedures:":
            current_sector = "Procedure"
            continue
        if line == "Medications:":
            current_sector = "Medication"
            continue
        
        # Skip empty lines or "No documented" lines
        if not line or line.lower().startswith("no documented"):
            continue
        
        # Skip patient ID line
        if line.startswith("Patient ID:"):
            continue
        
        # Process concept lines (start with digit)
        if current_visit is not None and current_sector is not None:
            if line and line[0].isdigit():
                normalized = normalize_concept(line)
                if normalized:  # Only add non-empty concepts
                    visits[current_visit][current_sector].append(normalized)
    
    return visits


def union_order(target_list: List[str], similar_list: List[str]) -> List[str]:
    """
    Create stable ordering from union of target and similar lists.
    
    Stable order: keep target order first, then append similar-only items in their order.
    This ensures consistent display across target and similar views.
    
    Args:
        target_list: Concepts from target patient
        similar_list: Concepts from similar patient
        
    Returns:
        Ordered union of concepts
    """
    seen = set()
    ordered = []
    
    # Add target items first (preserves target order)
    for item in target_list:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    
    # Add similar-only items (preserves similar order)
    for item in similar_list:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    
    return ordered


def build_contrastive_visit_view(
    target_visits: Dict[int, Dict[str, List[str]]], 
    similar_visits: Dict[int, Dict[str, List[str]]]
) -> Tuple[str, str]:
    """
    Build contrastive views showing shared and unique items for each visit.
    
    This creates two parallel formatted strings:
    - Target view: shows what target has (shared with similar, unique to target)
    - Similar view: shows what similar has (shared with target, unique to similar)
    
    Args:
        target_visits: Parsed target patient visits
        similar_visits: Parsed similar patient visits
        
    Returns:
        Tuple of (target_view_text, similar_view_text)
    """
    # Find maximum visit number across both patients
    max_visit = max(
        max(target_visits.keys(), default=-1),
        max(similar_visits.keys(), default=-1)
    )
    
    target_lines = []
    similar_lines = []
    
    for visit_num in range(max_visit + 1):
        # Get visit data (default to empty if visit doesn't exist)
        target_data = target_visits.get(visit_num, {"ICD": [], "Procedure": [], "Medication": []})
        similar_data = similar_visits.get(visit_num, {"ICD": [], "Procedure": [], "Medication": []})
        
        # Add visit header
        target_lines.append(f"Visit {visit_num}:")
        similar_lines.append(f"Visit {visit_num}:")
        
        # Process each sector (ICD, Procedure, Medication)
        for sector in ["ICD", "Procedure", "Medication"]:
            target_items = target_data.get(sector, [])
            similar_items = similar_data.get(sector, [])
            
            # Create union ordering for stable display
            union_ordered = union_order(target_items, similar_items)
            
            # Compute sets for intersection/difference
            target_set = set(target_items)
            similar_set = set(similar_items)
            
            # Classify items: shared (in both), target-only, similar-only
            shared = [x for x in union_ordered if (x in target_set and x in similar_set)]
            target_only = [x for x in union_ordered if (x in target_set and x not in similar_set)]
            similar_only = [x for x in union_ordered if (x in similar_set and x not in target_set)]
            
            # Map sector to readable name
            sector_name = {
                "ICD": "Conditions",
                "Procedure": "Procedures", 
                "Medication": "Medications"
            }[sector]
            
            # ==== Render for TARGET view ====
            target_lines.append(f"{sector_name}:")
            target_lines.append("  Shared with Similar:")
            if shared:
                for i, concept in enumerate(shared, 1):
                    target_lines.append(f"    {i}. {concept}")
            else:
                target_lines.append("    (none)")
            
            target_lines.append("  Unique to Target:")
            if target_only:
                for i, concept in enumerate(target_only, 1):
                    target_lines.append(f"    {i}. {concept}")
            else:
                target_lines.append("    (none)")
            
            # ==== Render for SIMILAR view ====
            similar_lines.append(f"{sector_name}:")
            similar_lines.append("  Shared with Target:")
            if shared:
                for i, concept in enumerate(shared, 1):
                    similar_lines.append(f"    {i}. {concept}")
            else:
                similar_lines.append("    (none)")
            
            similar_lines.append("  Unique to Similar:")
            if similar_only:
                for i, concept in enumerate(similar_only, 1):
                    similar_lines.append(f"    {i}. {concept}")
            else:
                similar_lines.append("    (none)")
        
        # Add spacing between visits
        target_lines.append("")
        similar_lines.append("")
    
    return "\n".join(target_lines).strip(), "\n".join(similar_lines).strip()


def build_label_blind_analyst_inputs(
    target_context_text: str,
    positive_similar_text: str,
    negative_similar_text: str
) -> Tuple[str, str]:
    """
    Build label-blind inputs for two analysts analyzing similar patients.
    
    Analyst 1 analyzes target vs positive similar (mortality=1 case)
    Analyst 2 analyzes target vs negative similar (survival=0 case)
    
    Both analysts receive IDENTICAL formatting and instructions, with NO labels.
    They only see shared/unique patterns.
    
    Args:
        target_context_text: Formatted target patient EHR context
        positive_similar_text: Formatted similar patient with mortality=1
        negative_similar_text: Formatted similar patient with survival=0
        
    Returns:
        Tuple of (analyst1_input, analyst2_input)
    """
    # Parse all contexts into structured visits
    target_visits = parse_patient_context(target_context_text)
    positive_visits = parse_patient_context(positive_similar_text)
    negative_visits = parse_patient_context(negative_similar_text)
    
    # Build contrastive view for target vs positive similar
    target_pos_view, positive_view = build_contrastive_visit_view(target_visits, positive_visits)
    
    # Build contrastive view for target vs negative similar  
    target_neg_view, negative_view = build_contrastive_visit_view(target_visits, negative_visits)
    
    # Create label-blind analyst inputs (IDENTICAL instructions)
    analyst_instruction = """Analyze the clinical patterns between the target patient and similar patient.

Focus on:
1. **Shared patterns**: What conditions, procedures, and medications appear in BOTH patients?
2. **Unique to target**: What is present ONLY in the target patient?
3. **Unique to similar**: What is present ONLY in the similar patient?
4. **Temporal progression**: How do shared/unique patterns evolve across visits?
5. **Clinical significance**: Which shared or unique factors are most clinically relevant?

IMPORTANT: Do NOT speculate about outcomes or mortality risk. Only analyze clinical similarity patterns."""
    
    analyst1_input = f"""## Target Patient EHR Context ##
{target_pos_view}

## Similar Patient EHR Context ##
{positive_view}

{analyst_instruction}"""
    
    analyst2_input = f"""## Target Patient EHR Context ##
{target_neg_view}

## Similar Patient EHR Context ##
{negative_view}

{analyst_instruction}"""
    
    return analyst1_input, analyst2_input


def format_integrator_history_with_labels(
    analyst1_response: str,
    analyst2_response: str
) -> str:
    """
    Format analyst responses for integrator with outcome labels added.
    
    This is where labels are revealed:
    - Analyst 1 analyzed a mortality=1 case
    - Analyst 2 analyzed a survival=0 case
    
    Args:
        analyst1_response: Response from analyst analyzing positive similar (mortality=1)
        analyst2_response: Response from analyst analyzing negative similar (survival=0)
        
    Returns:
        Formatted history string with labels for integrator
    """
    history = f"""## Analysis of Similar Cases ##

### Similar Case with Mortality=1 (positive class) Analysis:
{analyst1_response}

### Similar Case with Survival=0 (negative class) Analysis:
{analyst2_response}

Note: The above analyses were conducted without knowledge of outcomes. Use these pattern comparisons to inform your assessment."""
    
    return history


def preprocess_for_debate(
    target_context: str,
    positive_similars: str,
    negative_similars: str
) -> Dict[str, str]:
    """
    Main preprocessing function to prepare inputs for label-blind debate.
    
    This function:
    1. Parses target and similar patient contexts
    2. Creates contrastive shared/unique views
    3. Generates label-blind analyst inputs
    4. Prepares integrator input template (labels added after analyst responses)
    
    Args:
        target_context: Target patient EHR context
        positive_similars: Similar patient(s) with mortality=1
        negative_similars: Similar patient(s) with survival=0
        
    Returns:
        Dictionary with:
        - 'analyst1_input': Label-blind input for mortality case analyst
        - 'analyst2_input': Label-blind input for survival case analyst
        - 'original_target': Original target context (for reference)
    """
    # Handle multiple similar patients (split by double newline if present)
    # Take first similar patient from each category for contrastive analysis
    pos_patients = [p.strip() for p in positive_similars.split('\n\n\n') if p.strip()]
    neg_patients = [p.strip() for p in negative_similars.split('\n\n\n') if p.strip()]
    
    # Use first similar patient from each category
    pos_similar = pos_patients[0] if pos_patients else "No similar patients available."
    neg_similar = neg_patients[0] if neg_patients else "No similar patients available."
    
    # Build label-blind analyst inputs
    analyst1_input, analyst2_input = build_label_blind_analyst_inputs(
        target_context,
        pos_similar,
        neg_similar
    )
    
    return {
        'analyst1_input': analyst1_input,
        'analyst2_input': analyst2_input,
        'original_target': target_context,
        'positive_similar_raw': pos_similar,
        'negative_similar_raw': neg_similar
    }


# Test/demo function
if __name__ == "__main__":
    """Test the preprocessing with patient 188_2 example"""
    import sys
    sys.path.insert(0, '/data/wang/junh/githubs/Debate/KARE')
    from kare_data_adapter import KAREDataAdapter
    
    # Load data
    adapter = KAREDataAdapter(split="test")
    
    # Find patient 188_2
    for idx in range(len(adapter.test_data)):
        sample = adapter.get_test_sample(idx)
        if sample['patient_id'] == "188_2":
            print("="*80)
            print(f"PREPROCESSING TEST: Patient {sample['patient_id']}")
            print("="*80)
            
            # Run preprocessing
            preprocessed = preprocess_for_debate(
                sample['target_context'],
                sample['positive_similars'],
                sample['negative_similars']
            )
            
            print("\n" + "="*80)
            print("ANALYST 1 INPUT (Label-Blind - Analyzing Mortality=1 Case)")
            print("="*80)
            print(preprocessed['analyst1_input'][:2000] + "\n... (truncated)")
            
            print("\n" + "="*80)
            print("ANALYST 2 INPUT (Label-Blind - Analyzing Survival=0 Case)")
            print("="*80)
            print(preprocessed['analyst2_input'][:2000] + "\n... (truncated)")
            
            # Test integrator history formatting
            mock_analyst1_response = "Analysis: Target and similar both show severe liver disease..."
            mock_analyst2_response = "Analysis: Target and similar share respiratory failure..."
            
            integrator_history = format_integrator_history_with_labels(
                mock_analyst1_response,
                mock_analyst2_response
            )
            
            print("\n" + "="*80)
            print("INTEGRATOR HISTORY (With Labels Added)")
            print("="*80)
            print(integrator_history)
            
            break
