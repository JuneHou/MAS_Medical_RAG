#!/usr/bin/env python3
"""
Improved FAISS-based Similar Patient Retrieval for Debate System
Addresses class imbalance issues and ensures all patients get negative examples.
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import faiss

# Configuration
MAX_K = 1  # Number of similar patients to retrieve per class
DATASET = 'mimic3'
TASK = 'mortality'
MAX_CONTEXT_LENGTH = 30000  # Relaxed from 20000 to get more candidates
MIN_SEARCH_NEIGHBORS = 1000  # Increased from 100 to ensure diversity

# File paths - Updated to use the correct directories
BASE_PATH = "./data"
PATIENT_CONTEXT_PATH = f"{BASE_PATH}/patient_context/base_context_qwen/patient_contexts_{DATASET}_{TASK}.json"
PATIENT_EMBEDDINGS_PATH = f"{BASE_PATH}/patient_context/base_context_qwen/patient_embeddings_{DATASET}_{TASK}.pkl"
PATIENT_DATA_PATH = f"{BASE_PATH}/ehr_data/pateint_{DATASET}_{TASK}.json"

def is_context_valid(context):
    """Check if context is within acceptable length limits."""
    return len(context) <= MAX_CONTEXT_LENGTH

def load_data():
    """Load all required data files."""
    print("Loading patient contexts...")
    with open(PATIENT_CONTEXT_PATH, "r") as f:
        patient_contexts = json.load(f)
    
    print("Loading patient data with labels...")
    with open(PATIENT_DATA_PATH, "r") as f:
        patient_data = json.load(f)
    
    print("Loading patient embeddings...")
    with open(PATIENT_EMBEDDINGS_PATH, "rb") as f:
        patient_embeddings = pickle.load(f)
    
    return patient_contexts, patient_data, patient_embeddings

def build_faiss_index(patient_embeddings, patient_ids):
    """Build FAISS index for similarity search."""
    print("Building FAISS index...")
    
    # Prepare embedding matrix
    embedding_matrix = np.array([patient_embeddings[pid] for pid in patient_ids]).astype('float32')
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embedding_matrix)
    
    # Build index (using inner product for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    
    # Create patient ID to index mapping
    patient_id_to_index = {pid: idx for idx, pid in enumerate(patient_ids)}
    
    return index, embedding_matrix, patient_id_to_index

def find_similar_patients(patient_id, patient_data, patient_contexts, 
                         index, embedding_matrix, patient_id_to_index, patient_ids):
    """Find similar patients for a given patient, ensuring balanced positive/negative examples."""
    
    if patient_id not in patient_id_to_index:
        print(f"Patient ID {patient_id} not found in embeddings.")
        return {'positive': ["None"], 'negative': ["None"]}
    
    # Skip patients with overly long contexts
    if len(patient_contexts[patient_id]) > MAX_CONTEXT_LENGTH:
        return {'positive': ["None"], 'negative': ["None"]}
    
    target_label = patient_data[patient_id]['label']
    target_idx = patient_id_to_index[patient_id]
    target_embedding = embedding_matrix[target_idx].reshape(1, -1)
    
    # Search for many more neighbors to ensure we find examples from both classes
    search_k = min(len(patient_ids) - 1, MIN_SEARCH_NEIGHBORS)
    D, I = index.search(target_embedding, search_k)
    
    # Collect candidate neighbors (excluding self and same patient instances)
    base_patient_id = patient_id.split("_")[0]
    candidate_neighbors = []
    
    for neighbor_idx in I[0]:
        neighbor_id = patient_ids[neighbor_idx]
        neighbor_base_id = neighbor_id.split("_")[0]
        
        # Skip self and other instances of same patient
        if neighbor_id == patient_id or neighbor_base_id == base_patient_id:
            continue
            
        # Check if context is valid
        if not is_context_valid(patient_contexts[neighbor_id]):
            continue
            
        candidate_neighbors.append(neighbor_id)
        
        # Stop when we have enough candidates
        if len(candidate_neighbors) >= 200:
            break
    
    # Calculate similarity scores for candidates
    similarity_scores = {}
    for neighbor_id in candidate_neighbors:
        neighbor_idx = patient_id_to_index[neighbor_id]
        similarity = np.dot(target_embedding[0], embedding_matrix[neighbor_idx])
        similarity_scores[neighbor_id] = similarity
    
    # Separate by label
    positive_candidates = [pid for pid in similarity_scores if patient_data[pid]['label'] == target_label]
    negative_candidates = [pid for pid in similarity_scores if patient_data[pid]['label'] != target_label]
    
    # Sort by similarity (highest first)
    positive_sorted = sorted(positive_candidates, key=lambda pid: similarity_scores[pid], reverse=True)
    negative_sorted = sorted(negative_candidates, key=lambda pid: similarity_scores[pid], reverse=True)
    
    # Select top candidates
    selected_positive = positive_sorted[:MAX_K]
    selected_negative = negative_sorted[:MAX_K]
    
    # Format results with context and labels
    def format_patient_context(pid):
        context = patient_contexts[pid]
        label = patient_data[pid]['label']
        return f"{context}\n\nLabel:\n{label}\n\n"
    
    result = {
        'positive': [format_patient_context(pid) for pid in selected_positive] if selected_positive else ["None"],
        'negative': [format_patient_context(pid) for pid in selected_negative] if selected_negative else ["None"]
    }
    
    return result

def main():
    """Main function to run improved FAISS retrieval."""
    
    # Load data
    patient_contexts, patient_data, patient_embeddings = load_data()
    patient_ids = list(patient_embeddings.keys())
    
    print(f"Loaded {len(patient_contexts)} patient contexts")
    print(f"Loaded {len(patient_data)} patient data entries")
    print(f"Loaded {len(patient_embeddings)} patient embeddings")
    
    # Check label distribution
    label_counts = {}
    for pid, data in patient_data.items():
        label = data['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution: {label_counts}")
    
    # Build FAISS index
    index, embedding_matrix, patient_id_to_index = build_faiss_index(patient_embeddings, patient_ids)
    
    # Process all patients
    results = {}
    patients_without_negative = 0
    patients_without_positive = 0
    
    print("Finding similar patients for all patients...")
    for patient_id in tqdm(patient_ids):
        similar_patients = find_similar_patients(
            patient_id, patient_data, patient_contexts,
            index, embedding_matrix, patient_id_to_index, patient_ids
        )
        
        results[patient_id] = similar_patients
        
        # Track statistics
        if similar_patients['negative'][0] == "None":
            patients_without_negative += 1
        if similar_patients['positive'][0] == "None":
            patients_without_positive += 1
    
    # Print statistics
    total_patients = len(patient_ids)
    print(f"\nResults Summary:")
    print(f"Total patients processed: {total_patients}")
    print(f"Patients without negative examples: {patients_without_negative} ({patients_without_negative/total_patients*100:.1f}%)")
    print(f"Patients without positive examples: {patients_without_positive} ({patients_without_positive/total_patients*100:.1f}%)")
    
    # Save results
    output_dir = f"{BASE_PATH}/patient_context/similar_patient_debate"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/patient_to_top_{MAX_K}_patient_contexts_{DATASET}_{TASK}_improved.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nImproved similar patient contexts saved to: {output_path}")
    
    # Test specific patient
    if '10721_0' in results:
        print(f"\nTest case - Patient 10721_0:")
        print(f"  Has positive examples: {results['10721_0']['positive'][0] != 'None'}")
        print(f"  Has negative examples: {results['10721_0']['negative'][0] != 'None'}")

if __name__ == "__main__":
    main()