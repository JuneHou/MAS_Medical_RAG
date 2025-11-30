# Setup Guide

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for large models)
- At least 16GB RAM (32GB+ recommended for 70B models)

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n medrag python=3.9
conda activate medrag
```

### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install MedRAG (Required for Medical Retrieval)

```bash
git clone https://github.com/Teddy-XiongGZ/MedRAG.git
cd MedRAG
pip install -e .
cd ..
```

### 5. Install FAISS with GPU Support (Optional but Recommended)

```bash
# Uninstall CPU version first
pip uninstall faiss-cpu

# Install GPU version
conda install faiss-gpu -c conda-forge
```

## Data Setup

### KARE Dataset Setup

1. **Obtain KARE Dataset**: Follow KARE repository instructions to get access
2. **Data Structure**: Ensure your data follows this structure:
```
data/
├── kare_mortality_data.json
├── similar_patients/
│   ├── positive_similars.json
│   └── negative_similars.json
└── medical_knowledge/
    └── retrieval_corpus.json
```

### Medical QA Dataset Setup

1. **Download Datasets**: 
   - MedQA: https://github.com/jind11/MedQA
   - PubMedQA: https://pubmedqa.github.io/
   - BioASQ: http://bioasq.org/

2. **MedRAG Corpus**: Follow MedRAG setup for retrieval corpus

## Model Setup

### Download Models via Hugging Face

```bash
# For 8B models (faster inference)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct

# For 70B models (better performance, requires more GPU memory)
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct
```

### vLLM Configuration

Create a model configuration file `models/model_config.yaml`:

```yaml
models:
  llama3_8b:
    name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu_memory_utilization: 0.8
    max_model_len: 8192
    
  llama3_70b:
    name: "meta-llama/Meta-Llama-3.1-70B-Instruct"
    gpu_memory_utilization: 0.95
    max_model_len: 4096
    tensor_parallel_size: 4  # Adjust based on available GPUs
```

## Quick Test

### Test KARE System

```bash
cd KARE
python mortality_debate_rag.py --test
```

### Test Medical QA System

```bash
python run_debate_medrag_rag.py --dataset medqa --num_samples 5 --test
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `max_model_len` in model config
   - Use smaller batch sizes
   - Use 8B model instead of 70B

2. **MedRAG Import Errors**:
   - Ensure MedRAG is installed in the same conda environment
   - Check Python path includes MedRAG directory

3. **FAISS Issues**:
   - Reinstall with correct CPU/GPU version
   - Check CUDA compatibility

### Performance Optimization

1. **GPU Memory**:
   - Use `gpu_memory_utilization: 0.9` for maximum utilization
   - Enable tensor parallelism for multi-GPU setups

2. **Inference Speed**:
   - Use smaller models for development/testing
   - Enable vLLM's continuous batching
   - Use appropriate `max_tokens` limits

## Directory Structure After Setup

```
multi-agent-medical-debate/
├── data/                    # Your datasets (not in git)
├── results/                 # Output results (not in git)
├── models/                  # Downloaded models (not in git)
├── KARE/                    # KARE mortality prediction
├── logs/                    # Debug logs (not in git)
├── requirements.txt
├── README.md
└── setup.md
```

## Next Steps

1. Run the quick tests to verify setup
2. Configure your specific datasets and models
3. Review the main README.md for usage instructions
4. Check out the example scripts in each subdirectory