# Multi-Agent Medical Debate System

A sophisticated multi-agent debate system for medical question answering and mortality prediction, integrating retrieval-augmented generation (RAG) with structured clinical reasoning.

## Overview

This project implements two main debate architectures:

1. **Medical QA Debate System**: Multi-agent debate for medical question answering using MedRAG retrieval
2. **KARE Mortality Prediction**: Multi-agent debate system for clinical mortality prediction with EHR data

## Key Features

- **Multi-Agent Architecture**: Specialized agents with distinct roles and reasoning patterns
- **RAG Integration**: Medical document retrieval using MedRAG and MedCPT
- **Conservative Prediction Framework**: Designed to reduce over-prediction bias in clinical settings
- **Structured Debate Rounds**: Systematic evidence gathering and integration process
- **Comprehensive Logging**: Detailed tracking of agent decisions and probability estimates

## Project Structure

```
├── KARE/                           # KARE mortality prediction system
│   ├── mortality_debate_rag.py     # Main debate system implementation
│   ├── kare_data_adapter.py       # Data loading and formatting
│   ├── run_kare_debate_mortality.py # Execution script
│   └── ...
├── analyze_debate.py              # Analysis tools
├── run_debate_medrag_rag.py       # Medical QA debate system
└── ...
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/[your-username]/multi-agent-medical-debate.git
cd multi-agent-medical-debate
```

2. **Set up conda environment**:
```bash
conda create -n medrag python=3.9
conda activate medrag
```

3. **Install dependencies**:
```bash
pip install torch transformers vllm
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install sentence-transformers
pip install datasets tqdm wandb
```

4. **Install MedRAG** (follow MedRAG installation instructions):
```bash
# Install MedRAG from source
git clone https://github.com/Teddy-XiongGZ/MedRAG.git
cd MedRAG
pip install -e .
```

## Usage

### KARE Mortality Prediction

```bash
python KARE/run_kare_debate_mortality.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --integrator_model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --dataset_path "path/to/kare/data" \
    --output_dir "results/kare_debate" \
    --num_samples 100
```

### Medical QA Debate

```bash
python run_debate_medrag_rag.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_name "medqa" \
    --output_dir "results/medqa_debate" \
    --num_samples 50
```

## System Architecture

### KARE Multi-Agent Debate

The KARE system uses a 4-agent architecture for mortality prediction:

1. **Target Patient Analyst**: Comprehensive patient assessment with risk/protective factor analysis
2. **Mortality Risk Assessor**: Identifies factors that increase mortality risk
3. **Protective Factor Analyst**: Identifies factors that support survival
4. **Balanced Clinical Integrator**: Makes final conservative prediction using all evidence

### Conservative Prediction Framework

The system implements KARE's conservative approach:
- Only predict mortality when evidence strongly indicates death is very likely
- When uncertain, err toward survival prediction
- Integrate both risk factors and protective factors in decision making

## Configuration

Key configuration options:

- **Model Selection**: Support for different LLaMA models and sizes
- **Temperature Settings**: Agent-specific temperature for response diversity
- **Token Limits**: Role-based token allocation for comprehensive reasoning
- **RAG Parameters**: Retrieval corpus selection and similarity thresholds

## Evaluation Metrics

The system tracks comprehensive metrics:
- **Accuracy, Precision, Recall, F1-score**
- **Probability Estimates**: Mortality and survival probabilities
- **Confidence Levels**: Agent confidence in predictions
- **Debate Quality**: Inter-agent agreement and reasoning quality

## Data Requirements

### KARE Dataset
- Patient EHR contexts with mortality labels
- Similar patient examples (positive/negative)
- Medical knowledge retrieval corpus

### Medical QA Dataset
- Question-answer pairs from medical exams
- MedRAG retrieval corpus (MedCorp, PubMed, etc.)

**Note**: Data files are excluded from this repository. Set up your own data following the format specifications in the documentation.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
    title={Multi-Agent Medical Debate System for Clinical Decision Making},
    author={Your Name},
    journal={Your Journal},
    year={2025}
}
```

## Acknowledgments

- [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) for medical document retrieval
- [KARE](https://github.com/xxx/KARE) for the mortality prediction framework
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference

## Contact

For questions or collaboration, please contact [your-email@domain.com]