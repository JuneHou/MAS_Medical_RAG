# Multi-Agent Medical Debate Playground

This repository provides an experiment harness for comparing two similar-sized
language models on the [MedQA](https://huggingface.co/datasets/medalpaca/medqa)
benchmark using a debate protocol.  It is designed to pit a general-purpose
Llama 3.1 8B assistant against the medical-focused
[`chaoyi-wu/PMC_LLAMA_7B`](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B) model
while both are served through [vLLM](https://github.com/vllm-project/vllm).

## Requirements

* Python 3.10+
* CUDA-enabled GPUs with sufficient memory to host the models (e.g. two A100s)
* Access to the Hugging Face model and dataset repositories (requires `huggingface_hub`
authentication for gated models, if applicable)

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Launching the vLLM servers

Use the helper script to start two OpenAI-compatible vLLM servers, one per GPU.
The example below binds the Llama 3.1 8B model to GPU `3` on port `8000` and the
PMC-LLaMA model to GPU `4` on port `8001`.

```bash
python scripts/launch_vllm_servers.py \
  --llama-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --llama-gpu 3 --llama-port 8000 \
  --pmc-model chaoyi-wu/PMC_LLAMA_7B \
  --pmc-gpu 4 --pmc-port 8001
```

The script keeps running and streams logs for both servers.  Press `Ctrl+C` when
you want to shut everything down.

### Manual launch

If you prefer to start the servers yourself, the following commands are
sufficient:

```bash
CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY=dummy \
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000

CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY=dummy \
python -m vllm.entrypoints.openai.api_server \
  --model chaoyi-wu/PMC_LLAMA_7B --port 8001
```

## Running the debate evaluation

Once the servers are up, run the experiment script.  It will download the
MedQA dataset via the 🤗 `datasets` library, orchestrate the debate, and emit the
transcripts plus basic accuracy metrics.

```bash
# Optionally export a shared API key for the OpenAI-compatible endpoints.
export OPENAI_API_KEY=debate-key

python run_debate.py \
  --llama-url http://localhost:8000 \
  --pmc-url http://localhost:8001 \
  --medqa-split validation \
  --num-samples 20 \
  --rounds 3 \
  --output outputs/medqa_debate.jsonl
```

This command writes two files:

* `outputs/medqa_debate.jsonl`: a JSONL file containing the full conversation
  history for each question.
* `outputs/medqa_debate.metrics.json`: aggregate accuracy statistics derived
  from the final answers of both agents.

Adjust `--num-samples` for larger evaluations.  The debate loop currently runs a
fixed sequence of alternating rebuttals beginning with the generalist model.

## Repository layout

```
.
├── README.md                # You are here
├── requirements.txt         # Python dependencies
├── run_debate.py            # Debate orchestration script
└── scripts/
    └── launch_vllm_servers.py  # Helper to start both vLLM servers
```

## Notes

* Ensure you have the necessary Hugging Face credentials cached locally before
  launching vLLM, especially for Llama 3.1 weights.
* The debate script uses simple regex heuristics to extract the final answer
  letter (A/B/C/...).  Inspect the transcripts if the metrics look suspicious.
* Feel free to extend the conversation policy or add a final judge model for
  more elaborate debate schemes.
