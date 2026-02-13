# How effGen Model Loading Works

## Understanding the Model Hierarchy

When you call `load_model()` from effGen, here's what happens:

```python
from effgen import load_model

# This returns a TransformersEngine (or VLLMEngine) wrapper
model = load_model("Qwen/Qwen2.5-7B-Instruct", device_map={"": 0})

# The wrapper has these attributes:
# - model.model        → The actual HuggingFace AutoModelForCausalLM
# - model.tokenizer    → The HuggingFace tokenizer  
# - model.generate()   → effGen's wrapped generate method
```

## Why We Don't Use Agent

effGen's `Agent` class is designed for agentic workflows with tools and ReAct loops:

```python
# Standard effGen pattern (NOT what we want):
agent = Agent(config=AgentConfig(
    model=model,
    tools=[...],
    temperature=0.7,  # Only temperature supported
    # NO top_p, max_new_tokens, repetition_penalty!
))
result = agent.run(prompt)  # Uses ReAct loop
```

**Problem:** `AgentConfig` only accepts `temperature` and doesn't support the other hyperparameters we need (top_p, max_new_tokens, repetition_penalty).

## Our Solution: Direct HuggingFace Access

Since we need **exact hyperparameter control**, we access the underlying HuggingFace model directly:

```python
from effgen import load_model
from transformers import AutoTokenizer
import torch

# Load via effGen (handles quantization, device mapping, etc.)
model_wrapper = load_model("Qwen/Qwen2.5-7B-Instruct", device_map={"": 0})

# Access the actual HuggingFace model
hf_model = model_wrapper.model  # This is AutoModelForCausalLM

# Load tokenizer separately  
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Generate with EXACT parameters
inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
with torch.no_grad():
    outputs = hf_model.generate(
        **inputs,
        temperature=0.7,  # EXACT
        top_p=0.9,  # EXACT
        max_new_tokens=32768,  # EXACT
        repetition_penalty=1.2,  # EXACT
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
```

## Why This Works

1. **effGen's value**: Handles model loading, quantization, device mapping, memory optimization
2. **HuggingFace's value**: Provides exact control over generation parameters
3. **Best of both**: We use effGen for setup, HuggingFace for generation

## Our Implementation

In `mortality_single_agent_effgen_rag.py`:

```python
# Line ~230: Load model via effGen
self.model = load_model(
    model_name,
    device_map={"": 0},
    trust_remote_code=True,
    attn_implementation="eager"
)

# Line ~431: Generate with direct HuggingFace calls
tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

with torch.no_grad():
    outputs = self.model.generate(  # This is model_wrapper.model.generate()
        **inputs,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=32768,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
```

## Verification

To confirm `self.model` is the HuggingFace model:

```python
from effgen import load_model

model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
print(type(model))  # <class 'effgen.models.transformers_engine.TransformersEngine'>
print(type(model.model))  # <class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>

# This is the actual HuggingFace model we can use directly!
```

## Summary

✓ **effGen's `load_model()`**: Returns a wrapper (TransformersEngine or VLLMEngine)
✓ **Wrapper's `.model`**: The actual HuggingFace AutoModelForCausalLM  
✓ **Our approach**: Use `.model.generate()` directly for exact parameter control
✓ **Why not Agent**: AgentConfig doesn't support all needed hyperparameters
✓ **Result**: Exact parity with original vllm implementation
