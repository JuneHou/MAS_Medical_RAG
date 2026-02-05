# MedRAG GPU Setup - Common Error and Fix

## Problem

When using vLLM/MedRAG models, you may encounter this error:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## Root Cause

**Setting `CUDA_VISIBLE_DEVICES` multiple times or changing it after vLLM initialization causes CUDA re-initialization errors.**

The problem occurs when:
1. Code sets `CUDA_VISIBLE_DEVICES` in `main()` before creating vLLM instance
2. Code tries to save/restore the original `CUDA_VISIBLE_DEVICES` value
3. vLLM instance is created, then CUDA environment is changed again

## WRONG Approach ❌

```python
def main():
    # Setting CUDA here is TOO LATE
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Initialize vLLM
    integrator = VLLMWrapper(model_name=args.model)
    
    # Process samples...
```

OR:

```python
def initialize_model(model_name, gpu_id):
    # Saving and restoring causes problems
    original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    model = VLLMWrapper(model_name=model_name)
    
    # This breaks vLLM!
    if original_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
    
    return model
```

## CORRECT Approach ✅

**Set `CUDA_VISIBLE_DEVICES` ONCE inside the class `__init__` method and NEVER change it.**

```python
from vllm import LLM, SamplingParams

class QwenIntegrator:
    """Correct pattern - EXACT copy from mortality_single_agent_cot.py"""
    
    def __init__(self, model_name: str, gpu_id: str):
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        # Set CUDA_VISIBLE_DEVICES ONCE in __init__
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id}")
        
        # Initialize vLLM directly
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        print(f"VLLM initialized for {model_name}")
    
    def __call__(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7):
        """Generate response"""
        if "qwen" in self.model_name.lower():
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = prompt
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "</s>"],
            repetition_penalty=1.2
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text


# In main():
def main():
    args = parser.parse_args()
    
    # DO NOT set CUDA_VISIBLE_DEVICES here!
    # Let the class handle it
    
    # Initialize integrator - class sets CUDA internally
    integrator = QwenIntegrator(model_name=args.model, gpu_id=args.gpu_id)
    
    # Use integrator for all samples
    for sample in samples:
        response = integrator(prompt)
```

## Key Rules

1. ✅ **Set `CUDA_VISIBLE_DEVICES` ONCE** inside the model wrapper class `__init__`
2. ✅ **Use `vllm.LLM` directly** (not VLLMWrapper if you want full control)
3. ✅ **Initialize the model class ONCE** and reuse it for all samples
4. ❌ **NEVER save/restore `CUDA_VISIBLE_DEVICES`** - it breaks vLLM
5. ❌ **NEVER set `CUDA_VISIBLE_DEVICES` in main()** before creating the model
6. ❌ **NEVER change `CUDA_VISIBLE_DEVICES` after vLLM is initialized**

## Working Examples

See these files for correct patterns:
- `/data/wang/junh/githubs/Debate/KARE/mortality_single_agent_cot.py` - Single GPU setup
- `/data/wang/junh/githubs/Debate/KARE/mortality_single_agent_rag.py` - Multi-GPU with tensor parallelism
- `/data/wang/junh/githubs/Debate/KARE/gpt/src/run_condition_D.py` - Fixed Qwen integrator

## Summary

**The golden rule: Set CUDA_VISIBLE_DEVICES exactly ONCE inside your model class `__init__`, then NEVER touch it again.**

This ensures vLLM initializes with the correct GPU mapping and never tries to re-initialize CUDA, which causes the forked subprocess error.
