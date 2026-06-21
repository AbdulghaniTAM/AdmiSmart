import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from config import MODEL_ID

print(f"Loading {MODEL_ID} onto GPU via ROCm/CUDA...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Optimized GPU Loading:
# 1. torch.bfloat16 prevents VRAM overflow while preserving precision
# 2. sdpa drastically accelerates RAG context processing
# 3. low_cpu_mem_usage ensures system RAM doesn't spike during model transfer
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", 
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config
)

# Pipeline configured specifically for deterministic agent execution
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.0, # Deterministic outputs are critical for reliable tool calling
    do_sample=False,
    return_full_text=False
)

base_llm = HuggingFacePipeline(pipeline=hf_pipe)
llm = ChatHuggingFace(llm=base_llm)

print("[SUCCESS] Optimized Compute infrastructure ready.")

def flush_memory():
    """Flushes GPU/ROCm memory cache and runs garbage collection to clear the context window."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
