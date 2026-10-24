"""Model / processor loading and prompt construction for Qwen2-VL (the small open VLM we train)."""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = ("{instruction}\nReply with only the click point as (x, y), where x and y are integers from 0 to 1000 "
          "giving the position relative to the image width and height.")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|" \
               r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def load_processor(model_id: str = MODEL_ID, max_pixels: int | None = None):
    kw = {"max_pixels": max_pixels} if max_pixels else {}
    proc = AutoProcessor.from_pretrained(model_id, **kw)
    proc.tokenizer.padding_side = "left"  # batched generation needs left padding
    return proc


def load_model(model_id: str = MODEL_ID, adapter: str | None = None, trainable: bool = False,
               lora_r: int = 16, lora_alpha: int = 32, device: str = "cuda"):
    """Base model in bf16. ``adapter`` loads a saved LoRA; ``trainable`` attaches a fresh one."""
    from peft import LoraConfig, PeftModel, get_peft_model

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                           attn_implementation="sdpa").to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    if trainable:
        for n, p in model.named_parameters():  # LoRA weights train in fp32 for stable updates
            if p.requires_grad:
                p.data = p.data.float()
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or 0
    return model


def chat_text(processor, instruction: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(instruction=instruction)}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_inputs(processor, images, instructions, device: str = "cuda"):
    texts = [chat_text(processor, i) for i in instructions]
    return processor(text=texts, images=list(images), padding=True, return_tensors="pt").to(device)
