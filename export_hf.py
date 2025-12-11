# export_hf.py
"""
If you want to persist the MoE-patched model as a usual HF folder
(so you can `from_pretrained` it without re-patching), just save_pretrained.
Note: vLLM may not support custom MoE modules; serving might require
a supported MoE backend (e.g., DS-MoE, Tutel) or sticking to dense blocks.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default="qwen_moe_export")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    sd = torch.load(args.ckpt, map_location="cpu")["model_state"]
    model.load_state_dict(sd, strict=False)

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[INFO] Saved HF folder to: {args.out_dir}")

if __name__ == "__main__":
    main()
