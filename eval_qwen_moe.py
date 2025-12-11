# eval_qwen_moe.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def generate(model, tok, prompt: str, device="cuda", max_new_tokens=256):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        out = model(input_ids=ids)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    sd = torch.load(args.ckpt, map_location=args.device)["model_state"]
    model.load_state_dict(sd, strict=False)

    sample = """[EDA]
You are a data EDA assistant.
Raw Table:
HouseholdID,Date,Energy_kWh,Device_Type,Region
001,2023-01-01,12.5,heating,North
...
<assistant>
"""
    out = generate(model, tok, sample, device=args.device, max_new_tokens=256)
    print("=== MODEL OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()
