# train_qwen_moe.py
import math
import argparse
import os

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from moe_qwen_patch import patch_model_with_moe, sum_aux_losses, QwenMoEFFN
from data_sft import build_dataloader

# Set memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=512)  # Reduced from 1024
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--capacity_factor", type=float, default=1.25)
    parser.add_argument("--noise_std", type=float, default=1e-2)
    parser.add_argument("--aux_weight", type=float, default=0.01)
    parser.add_argument("--save_best", type=str, default="deepanalyze_moe_best_2")
    parser.add_argument("--save_each_epoch", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=4)  # Gradient accumulation
    args = parser.parse_args()
    
    wandb.init(
        project="moe-qwen",
        name=f"moe_experts{args.n_experts}_lr{args.lr}_aux{args.aux_weight}",
        config={
            "base_model": args.base_model,
            "n_experts": args.n_experts,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "aux_weight": args.aux_weight,
            "capacity_factor": args.capacity_factor,
            "epochs": args.epochs,
        }
    )

    dtype = torch.bfloat16
    
    # Tokenizer
    tok = AutoTokenizer.from_pretrained("RUC-DataLab/DeepAnalyze-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        "RUC-DataLab/DeepAnalyze-8B",
        quantization_config=bnb,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Prepare model for k-bit training (required for LoRA on quantized models)
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Patch MLP -> MoE with 4-bit experts
    print("[INFO] Patching model with 4-bit MoE...")
    moe_modules = patch_model_with_moe(
        model,
        n_experts=args.n_experts,
        # capacity_factor=args.capacity_factor,
        # noise_std=args.noise_std,
        use_4bit=False,
    )
    
    # Apply LoRA - train adapters on the router (which is bfloat16, not quantized)
    print("[INFO] Applying LoRA to MoE routers...")
    
    # Find all router projection layers to apply LoRA
    target_modules = []
    for name, module in model.named_modules():
        if "mlp.router.proj" in name:
            # Extract the relative path for LoRA targeting
            target_modules.append(name.split(".")[-1])  # Just "proj"
    
    # If we couldn't find router.proj, use a more generic approach
    if not target_modules:
        target_modules = ["proj"]  # Router projection layer
    
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable/1e6:.2f}M")
    
    if trainable == 0:
        raise ValueError("No trainable parameters!")
    
    model.train()

    # Data
    dl = build_dataloader(args.data, tok, max_len=args.max_len, batch_size=args.batch_size, shuffle=True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Creating optimizer with {len(trainable_params)} parameter groups")

    # Optimizer
    opt = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss, total_aux = 0.0, 0.0
        steps = 0

        progress = tqdm(dl, desc=f"Epoch {epoch}", leave=True)
        for batch_idx, (input_ids, attn, labels) in enumerate(progress):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            # Forward
            out = model(input_ids=input_ids, labels=labels)
            lm_loss = out.loss

            # Collect aux loss
            aux_loss = sum(m.aux_loss().to(args.device) for m in moe_modules)

            loss = (lm_loss + args.aux_weight * aux_loss) / args.grad_accum_steps

            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            total_loss += float(lm_loss.item())
            total_aux += float(aux_loss.item())
            steps += 1
            
            wandb.log({
                "train/lm_loss": lm_loss.item(),
                "train/aux_loss": aux_loss.item(),
                "train/total_loss": loss.item() * args.grad_accum_steps,
                "train/step": steps,
            })

            progress.set_postfix({"lm": f"{total_loss/steps:.4f}", "aux": f"{total_aux/steps:.4f}"})
            
            # Free memory periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg = total_loss / max(1, steps)
        print(f"[INFO] epoch {epoch} avg_lm={avg:.4f} avg_aux={total_aux/max(1,steps):.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train/epoch_avg_lm_loss": avg,
            "train/epoch_avg_aux_loss": total_aux / max(1, steps),
        })

        # Save best
        if avg < best_loss:
            best_loss = avg
            # Save LoRA adapters only
            model.save_pretrained(args.save_best)
            print(f"[INFO] ⭐ Saved best LoRA adapters to {args.save_best} (loss={avg:.4f})")

        if args.save_each_epoch:
            path = f"qwen_moe_epoch{epoch}"
            model.save_pretrained(path)
            print(f"[INFO] Saved {path}")
            
    wandb.finish()


if __name__ == "__main__":
    main()