# moe_qwen_patch.py
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit


# -----------------------------
# Utilities
# -----------------------------
def _find_hidden_and_intermediate(mlp_module: nn.Module) -> Tuple[int, int]:
    """
    Try to infer (hidden_size, intermediate_size) from a Qwen-style MLP.
    """
    if hasattr(mlp_module, "gate_proj") and hasattr(mlp_module, "up_proj") and hasattr(mlp_module, "down_proj"):
        inter = mlp_module.gate_proj.out_features
        hid = mlp_module.gate_proj.in_features
        return hid, inter

    linears = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    if len(linears) >= 3:
        ins = [l.in_features for l in linears]
        outs = [l.out_features for l in linears]
        hidden_guess = max(set(ins), key=ins.count)
        inter_guess = max(set(outs), key=outs.count)
        return hidden_guess, inter_guess

    raise ValueError("Could not infer hidden/intermediate sizes from MLP block")


# -----------------------------
# Expert FFN (Qwen-style SwiGLU)
# -----------------------------
class QwenExpertFFN(nn.Module):
    """
    Matches Qwen(Qwen2) MLP layout with optional 4-bit quantization
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False, use_4bit: bool = False, device=None):
        super().__init__()
        
        if use_4bit:
            # Create on CPU first to avoid OOM during init
            self.gate_proj = Linear4bit(
                hidden_size, intermediate_size, bias=bias,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4',
                device='cpu'  # Create on CPU
            )
            self.up_proj = Linear4bit(
                hidden_size, intermediate_size, bias=bias,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4',
                device='cpu'
            )
            self.down_proj = Linear4bit(
                intermediate_size, hidden_size, bias=bias,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4',
                device='cpu'
            )
            
            # Move to target device if provided
            if device is not None and device.type != 'cpu':
                self.gate_proj = self.gate_proj.to(device)
                self.up_proj = self.up_proj.to(device)
                self.down_proj = self.down_proj.to(device)
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
            
            if device is not None:
                self.gate_proj = self.gate_proj.to(device=device, dtype=torch.bfloat16)
                self.up_proj = self.up_proj.to(device=device, dtype=torch.bfloat16)
                self.down_proj = self.down_proj.to(device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.gate_proj(x))
        b = self.up_proj(x)
        return self.down_proj(a * b)


# -----------------------------
# Top-2 Router (with Gaussian noise)
# -----------------------------
class Top2Router(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, noise_std: float = 1e-2, device=None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_experts, bias=False)
        self.n_experts = n_experts
        self.noise_std = noise_std
        
        # Move to device if provided
        if device is not None:
            self.proj = self.proj.to(device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, training: bool = True):
        # x: [B, T, H]
        # Cast input to match weight dtype (important for gradient checkpointing)
        if x.dtype != self.proj.weight.dtype:
            x = x.to(self.proj.weight.dtype)
        
        logits = self.proj(x)
        if training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gate = torch.softmax(logits, dim=-1)  # [B,T,E]
        top2_val, top2_idx = torch.topk(gate, k=2, dim=-1)  # [B,T,2]
        return top2_val, top2_idx, gate


# -----------------------------
# MoE FFN (Top-2, naive dispatcher)
# -----------------------------
class QwenMoEFFN(nn.Module):
    """
    Drop-in replacement for Qwen MLP
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int = 8,
        capacity_factor: float = 1.25,
        noise_std: float = 1e-2,
        bias: bool = False,
        use_4bit: bool = False,
        device=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [QwenExpertFFN(hidden_size, intermediate_size, bias=bias, use_4bit=use_4bit, device=device) 
             for _ in range(n_experts)]
        )
        self.router = Top2Router(hidden_size, n_experts, noise_std=noise_std, device=device)

        self._last_aux = None

    def aux_loss(self):
        if self._last_aux is None:
            # Get device from router
            device = next(self.router.parameters()).device
            return torch.tensor(0.0, device=device)
        return self._last_aux

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: [B, T, H]
        """
        B, T, H = x.shape
        y = torch.zeros_like(x)

        # Router
        top2_val, top2_idx, gate = self.router(x, training=training)
        tokens = B * T
        cap = max(1, int(self.capacity_factor * (tokens * 2 / self.n_experts)))

        prob_per_expert = gate.mean(dim=(0, 1))
        used_per_expert = torch.zeros(self.n_experts, device=x.device)

        flat_x = x.reshape(-1, H)
        flat_y = y.reshape(-1, H)

        for k in range(2):
            idx = top2_idx[..., k].reshape(-1)
            val = top2_val[..., k].reshape(-1, 1)

            for e in range(self.n_experts):
                mask = (idx == e)
                if not mask.any():
                    continue
                sel = mask.nonzero(as_tuple=False).squeeze(-1)
                take = min(sel.numel(), cap - int(used_per_expert[e].item()))
                if take <= 0:
                    continue
                sel = sel[:take]

                out = self.experts[e](flat_x.index_select(0, sel))
                weighted = out * val.index_select(0, sel)
                flat_y.index_copy_(0, sel, flat_y.index_select(0, sel) + weighted)
                used_per_expert[e] += take

        y = flat_y.view(B, T, H)

        eps = 1e-9
        frac_used = used_per_expert.clamp_min(eps) / (tokens * 2 + eps)
        balance_loss = (prob_per_expert * frac_used).sum() * self.n_experts
        self._last_aux = balance_loss

        return y


# -----------------------------
# Patching helpers
# -----------------------------
def patch_qwen_mlp_to_moe(block: nn.Module, n_experts: int, capacity_factor: float, noise_std: float, use_4bit: bool = False) -> Optional[nn.Module]:
    """
    If the block has a Qwen-style MLP, replace it with QwenMoEFFN.
    """
    mlp_attr = None
    for cand in ["mlp", "feed_forward", "ffn"]:
        if hasattr(block, cand):
            mlp_attr = cand
            break
    if mlp_attr is None:
        return None

    mlp = getattr(block, mlp_attr)
    try:
        hidden_size, intermediate_size = _find_hidden_and_intermediate(mlp)
    except Exception:
        return None

    # Get device from original MLP
    try:
        device = next(mlp.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create MoE with device passed to constructor
    moe = QwenMoEFFN(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_experts=n_experts,
        capacity_factor=capacity_factor,
        noise_std=noise_std,
        bias=False,
        use_4bit=use_4bit,
        device=device,
    )
    for expert in moe.experts:
        # Copy trained weights to each expert
        expert.gate_proj.weight.data = mlp.gate_proj.weight.data.clone()
        expert.up_proj.weight.data = mlp.up_proj.weight.data.clone()
        expert.down_proj.weight.data = mlp.down_proj.weight.data.clone()

    setattr(block, mlp_attr, moe)
    return moe


def patch_model_with_moe(model: nn.Module, n_experts: int = 8, capacity_factor: float = 1.25, noise_std: float = 1e-2, use_4bit: bool = False) -> List[nn.Module]:
    """
    Walks model layers and replaces Qwen MLPs with MoE.
    """
    moe_modules = []
    print(f"[INFO] Patching model with MoE FFNs (4-bit: {use_4bit})...")
    for name, module in model.named_modules():
        if any(hasattr(module, cand) for cand in ["mlp", "feed_forward", "ffn"]):
            moe = patch_qwen_mlp_to_moe(module, n_experts, capacity_factor, noise_std, use_4bit=use_4bit)
            if moe is not None:
                moe_modules.append(moe)
                
    if not moe_modules:
        raise RuntimeError("No Qwen-style MLPs found to patch.")
    
    print(f"[INFO] Successfully patched {len(moe_modules)} layers with MoE")
    return moe_modules


def freeze_all_but_moe(model: nn.Module):
    """
    Freeze all weights except MoE experts + router.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze MoE modules
    for module in model.modules():
        if isinstance(module, QwenMoEFFN):
            # Unfreeze all parameters in MoE module
            for param in module.parameters():
                if param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                    param.requires_grad = True

def sum_aux_losses(moe_modules: List[nn.Module]) -> torch.Tensor:
    """
    Collect and sum aux losses from each MoE module.
    """
    aux = None
    for m in moe_modules:
        if isinstance(m, QwenMoEFFN):
            val = m.aux_loss()
            aux = val if aux is None else aux + val
    if aux is None:
        device = next(moe_modules[0].parameters()).device if moe_modules else "cpu"
        aux = torch.tensor(0.0, device=device)
    return aux