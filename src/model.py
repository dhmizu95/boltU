"""Decoder-only GPT, pure PyTorch — no Hugging Face modeling classes."""
import argparse
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.dropout = cfg["dropout"]
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"])
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.attn_dropout = cfg["dropout"]
        self.resid_dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"])
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg["block_size"]

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
                wpe=nn.Embedding(cfg["block_size"], cfg["n_embd"]),
                drop=nn.Dropout(cfg["dropout"]),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]),
                ln_f=nn.LayerNorm(cfg["n_embd"]),
            )
        )
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generator: yields (next_token, sequence_so_far) each step. A `with torch.no_grad()`
        decorator would exit before the caller ever iterates (a generator function's body
        doesn't run until first `next()`), so the no_grad context is opened inside instead."""
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)

                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum_probs > top_p
                    cutoff[:, 1:] = cutoff[:, :-1].clone()
                    cutoff[:, 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
                else:
                    next_id = torch.multinomial(probs, 1)

                idx = torch.cat([idx, next_id], dim=1)
                yield next_id, idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model = GPT(cfg)
    print(model)
    print(f"total params: {model.num_params():,}")

    B, T = 2, min(64, cfg["block_size"])
    x = torch.randint(0, cfg["vocab_size"], (B, T))
    y = torch.randint(0, cfg["vocab_size"], (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, cfg["vocab_size"]), logits.shape
    print(f"forward pass ok: logits {tuple(logits.shape)}, loss {loss.item():.4f}")
