import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class causalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.in_linear = nn.Linear(d_model, d_model * 3)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.in_linear(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch_size, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = causalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, n_heads, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the output head shares weights with the token embedding
        self.head.weight = self.token_embedding.weight

        # Init (after tying, so the shared weight is initialised once)
        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("out_linear.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        assert seq_len <= self.context_length, (
            f"Sequence length {seq_len} exceeds context length {self.context_length}"
        )

        pos = torch.arange(seq_len, device=x.device)  # (T,)
        x = self.token_embedding(x) + self.position_embedding(pos)  # (B, T, C) + (T, C)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Raw logits go straight into cross_entropy (it applies log-softmax internally)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    model = GPT(vocab_size=8192, d_model=384, context_length=256, n_heads=6, n_layers=6)
    x = torch.randint(0, 8192, (2, 16))
    targets = torch.randint(0, 8192, (2, 16))

    logits, loss = model(x, targets)
    print(logits.shape)  # torch.Size([2, 16, 8192])
    print(f"loss: {loss.item():.3f} (expected ~{math.log(8192):.3f} at init)")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

