import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
import random
import copy
import math
from typing import List, Optional


@dataclass
class Gene:
    """One computational block in the architecture."""
    op: str          # 'linear', 'conv1d', 'attention', 'gru', 'identity', 'gate', 'norm', 'moe_router'
    in_dim: int
    out_dim: int
    activation: str  # 'relu', 'gelu', 'silu', 'tanh', 'none'
    skip: bool       # residual/skip connection


# ---------------------------------------------------------------------------
# Operations — the building blocks evolution can combine
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal 1D convolution — has implicit positional information via kernel."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with SDPA."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # Ensure divisibility
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Use PyTorch SDPA (handles causal mask efficiently)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class GRUBlock(nn.Module):
    """GRU wrapper — has implicit positional information via recurrence."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class GateBlock(nn.Module):
    """Gated Linear Unit — element-wise gating."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x) * torch.sigmoid(self.gate(x))


class MoERouter(nn.Module):
    """Soft Mixture-of-Experts — weighted sum of expert outputs."""
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)  # (B, T, E)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + probs[:, :, i:i+1] * expert(x)
        return out


# ---------------------------------------------------------------------------
# EvolvedModel — builds a PyTorch model from a genome
# ---------------------------------------------------------------------------

class EvolvedModel(nn.Module):
    def __init__(self, genome: 'Genome'):
        super().__init__()
        self.genome = genome

        # Token + Position embeddings (CRITICAL: without position info, attention is useless)
        self.token_embedding = nn.Embedding(genome.vocab_size, genome.embedding_dim)
        self.position_embedding = nn.Embedding(genome.sequence_len, genome.embedding_dim)

        layers = []
        current_dim = genome.embedding_dim

        for gene in genome.genes:
            layer_ops = nn.ModuleDict()

            # Dimension projection if needed
            if gene.in_dim != current_dim:
                layer_ops['pre_proj'] = nn.Linear(current_dim, gene.in_dim, bias=False)

            # Build operation
            if gene.op == 'linear':
                layer_ops['op'] = nn.Linear(gene.in_dim, gene.out_dim)
            elif gene.op == 'conv1d':
                layer_ops['op'] = CausalConv1d(gene.in_dim, gene.out_dim)
            elif gene.op == 'attention':
                if gene.in_dim != gene.out_dim:
                    layer_ops['op'] = nn.Sequential(
                        CausalSelfAttention(gene.in_dim),
                        nn.Linear(gene.in_dim, gene.out_dim, bias=False)
                    )
                else:
                    layer_ops['op'] = CausalSelfAttention(gene.in_dim)
            elif gene.op == 'gru':
                layer_ops['op'] = GRUBlock(gene.in_dim, gene.out_dim)
            elif gene.op == 'identity':
                if gene.in_dim != gene.out_dim:
                    layer_ops['op'] = nn.Linear(gene.in_dim, gene.out_dim, bias=False)
                else:
                    layer_ops['op'] = nn.Identity()
            elif gene.op == 'gate':
                layer_ops['op'] = GateBlock(gene.in_dim, gene.out_dim)
            elif gene.op == 'norm':
                if gene.in_dim != gene.out_dim:
                    layer_ops['op'] = nn.Sequential(
                        nn.LayerNorm(gene.in_dim),
                        nn.Linear(gene.in_dim, gene.out_dim, bias=False)
                    )
                else:
                    layer_ops['op'] = nn.LayerNorm(gene.in_dim)
            elif gene.op == 'moe_router':
                if gene.in_dim != gene.out_dim:
                    layer_ops['op'] = nn.Sequential(
                        MoERouter(gene.in_dim),
                        nn.Linear(gene.in_dim, gene.out_dim, bias=False)
                    )
                else:
                    layer_ops['op'] = MoERouter(gene.in_dim)

            # Activation
            if gene.activation == 'relu':
                layer_ops['act'] = nn.ReLU()
            elif gene.activation == 'gelu':
                layer_ops['act'] = nn.GELU()
            elif gene.activation == 'silu':
                layer_ops['act'] = nn.SiLU()
            elif gene.activation == 'tanh':
                layer_ops['act'] = nn.Tanh()

            # Skip projection if dims don't match
            if gene.skip and current_dim != gene.out_dim:
                layer_ops['skip_proj'] = nn.Linear(current_dim, gene.out_dim, bias=False)

            layers.append(layer_ops)
            current_dim = gene.out_dim

        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(current_dim)
        self.lm_head = nn.Linear(current_dim, genome.vocab_size, bias=False)

        # Weight tying if dims match
        if current_dim == genome.embedding_dim:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        n_embd = self.genome.embedding_dim
        std = 1.0 / math.sqrt(n_embd)

        # Embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # LM head (small init for stable start)
        if not (self.genome.embedding_dim == self.layers[-1]
                if not self.layers else True):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # All linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.genome.sequence_len, f"Sequence length {T} > max {self.genome.sequence_len}"

        # Token + Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)

        for i, layer in enumerate(self.layers):
            identity = x

            h = x
            if 'pre_proj' in layer:
                h = layer['pre_proj'](h)

            h = layer['op'](h)

            if 'act' in layer:
                h = layer['act'](h)

            if self.genome.genes[i].skip:
                if 'skip_proj' in layer:
                    identity = layer['skip_proj'](identity)
                x = identity + h
            else:
                x = h

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=reduction
            )
            return loss

        return logits


# ---------------------------------------------------------------------------
# Genome — architecture encoded as evolvable DNA
# ---------------------------------------------------------------------------

OPS = ['linear', 'conv1d', 'attention', 'gru', 'identity', 'gate', 'norm', 'moe_router']
ACTIVATIONS = ['relu', 'gelu', 'silu', 'tanh', 'none']
DIMS = [64, 128, 256, 512]  # all divisible by common head counts


@dataclass
class Genome:
    genes: List[Gene]
    embedding_dim: int
    vocab_size: int = 8192
    sequence_len: int = 2048

    def build(self) -> nn.Module:
        return EvolvedModel(self)

    def mutate(self, rate=0.3) -> 'Genome':
        new_genes = copy.deepcopy(self.genes)

        n_mutations = max(1, int(len(new_genes) * rate))
        for _ in range(n_mutations):
            m_type = random.choice([
                'change_op', 'change_activation', 'add_gene',
                'remove_gene', 'change_dims', 'toggle_skip', 'swap_genes'
            ])

            if m_type == 'change_op' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].op = random.choice(OPS)

            elif m_type == 'change_activation' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].activation = random.choice(ACTIVATIONS)

            elif m_type == 'add_gene' and len(new_genes) < 12:  # cap depth
                idx = random.randint(0, len(new_genes))
                prev_out = new_genes[idx-1].out_dim if idx > 0 else self.embedding_dim
                new_dim = random.choice(DIMS)
                new_gene = Gene(
                    random.choice(OPS), prev_out, new_dim,
                    random.choice(ACTIVATIONS), random.random() < 0.5
                )
                new_genes.insert(idx, new_gene)

            elif m_type == 'remove_gene' and len(new_genes) > 1:
                idx = random.randrange(len(new_genes))
                new_genes.pop(idx)

            elif m_type == 'change_dims' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].out_dim = random.choice(DIMS)

            elif m_type == 'toggle_skip' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].skip = not new_genes[idx].skip

            elif m_type == 'swap_genes' and len(new_genes) > 1:
                i, j = random.sample(range(len(new_genes)), 2)
                new_genes[i], new_genes[j] = new_genes[j], new_genes[i]

        # Fix dimension chain consistency
        curr_in = self.embedding_dim
        for g in new_genes:
            g.in_dim = curr_in
            curr_in = g.out_dim

        return Genome(new_genes, self.embedding_dim, self.vocab_size, self.sequence_len)

    def crossover(self, other: 'Genome') -> 'Genome':
        """Uniform crossover: each gene randomly from parent 1 or 2."""
        max_len = max(len(self.genes), len(other.genes))
        child_genes = []
        curr_in = self.embedding_dim

        for i in range(max_len):
            g1 = self.genes[i] if i < len(self.genes) else None
            g2 = other.genes[i] if i < len(other.genes) else None

            if g1 and g2:
                parent_gene = copy.deepcopy(random.choice([g1, g2]))
            elif g1:
                parent_gene = copy.deepcopy(g1)
            else:
                parent_gene = copy.deepcopy(g2)

            parent_gene.in_dim = curr_in
            child_genes.append(parent_gene)
            curr_in = parent_gene.out_dim

        return Genome(child_genes, self.embedding_dim, self.vocab_size, self.sequence_len)

    @staticmethod
    def random(max_depth=8, embedding_dim=256) -> 'Genome':
        depth = random.randint(2, max_depth)
        genes = []
        curr_dim = embedding_dim
        for _ in range(depth):
            op = random.choice(OPS)
            out_dim = random.choice(DIMS)
            genes.append(Gene(
                op=op, in_dim=curr_dim, out_dim=out_dim,
                activation=random.choice(ACTIVATIONS),
                skip=random.random() < 0.5
            ))
            curr_dim = out_dim

        return Genome(genes, embedding_dim)

    @staticmethod
    def transformer_baseline(n_layers=4, dim=256) -> 'Genome':
        """Create a standard transformer-like genome for baseline comparison."""
        genes = []
        curr_dim = dim
        for _ in range(n_layers):
            # Attention + MLP block (like a standard transformer layer)
            genes.append(Gene('attention', curr_dim, curr_dim, 'none', True))  # attention + skip
            genes.append(Gene('norm', curr_dim, curr_dim, 'none', False))      # post-attn norm
            genes.append(Gene('linear', curr_dim, curr_dim * 2, 'gelu', False))  # MLP up
            genes.append(Gene('linear', curr_dim * 2, curr_dim, 'none', True))   # MLP down + skip
            genes.append(Gene('norm', curr_dim, curr_dim, 'none', False))        # post-MLP norm
        return Genome(genes, dim)

    def count_parameters(self) -> int:
        model = self.build()
        return sum(p.numel() for p in model.parameters())

    def summary(self) -> str:
        ops = [g.op for g in self.genes]
        return f"{len(self.genes)}L: {'→'.join(ops)}"

    def to_dict(self) -> dict:
        return {
            "genes": [asdict(g) for g in self.genes],
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "sequence_len": self.sequence_len
        }

    @staticmethod
    def from_dict(d: dict) -> 'Genome':
        genes = [Gene(**g) for g in d['genes']]
        return Genome(
            genes=genes,
            embedding_dim=d['embedding_dim'],
            vocab_size=d.get('vocab_size', 8192),
            sequence_len=d.get('sequence_len', 2048)
        )
