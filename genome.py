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
    op: str          # 'linear', 'conv1d', 'attention', 'gru', 'identity', 'gate', 'norm', 'moe_router'
    in_dim: int
    out_dim: int
    activation: str  # 'relu', 'gelu', 'silu', 'tanh', 'none'
    skip: bool       # residual/skip connection

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # Fallback to 1 head if dim not divisible
        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MoERouter(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim)) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.size()
        logits = self.router(x) # (B, T, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Simple weighted sum MoE for differentiability and simplicity
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += probs[:, :, i:i+1] * expert(x)
        return out

class EvolvedModel(nn.Module):
    def __init__(self, genome: 'Genome'):
        super().__init__()
        self.genome = genome
        self.token_embedding = nn.Embedding(genome.vocab_size, genome.embedding_dim)
        
        layers = []
        current_dim = genome.embedding_dim
        
        for gene in genome.genes:
            layer_ops = nn.ModuleDict()
            
            # Dimension matching projection if needed for the op itself
            # But the gene specifies in_dim and out_dim
            if gene.in_dim != current_dim:
                layer_ops['pre_proj'] = nn.Linear(current_dim, gene.in_dim)
            
            # The core operation
            op_module = None
            if gene.op == 'linear':
                op_module = nn.Linear(gene.in_dim, gene.out_dim)
            elif gene.op == 'conv1d':
                op_module = CausalConv1d(gene.in_dim, gene.out_dim)
            elif gene.op == 'attention':
                # Attention usually keeps dimension. If in != out, we wrap it.
                if gene.in_dim != gene.out_dim:
                    op_module = nn.Sequential(CausalSelfAttention(gene.in_dim), nn.Linear(gene.in_dim, gene.out_dim))
                else:
                    op_module = CausalSelfAttention(gene.in_dim)
            elif gene.op == 'gru':
                class GRUWrapper(nn.Module):
                    def __init__(self, idim, odim):
                        super().__init__()
                        self.gru = nn.GRU(idim, odim, batch_first=True)
                    def forward(self, x):
                        out, _ = self.gru(x)
                        return out
                op_module = GRUWrapper(gene.in_dim, gene.out_dim)
            elif gene.op == 'identity':
                if gene.in_dim != gene.out_dim:
                    op_module = nn.Linear(gene.in_dim, gene.out_dim)
                else:
                    op_module = nn.Identity()
            elif gene.op == 'gate':
                class Gate(nn.Module):
                    def __init__(self, idim, odim):
                        super().__init__()
                        self.proj = nn.Linear(idim, odim)
                        self.gate = nn.Linear(idim, odim)
                    def forward(self, x):
                        return self.proj(x) * torch.sigmoid(self.gate(x))
                op_module = Gate(gene.in_dim, gene.out_dim)
            elif gene.op == 'norm':
                if gene.in_dim != gene.out_dim:
                    op_module = nn.Sequential(nn.LayerNorm(gene.in_dim), nn.Linear(gene.in_dim, gene.out_dim))
                else:
                    op_module = nn.LayerNorm(gene.in_dim)
            elif gene.op == 'moe_router':
                if gene.in_dim != gene.out_dim:
                    op_module = nn.Sequential(MoERouter(gene.in_dim), nn.Linear(gene.in_dim, gene.out_dim))
                else:
                    op_module = MoERouter(gene.in_dim)
            
            layer_ops['op'] = op_module
            
            # Activation
            if gene.activation == 'relu': layer_ops['act'] = nn.ReLU()
            elif gene.activation == 'gelu': layer_ops['act'] = nn.GELU()
            elif gene.activation == 'silu': layer_ops['act'] = nn.SiLU()
            elif gene.activation == 'tanh': layer_ops['act'] = nn.Tanh()
            
            # Skip connection projection if dims don't match
            if gene.skip and current_dim != gene.out_dim:
                layer_ops['skip_proj'] = nn.Linear(current_dim, gene.out_dim)
                
            layers.append(layer_ops)
            current_dim = gene.out_dim
            
        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(current_dim)
        self.lm_head = nn.Linear(current_dim, genome.vocab_size, bias=False)
        
        # Weight sharing between embedding and lm_head
        if current_dim == genome.embedding_dim:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        x = self.token_embedding(idx)
        
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
            return loss
        
        return logits

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
        new_embedding_dim = self.embedding_dim
        
        if random.random() < rate:
            # Randomly change embedding dim occasionally? User didn't specify, but let's keep it mostly stable.
            pass

        ops = ['linear', 'conv1d', 'attention', 'gru', 'identity', 'gate', 'norm', 'moe_router']
        acts = ['relu', 'gelu', 'silu', 'tanh', 'none']
        dims = [64, 128, 256, 512]
        
        # Apply mutations
        for _ in range(max(1, int(len(new_genes) * rate))):
            m_type = random.choice(['change_op', 'change_activation', 'add_gene', 'remove_gene', 'change_dims', 'toggle_skip', 'swap_genes'])
            
            if m_type == 'change_op' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].op = random.choice(ops)
            elif m_type == 'change_activation' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].activation = random.choice(acts)
            elif m_type == 'add_gene':
                idx = random.randint(0, len(new_genes))
                prev_out = new_genes[idx-1].out_dim if idx > 0 else new_embedding_dim
                next_in = new_genes[idx].in_dim if idx < len(new_genes) else new_embedding_dim
                # For simplicity, keep dimension transition smooth
                new_dim = random.choice(dims)
                new_gene = Gene(random.choice(ops), prev_out, new_dim, random.choice(acts), random.random() < 0.5)
                new_genes.insert(idx, new_gene)
                if idx + 1 < len(new_genes):
                    new_genes[idx+1].in_dim = new_dim
            elif m_type == 'remove_gene' and len(new_genes) > 1:
                idx = random.randrange(len(new_genes))
                removed = new_genes.pop(idx)
                if idx < len(new_genes):
                    new_genes[idx].in_dim = new_genes[idx-1].out_dim if idx > 0 else new_embedding_dim
            elif m_type == 'change_dims' and new_genes:
                idx = random.randrange(len(new_genes))
                new_dim = random.choice(dims)
                new_genes[idx].out_dim = new_dim
                if idx + 1 < len(new_genes):
                    new_genes[idx+1].in_dim = new_dim
            elif m_type == 'toggle_skip' and new_genes:
                idx = random.randrange(len(new_genes))
                new_genes[idx].skip = not new_genes[idx].skip
            elif m_type == 'swap_genes' and len(new_genes) > 1:
                idx1, idx2 = random.sample(range(len(new_genes)), 2)
                new_genes[idx1], new_genes[idx2] = new_genes[idx2], new_genes[idx1]
                # Fix dimensions after swap
                for i in range(len(new_genes)):
                    new_genes[i].in_dim = new_genes[i-1].out_dim if i > 0 else new_embedding_dim

        # Final consistency check
        curr_in = new_embedding_dim
        for g in new_genes:
            g.in_dim = curr_in
            curr_in = g.out_dim

        return Genome(new_genes, new_embedding_dim, self.vocab_size, self.sequence_len)
    
    def crossover(self, other: 'Genome') -> 'Genome':
        # Uniform Crossover
        max_len = max(len(self.genes), len(other.genes))
        child_genes = []
        curr_in = self.embedding_dim # Assume both have same embedding_dim for now
        
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
        ops = ['linear', 'conv1d', 'attention', 'gru', 'identity', 'gate', 'norm', 'moe_router']
        acts = ['relu', 'gelu', 'silu', 'tanh', 'none']
        dims = [64, 128, 256, 512]  # all divisible by 4 for attention heads
        
        depth = random.randint(2, max_depth)
        genes = []
        curr_dim = embedding_dim
        for i in range(depth):
            op = random.choice(ops)
            out_dim = random.choice(dims)
            # Ensure attention dims are divisible by num_heads (4)
            if op == 'attention':
                while curr_dim % 4 != 0:
                    curr_dim = random.choice(dims)
                while out_dim % 4 != 0:
                    out_dim = random.choice(dims)
            genes.append(Gene(
                op=op,
                in_dim=curr_dim,
                out_dim=out_dim,
                activation=random.choice(acts),
                skip=random.random() < 0.5
            ))
            curr_dim = out_dim
            
        return Genome(genes, embedding_dim)
    
    def count_parameters(self) -> int:
        model = self.build()
        return sum(p.numel() for p in model.parameters())
    
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
