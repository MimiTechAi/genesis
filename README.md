# genesis

![progress](progress.png)

In the beginning, there was no transformer. No attention mechanism. No skip connections. No architecture papers. There were only random neurons, a dataset, and the pressure to predict the next token. 25 generations later, evolution had built its own architecture from scratch — and it looked nothing like what we expected. — [@MimiTechAI](https://github.com/MimiTechAI), March 2026.

## The idea

Take a pool of random neural architectures — no templates, no human priors — and let evolution discover what works for language modeling. Each architecture trains for 30 seconds on the same task. The best survive, mutate, reproduce. The question: **will evolution reinvent attention? Or will it find something else entirely?**

After 25 generations on a single NVIDIA GB10 GPU, the answer surprised us. Evolution didn't pick attention. It didn't pick convolution, or recurrence, or mixture-of-experts. It built a **deep normalized MLP** — stacked linear layers with heavy layer normalization and residual connections. The simplest possible thing that works.

This repo is deliberately kept minimal (under 600 lines total) and uses the same data pipeline and evaluation metric (`val_bpb`) as [autoresearch](https://github.com/karpathy/autoresearch) and [nanochat](https://github.com/karpathy/nanochat), so results are directly comparable.

## What evolution found

| Generation | Best val\_bpb | Architecture | What happened |
|-----------|-------------|-------------|---------------|
| 0 | 10.481 | `linear→gate→norm→moe→conv` | Random architectures. MoE and gating appear. |
| 6 | 9.791 | `linear→norm→gate→identity→norm→linear` | Norm + Gate pattern emerges. Conv dropped. |
| 14 | 9.727 | `linear→norm→norm→identity→norm→linear³` | Gate dropped. Deep normalization takes over. |
| 24 | **9.484** | `linear→norm→norm→identity→norm→linear⁴` | Final form: 9 layers, 7.4M params. Pure Norm+Linear. |

**Key finding:** At this scale (≤10M parameters, 30s training budget), evolution consistently converges on deep normalized MLPs. Attention, convolution, GRUs, and MoE all appeared in early generations but were eliminated by selection pressure.

## Quick start

Requirements: A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run evolution (let it go overnight for best results)
uv run evolve.py
```

Results are logged to `results.tsv` and the best architecture is saved to `best_genome.json`. A `progress.png` plot is updated after each generation.

## How it works

Three files:

- **prepare.py** — data prep + evaluation utilities (from [nanochat](https://github.com/karpathy/nanochat)). Do not modify. Downloads [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and trains a BPE tokenizer.
- **genome.py** — encodes neural architectures as evolvable "DNA." Each genome is a sequence of genes, where each gene specifies an operation (`linear`, `conv1d`, `attention`, `gru`, `gate`, `norm`, `identity`, `moe_router`), dimensions, activation function, and whether to use a skip connection. Supports mutation (add/remove/modify genes) and crossover (combine two parent genomes).
- **evolve.py** — the evolution engine. For each generation: build each genome into a PyTorch model → train for a fixed time budget → evaluate `val_bpb` → select survivors → produce offspring via mutation and crossover → repeat.

The fitness metric is **val\_bpb** (validation bits per byte) — lower is better, and vocabulary-size-independent so architectural changes are fairly compared.

## Configuration

Edit the constants at the top of `evolve.py`:

```python
POPULATION_SIZE = 12       # genomes per generation
SURVIVORS = 3              # top-k survive (elitism)
MUTATION_RATE = 0.3        # probability of mutation per gene
CROSSOVER_RATE = 0.5       # probability of crossover vs mutation
TIME_BUDGET = 30           # seconds of training per genome
MAX_PARAMS = 10_000_000    # parameter limit
MAX_GENERATIONS = 25       # stop after N generations
```

With these defaults, one generation takes ~6 minutes, so 25 generations complete in about 2.5 hours. For overnight runs, set `MAX_GENERATIONS = 200` and `TIME_BUDGET = 60`.

## The building blocks

Evolution can combine these operations in any order, depth, and dimension:

| Operation | What it does | Did evolution use it? |
|-----------|-------------|----------------------|
| `linear` | Dense layer (matrix multiply) | ✅ Yes — dominant in final architecture |
| `norm` | Layer normalization | ✅ Yes — heavily used (3 norm layers) |
| `identity` | Skip / residual connection | ✅ Yes — one identity layer for residual path |
| `gate` | Gated linear unit (GLU-style) | ⚠️ Used mid-evolution, then dropped |
| `conv1d` | Causal 1D convolution | ❌ Dropped after generation 5 |
| `attention` | Multi-head causal self-attention | ❌ Never in top architectures |
| `gru` | Gated recurrent unit | ❌ Never in top architectures |
| `moe_router` | Soft mixture-of-experts routing | ❌ Dropped after generation 2 |

## Important caveats

- **Scale matters.** These results are for small models (≤10M params) with short training budgets (30s). At larger scales, the optimal architecture may be very different — attention's advantages grow with sequence length and model size.
- **This is not NAS.** Neural Architecture Search is a mature field with sophisticated methods (see [The Evolved Transformer](https://arxiv.org/abs/1901.11117), [ENAS](https://arxiv.org/abs/1802.03268), [Microsoft NNI](https://github.com/microsoft/nni)). genesis is a minimal educational tool, not a competitor to those systems.
- **Evolution is stochastic.** Different random seeds will produce different results. Run multiple times for robust conclusions.
- **The search space constrains the results.** The 8 available operations define what evolution *can* find. A richer search space might yield different architectures.

## Related work

- [autoresearch](https://github.com/karpathy/autoresearch) — AI agents optimizing training code (hyperparameters, tricks). genesis is complementary: autoresearch optimizes *how* you train, genesis optimizes *what* you train.
- [The Evolved Transformer](https://arxiv.org/abs/1901.11117) (Google Brain, 2019) — evolutionary search seeded with the Transformer architecture. genesis differs by starting from random architectures with no human prior.
- [Mamba](https://github.com/state-spaces/mamba) / [RWKV](https://github.com/BlinkDL/RWKV-LM) — human-designed alternatives to transformers, showing that attention isn't the only viable approach for language modeling.

## License

MIT
