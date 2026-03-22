"""
genesis — Neural Architecture Evolution

Evolution engine: evaluate → select → mutate → repeat.
Uses the same data pipeline and eval metric (val_bpb) as autoresearch/nanochat.

Features:
  - Curriculum training: fast screening at 512 tokens, full eval at 2048
  - Weight inheritance: offspring inherit compatible weights from parents
  - Crash-tolerant: checkpoint after every generation, watchdog auto-restarts
"""
import os
import time
import json
import csv
import random
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from genome import Genome
import prepare

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POPULATION_SIZE = 20       # genomes per generation
SURVIVORS = 5              # top-k survive (elitism)
MUTATION_RATE = 0.3        # per-gene mutation probability
CROSSOVER_RATE = 0.5       # crossover vs mutation for offspring
TIME_BUDGET = 60           # seconds of training per genome (full eval)
SCREEN_TIME = 15           # seconds for curriculum screening phase
MAX_PARAMS = 10_000_000    # parameter ceiling
MAX_GENERATIONS = 200      # stop after N generations
BATCH_SIZE = 8             # per-genome training batch size
LEARNING_RATE = 3e-4       # AdamW learning rate
WEIGHT_DECAY = 0.1         # AdamW weight decay
GRAD_CLIP = 1.0            # gradient clipping norm
WARMUP_FRAC = 0.1          # fraction of time budget for LR warmup
SEED = 42                  # reproducibility

# Curriculum settings
SCREEN_SEQ_LEN = 512       # short sequences for fast screening
SCREEN_TOP_K = 8           # how many pass screening to full training
FULL_SEQ_LEN = 2048        # full sequence length (= prepare.MAX_SEQ_LEN)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
console = Console()


# ---------------------------------------------------------------------------
# Weight inheritance — offspring inherit compatible weights from parents
# ---------------------------------------------------------------------------

def inherit_weights(child_model, parent_model, child_genome, parent_genome):
    """Copy weights from parent to child where architecture matches.
    
    For each layer: if same op type and same dimensions, copy weights.
    This gives offspring a head start instead of training from scratch.
    Like plant cuttings vs. growing from seed.
    """
    inherited = 0
    total = 0
    
    min_layers = min(len(child_genome.genes), len(parent_genome.genes))
    
    for i in range(min_layers):
        cg = child_genome.genes[i]
        pg = parent_genome.genes[i]
        total += 1
        
        # Must match: same op, same dims
        if cg.op != pg.op or cg.in_dim != pg.in_dim or cg.out_dim != pg.out_dim:
            continue
        
        child_layer = child_model.layers[i]
        parent_layer = parent_model.layers[i]
        
        try:
            # Copy all matching parameters in this layer
            for key in child_layer.keys():
                if key in parent_layer:
                    child_mod = child_layer[key]
                    parent_mod = parent_layer[key]
                    
                    child_params = dict(child_mod.named_parameters())
                    parent_params = dict(parent_mod.named_parameters())
                    
                    for name, cp in child_params.items():
                        if name in parent_params and cp.shape == parent_params[name].shape:
                            cp.data.copy_(parent_params[name].data)
            inherited += 1
        except Exception:
            continue  # Skip on any shape mismatch
    
    # Also inherit token + position embeddings (always same shape)
    if child_model.token_embedding.weight.shape == parent_model.token_embedding.weight.shape:
        child_model.token_embedding.weight.data.copy_(parent_model.token_embedding.weight.data)
        child_model.position_embedding.weight.data.copy_(parent_model.position_embedding.weight.data)
    
    # Inherit lm_head if same shape and not weight-tied
    if (child_model.lm_head.weight is not child_model.token_embedding.weight and
        child_model.lm_head.weight.shape == parent_model.lm_head.weight.shape):
        child_model.lm_head.weight.data.copy_(parent_model.lm_head.weight.data)
    
    # Inherit final layer norm
    if child_model.ln_f.weight.shape == parent_model.ln_f.weight.shape:
        child_model.ln_f.weight.data.copy_(parent_model.ln_f.weight.data)
        child_model.ln_f.bias.data.copy_(parent_model.ln_f.bias.data)
    
    return inherited, total


def train_model(model, train_loader, time_budget, param_count):
    """Train a model for time_budget seconds. Returns trained model and step count."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    start_time = time.time()
    steps = 0

    while time.time() - start_time < time_budget:
        x, y, _ = next(train_loader)
        elapsed_frac = (time.time() - start_time) / time_budget

        # Warmup + cosine decay
        if elapsed_frac < WARMUP_FRAC:
            lr = LEARNING_RATE * (elapsed_frac / WARMUP_FRAC)
        else:
            decay_frac = (elapsed_frac - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
            lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * decay_frac))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            loss = model(x, y)

        if torch.isnan(loss) or torch.isinf(loss):
            return None, steps  # Signal: training diverged

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        steps += 1

    return model, steps


def screen_genome(genome, tokenizer, screen_loader, device):
    """Fast screening: short training on short sequences. Returns approximate fitness."""
    try:
        model = genome.build().to(device)
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        return float('inf'), 0

    param_count = sum(p.numel() for p in model.parameters())
    if param_count > MAX_PARAMS:
        console.print(f"[yellow]Skipped: {param_count/1e6:.1f}M > {MAX_PARAMS/1e6:.0f}M[/yellow]")
        del model; torch.cuda.empty_cache()
        return float('inf'), param_count

    try:
        model, steps = train_model(model, screen_loader, SCREEN_TIME, param_count)
        if model is None:
            console.print(f"[red]NaN/Inf during screening[/red]")
            torch.cuda.empty_cache()
            return float('inf'), param_count

        # Quick eval: just use training loss as proxy (not full BPB)
        model.eval()
        with torch.no_grad():
            x, y, _ = next(screen_loader)
            with torch.amp.autocast('cuda'):
                loss = model(x, y).item()

        console.print(f"[dim]  screen: {steps} steps, loss={loss:.4f}, {param_count/1e6:.2f}M[/dim]")
        return loss, param_count
    except Exception as e:
        console.print(f"[red]Screen error: {e}[/red]")
        return float('inf'), param_count
    finally:
        del model
        torch.cuda.empty_cache()


def full_evaluate(genome, tokenizer, train_loader, device, parent_model=None, parent_genome=None):
    """Full training + proper BPB evaluation. Optionally inherits weights from parent."""
    try:
        model = genome.build().to(device)
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        return float('inf'), 0

    param_count = sum(p.numel() for p in model.parameters())
    if param_count > MAX_PARAMS:
        del model; torch.cuda.empty_cache()
        return float('inf'), param_count

    # Weight inheritance from parent
    inherited_info = ""
    if parent_model is not None and parent_genome is not None:
        inherited, total = inherit_weights(model, parent_model, genome, parent_genome)
        inherited_info = f" [inherited {inherited}/{total} layers]"

    try:
        model, steps = train_model(model, train_loader, TIME_BUDGET, param_count)
        if model is None:
            console.print(f"[red]NaN/Inf during full training[/red]")
            torch.cuda.empty_cache()
            return float('inf'), param_count

        model.eval()
        val_bpb = prepare.evaluate_bpb(model, tokenizer, BATCH_SIZE)
        console.print(f"[green]  {steps} steps, val_bpb={val_bpb:.4f}, "
                     f"{param_count/1e6:.2f}M{inherited_info}[/green]")
        return val_bpb, param_count

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return float('inf'), param_count
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Checkpoint & plotting
# ---------------------------------------------------------------------------

def save_checkpoint(generation, population, history, baseline_bpb=None):
    """Save full state for resume capability."""
    data = {
        "generation": generation,
        "seed": SEED,
        "baseline_bpb": baseline_bpb,
        "config": {
            "population_size": POPULATION_SIZE,
            "survivors": SURVIVORS,
            "time_budget": TIME_BUDGET,
            "screen_time": SCREEN_TIME,
            "screen_seq_len": SCREEN_SEQ_LEN,
            "max_params": MAX_PARAMS,
        },
        "history": history,
        "population": [
            {"genome": g.to_dict(), "fitness": f, "params": p}
            for g, f, p in population
        ]
    }
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)

    if population:
        best_g, best_f, best_p = min(population, key=lambda x: x[1])
        with open("best_genome.json", "w") as f:
            json.dump({
                "genome": best_g.to_dict(),
                "val_bpb": best_f,
                "params": best_p,
                "summary": best_g.summary()
            }, f, indent=2)


def load_checkpoint():
    if os.path.exists("results.json"):
        with open("results.json") as f:
            data = json.load(f)
        gen = data["generation"]
        history = data["history"]
        baseline_bpb = data.get("baseline_bpb", None)
        population = [
            (Genome.from_dict(d["genome"]), d["fitness"], d["params"])
            for d in data["population"]
        ]
        return gen, population, history, baseline_bpb
    return 0, [], [], None


def update_plot(history):
    if not history:
        return
    gens = [h["generation"] for h in history]
    best = [h["best_val_bpb"] for h in history]
    avg = [h["avg_val_bpb"] for h in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.plot(gens, best, color='#58a6ff', linewidth=2.5, label='Best val_bpb', marker='o', markersize=3)
    ax.plot(gens, avg, color='#8b949e', linewidth=1.5, label='Population avg', alpha=0.7, linestyle='--')
    ax.fill_between(gens, best, avg, alpha=0.1, color='#58a6ff')

    baseline = None
    for h in history:
        if h.get("baseline_bpb") is not None:
            baseline = h["baseline_bpb"]
            break
    if baseline is not None:
        ax.axhline(y=baseline, color='#f85149', linestyle=':', linewidth=1.5,
                   label=f'Transformer baseline ({baseline:.3f})')

    ax.set_xlabel('Generation', color='#c9d1d9', fontsize=12)
    ax.set_ylabel('val_bpb (lower is better)', color='#c9d1d9', fontsize=12)
    ax.set_title('genesis — Neural Architecture Evolution', color='#e6edf3',
                 fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#8b949e')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=10)

    plt.tight_layout()
    plt.savefig('progress.png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()


def update_status(gen, best_fit, baseline_bpb, start_time, history):
    """Write live status to status.json for monitoring."""
    elapsed = time.time() - start_time
    avg_per_gen = elapsed / max(gen + 1, 1)
    remaining_gens = MAX_GENERATIONS - gen - 1
    eta_hours = (remaining_gens * avg_per_gen) / 3600

    status = {
        "generation": gen,
        "max_generations": MAX_GENERATIONS,
        "best_val_bpb": best_fit,
        "baseline_bpb": baseline_bpb,
        "vs_baseline": best_fit - baseline_bpb if baseline_bpb else None,
        "elapsed_hours": elapsed / 3600,
        "eta_hours": eta_hours,
        "avg_min_per_gen": avg_per_gen / 60,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("status.json", "w") as f:
        json.dump(status, f, indent=2)


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_baseline(tokenizer, full_loader, device):
    """Train a standard transformer baseline for comparison."""
    console.rule("[bold yellow]Transformer Baseline[/bold yellow]")
    genome = Genome.transformer_baseline(n_layers=2, dim=256)
    params = genome.count_parameters()
    console.print(f"Baseline: {genome.summary()} ({params/1e6:.2f}M params)")

    bpb, _ = full_evaluate(genome, tokenizer, full_loader, device)
    console.print(f"[bold yellow]Transformer baseline val_bpb: {bpb:.4f}[/bold yellow]")
    return bpb


def main():
    run_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = prepare.Tokenizer.from_directory()

    # Two dataloaders: short (screening) and full (evaluation)
    screen_loader = prepare.make_dataloader(tokenizer, BATCH_SIZE, SCREEN_SEQ_LEN, "train")
    full_loader = prepare.make_dataloader(tokenizer, BATCH_SIZE, FULL_SEQ_LEN, "train")

    start_gen, population, history, baseline_bpb = load_checkpoint()

    # Run transformer baseline first (or restore from checkpoint)
    if baseline_bpb is None:
        baseline_bpb = run_baseline(tokenizer, full_loader, device)
        console.print(f"\n[bold]Baseline established: {baseline_bpb:.4f} val_bpb[/bold]\n")

    # Initialize random population
    if not population:
        console.print("[bold blue]Initializing random population...[/bold blue]")
        population = [(Genome.random(), float('inf'), 0) for _ in range(POPULATION_SIZE)]

    # Store parent models for weight inheritance (survivors from last gen)
    parent_models = {}  # genome_summary -> (model, genome)

    for gen in range(start_gen, MAX_GENERATIONS):
        console.rule(f"[bold green]Generation {gen}[/bold green]")
        gen_start = time.time()

        # ----- PHASE 1: Curriculum Screening -----
        # Only screen genomes that haven't been evaluated yet (fitness == inf)
        unevaluated = [(i, g, f, p) for i, (g, f, p) in enumerate(population) if f == float('inf')]
        already_eval = [(g, f, p) for g, f, p in population if f != float('inf')]

        if unevaluated:
            console.print(f"[cyan]Phase 1: Screening {len(unevaluated)} new genomes "
                         f"({SCREEN_TIME}s × {SCREEN_SEQ_LEN} tokens)...[/cyan]")

            screen_results = []
            for idx, genome, _, _ in unevaluated:
                score, params = screen_genome(genome, tokenizer, screen_loader, device)
                screen_results.append((genome, score, params))

            # Sort by screening score, take top-K for full evaluation
            screen_results.sort(key=lambda x: x[1])
            promoted = screen_results[:SCREEN_TOP_K]
            skipped = screen_results[SCREEN_TOP_K:]

            console.print(f"[cyan]Phase 2: Full eval for top {len(promoted)} "
                         f"({TIME_BUDGET}s × {FULL_SEQ_LEN} tokens)...[/cyan]")

            # ----- PHASE 2: Full Training + Eval -----
            full_results = []
            for genome, screen_score, params in promoted:
                # Find best matching parent for weight inheritance
                parent_model, parent_genome = None, None
                for pg, (pm, pgen) in parent_models.items():
                    parent_model, parent_genome = pm, pgen
                    break  # Use first available parent (best from last gen)

                bpb, p = full_evaluate(genome, tokenizer, full_loader, device,
                                      parent_model, parent_genome)
                full_results.append((genome, bpb, p))

            # Skipped genomes get their screen score as proxy (penalized)
            for genome, screen_score, params in skipped:
                full_results.append((genome, screen_score * 1.1, params))  # 10% penalty

            new_population = already_eval + full_results
        else:
            new_population = list(population)

        # Sort by fitness
        new_population.sort(key=lambda x: x[1])

        # Stats
        valid_fits = [f for _, f, _ in new_population if f != float('inf')]
        best_fit = min(valid_fits) if valid_fits else float('inf')
        avg_fit = sum(valid_fits) / len(valid_fits) if valid_fits else float('inf')

        gen_time = time.time() - gen_start

        # Log to TSV
        file_exists = os.path.isfile("results.tsv")
        with open("results.tsv", "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if not file_exists:
                writer.writerow(["generation", "best_val_bpb", "avg_val_bpb", "best_params",
                                "best_architecture", "baseline_bpb", "gen_time_min"])
            best_g = new_population[0][0]
            writer.writerow([gen, f"{best_fit:.6f}", f"{avg_fit:.6f}",
                           f"{new_population[0][2]}", best_g.summary(),
                           f"{baseline_bpb:.6f}", f"{gen_time/60:.1f}"])

        entry = {"generation": gen, "best_val_bpb": best_fit, "avg_val_bpb": avg_fit,
                 "baseline_bpb": baseline_bpb}
        history.append(entry)

        update_plot(history)
        update_status(gen, best_fit, baseline_bpb, run_start, history)

        # Print top 5
        table = Table(title=f"Top 5 — Generation {gen} ({gen_time/60:.1f} min)")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("val_bpb", style="magenta")
        table.add_column("Params", justify="right")
        table.add_column("Architecture")
        for i, (g, f, p) in enumerate(new_population[:5]):
            table.add_row(str(i+1), f"{f:.4f}", f"{p/1e6:.2f}M", g.summary())
        console.print(table)

        delta = best_fit - baseline_bpb
        color = "green" if delta < 0 else "red"
        console.print(f"[{color}]vs baseline: {delta:+.4f} "
                     f"({'BETTER' if delta < 0 else 'WORSE'})[/{color}]")

        # Store best survivor's model for weight inheritance in next gen
        parent_models.clear()
        best_genome = new_population[0][0]
        try:
            best_model = best_genome.build().to(device)
            # Quick retrain to get good weights
            temp_loader = prepare.make_dataloader(tokenizer, BATCH_SIZE, FULL_SEQ_LEN, "train")
            best_model, _ = train_model(best_model, temp_loader, TIME_BUDGET // 2, 0)
            if best_model is not None:
                parent_models[best_genome.summary()] = (best_model, best_genome)
        except Exception:
            pass  # No parent available — children train from scratch

        # Selection & reproduction
        survivors = new_population[:SURVIVORS]
        next_gen = [copy.deepcopy(s) for s in survivors]  # elitism

        while len(next_gen) < POPULATION_SIZE:
            if random.random() < CROSSOVER_RATE and len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
                child = p1[0].crossover(p2[0])
            else:
                parent = random.choice(survivors)
                child = parent[0].mutate(rate=MUTATION_RATE)
            next_gen.append((child, float('inf'), 0))

        population = next_gen
        save_checkpoint(gen + 1, new_population, history, baseline_bpb)

        # Clean up parent models to free GPU memory
        for key in list(parent_models.keys()):
            del parent_models[key]
        parent_models.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
