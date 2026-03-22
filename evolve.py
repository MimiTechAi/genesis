"""
genesis — Neural Architecture Evolution

Evolution engine: evaluate → select → mutate → repeat.
Uses the same data pipeline and eval metric (val_bpb) as autoresearch/nanochat.
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
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from genome import Genome
import prepare

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POPULATION_SIZE = 20       # genomes per generation
SURVIVORS = 5              # top-k survive (elitism)
MUTATION_RATE = 0.3        # per-gene mutation probability
CROSSOVER_RATE = 0.5       # crossover vs mutation for offspring
TIME_BUDGET = 60           # seconds of training per genome
MAX_PARAMS = 10_000_000    # parameter ceiling
MAX_GENERATIONS = 200      # stop after N generations
BATCH_SIZE = 8             # per-genome training batch size
LEARNING_RATE = 3e-4       # AdamW learning rate
WEIGHT_DECAY = 0.1         # AdamW weight decay
GRAD_CLIP = 1.0            # gradient clipping norm
WARMUP_FRAC = 0.1          # fraction of time budget for LR warmup
SEED = 42                  # reproducibility

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
console = Console()


def train_and_evaluate(genome: Genome, tokenizer, train_loader, device):
    """Train a genome for TIME_BUDGET seconds, then evaluate with proper BPB."""
    # Build model
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

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    start_time = time.time()
    steps = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            x, y, _ = next(train_loader)
            elapsed_frac = (time.time() - start_time) / TIME_BUDGET

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
                console.print(f"[red]NaN/Inf at step {steps}[/red]")
                del model; del optimizer; torch.cuda.empty_cache()
                return float('inf'), param_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            steps += 1

        # Proper evaluation using prepare.evaluate_bpb (same as autoresearch)
        model.eval()
        val_bpb = prepare.evaluate_bpb(model, tokenizer, BATCH_SIZE)
        console.print(f"[green]  {steps} steps, val_bpb={val_bpb:.4f}, {param_count/1e6:.2f}M[/green]")
        return val_bpb, param_count

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return float('inf'), param_count
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def save_checkpoint(generation, population, history):
    """Save full state for resume capability."""
    data = {
        "generation": generation,
        "seed": SEED,
        "config": {
            "population_size": POPULATION_SIZE,
            "survivors": SURVIVORS,
            "time_budget": TIME_BUDGET,
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
        population = [
            (Genome.from_dict(d["genome"]), d["fitness"], d["params"])
            for d in data["population"]
        ]
        return gen, population, history
    return 0, [], []


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

    # Add baseline line if we have it
    if "baseline_bpb" in history[0]:
        baseline = history[0]["baseline_bpb"]
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


def run_baseline(tokenizer, train_loader, device):
    """Train a standard transformer baseline for comparison."""
    console.rule("[bold yellow]Transformer Baseline[/bold yellow]")
    genome = Genome.transformer_baseline(n_layers=2, dim=256)
    params = genome.count_parameters()
    console.print(f"Baseline: {genome.summary()} ({params/1e6:.2f}M params)")

    if params > MAX_PARAMS:
        # Reduce if too big
        genome = Genome.transformer_baseline(n_layers=1, dim=256)
        params = genome.count_parameters()
        console.print(f"Reduced: {genome.summary()} ({params/1e6:.2f}M params)")

    bpb, _ = train_and_evaluate(genome, tokenizer, train_loader, device)
    console.print(f"[bold yellow]Transformer baseline val_bpb: {bpb:.4f}[/bold yellow]")
    return bpb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = prepare.Tokenizer.from_directory()
    train_loader = prepare.make_dataloader(tokenizer, BATCH_SIZE, prepare.MAX_SEQ_LEN, "train")

    start_gen, population, history = load_checkpoint()

    # Run transformer baseline first
    baseline_bpb = None
    if not history:
        baseline_bpb = run_baseline(tokenizer, train_loader, device)
        console.print(f"\n[bold]Baseline established: {baseline_bpb:.4f} val_bpb[/bold]\n")

    # Initialize random population
    if not population:
        console.print("[bold blue]Initializing random population...[/bold blue]")
        population = [(Genome.random(), float('inf'), 0) for _ in range(POPULATION_SIZE)]

    for gen in range(start_gen, MAX_GENERATIONS):
        console.rule(f"[bold green]Generation {gen}[/bold green]")

        new_population = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task(f"Gen {gen}...", total=len(population))

            for genome, fitness, params in population:
                if fitness == float('inf'):
                    fit, p = train_and_evaluate(genome, tokenizer, train_loader, device)
                else:
                    fit, p = fitness, params
                new_population.append((genome, fit, p))

                valid = [f for _, f, _ in new_population if f != float('inf')]
                best_so_far = min(valid) if valid else float('inf')
                progress.update(task, advance=1,
                    description=f"Gen {gen} | Best: {best_so_far:.4f}")

        # Sort by fitness
        new_population.sort(key=lambda x: x[1])

        # Stats
        valid_fits = [f for _, f, _ in new_population if f != float('inf')]
        best_fit = min(valid_fits) if valid_fits else float('inf')
        avg_fit = sum(valid_fits) / len(valid_fits) if valid_fits else float('inf')

        # Log to TSV
        file_exists = os.path.isfile("results.tsv")
        with open("results.tsv", "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if not file_exists:
                writer.writerow(["generation", "best_val_bpb", "avg_val_bpb", "best_params",
                                "best_architecture", "baseline_bpb"])
            best_g = new_population[0][0]
            writer.writerow([gen, f"{best_fit:.6f}", f"{avg_fit:.6f}",
                           f"{new_population[0][2]}", best_g.summary(),
                           f"{baseline_bpb:.6f}" if baseline_bpb else ""])

        entry = {"generation": gen, "best_val_bpb": best_fit, "avg_val_bpb": avg_fit}
        if baseline_bpb:
            entry["baseline_bpb"] = baseline_bpb
        history.append(entry)

        update_plot(history)

        # Print top 5
        table = Table(title=f"Top 5 — Generation {gen}")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("val_bpb", style="magenta")
        table.add_column("Params", justify="right")
        table.add_column("Architecture")
        for i, (g, f, p) in enumerate(new_population[:5]):
            table.add_row(str(i+1), f"{f:.4f}", f"{p/1e6:.2f}M", g.summary())
        console.print(table)

        if baseline_bpb:
            delta = best_fit - baseline_bpb
            color = "green" if delta < 0 else "red"
            console.print(f"[{color}]vs baseline: {delta:+.4f} ({'BETTER' if delta < 0 else 'WORSE'})[/{color}]")

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
        save_checkpoint(gen + 1, new_population, history)


if __name__ == "__main__":
    main()
