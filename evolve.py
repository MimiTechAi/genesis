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
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from genome import Genome
import prepare

# Configuration
POPULATION_SIZE = 20
SURVIVORS = 5
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.5
TIME_BUDGET = 60 # seconds per genome training
MAX_PARAMS = 10_000_000
MAX_GENERATIONS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

console = Console()

def train_genome(genome: Genome, tokenizer, train_loader, device):
    model = genome.build().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    
    if param_count > MAX_PARAMS:
        return float('inf'), param_count

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    
    model.train()
    start_time = time.time()
    steps = 0
    
    try:
        while time.time() - start_time < TIME_BUDGET:
            x, y, epoch = next(train_loader)
            
            # Simple Cosine Decay based on steps (we don't know total steps, so we use a large constant or time-based)
            # For 60s training, let's just use a fixed small LR or a very simple schedule
            lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * (time.time() - start_time) / TIME_BUDGET))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                loss = model(x, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            steps += 1
            
        # Evaluation
        model.eval()
        val_bpb = prepare.evaluate_bpb(model, tokenizer, BATCH_SIZE)
        return val_bpb, param_count
        
    except Exception as e:
        console.print(f"[red]Error training genome: {e}[/red]")
        return float('inf'), param_count
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()

def save_checkpoint(generation, population, history):
    # population is list of (genome, fitness, params)
    data = {
        "generation": generation,
        "history": history,
        "population": [{"genome": g.to_dict(), "fitness": f, "params": p} for g, f, p in population]
    }
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Save best genome
    if population:
        best_g, best_f, best_p = min(population, key=lambda x: x[1])
        with open("best_genome.json", "w") as f:
            json.dump(best_g.to_dict(), f, indent=2)

def load_checkpoint():
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            data = json.load(f)
        generation = data["generation"]
        history = data["history"]
        population = [(Genome.from_dict(d["genome"]), d["fitness"], d["params"]) for d in data["population"]]
        return generation, population, history
    return 0, [], []

def update_plot(history):
    if not history: return
    gens = [h["generation"] for h in history]
    best = [h["best_val_bpb"] for h in history]
    avg = [h["avg_val_bpb"] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, label="Best BPB")
    plt.plot(gens, avg, label="Avg BPB")
    plt.xlabel("Generation")
    plt.ylabel("BPB")
    plt.title("Evolution Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("progress.png")
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = prepare.Tokenizer.from_directory()
    train_loader = prepare.make_dataloader(tokenizer, BATCH_SIZE, prepare.MAX_SEQ_LEN, "train")
    
    start_gen, population, history = load_checkpoint()
    
    if not population:
        console.print("[bold blue]Initializing random population...[/bold blue]")
        for _ in range(POPULATION_SIZE):
            population.append((Genome.random(), float('inf'), 0))

    for gen in range(start_gen, MAX_GENERATIONS):
        console.rule(f"[bold green]Generation {gen}[/bold green]")
        
        # Evaluate population
        new_population = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Evaluating Gen {gen}...", total=POPULATION_SIZE)
            
            for i, (genome, fitness, params) in enumerate(population):
                # Only train if not already trained in this generation (if we resumed)
                if fitness == float('inf'):
                    fit, p_count = train_genome(genome, tokenizer, train_loader, device)
                else:
                    fit, p_count = fitness, params
                
                new_population.append((genome, fit, p_count))
                progress.update(task, advance=1, description=f"Gen {gen} | Best: {min([x[1] for x in new_population if x[1] != float('inf')] + [float('inf')]):.4f}")

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
                writer.writerow(["generation", "best_val_bpb", "avg_val_bpb", "best_architecture_summary"])
            
            best_g = new_population[0][0]
            summary = f"{len(best_g.genes)} layers: " + "->".join([g.op for g in best_g.genes])
            writer.writerow([gen, f"{best_fit:.6f}", f"{avg_fit:.6f}", summary])
            
        history.append({
            "generation": gen,
            "best_val_bpb": best_fit,
            "avg_val_bpb": avg_fit
        })
        
        update_plot(history)
        
        # Print Top-5
        table = Table(title=f"Top 5 Genomes - Generation {gen}")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Fitness (BPB)", style="magenta")
        table.add_column("Params", justify="right")
        table.add_column("Architecture Summary")
        
        for i, (g, f, p) in enumerate(new_population[:5]):
            summary = f"{len(g.genes)}L: " + "->".join([gene.op[:4] for gene in g.genes])
            table.add_row(str(i+1), f"{f:.4f}", f"{p/1e6:.2f}M", summary)
        console.print(table)
        
        # Selection & Reproduction
        survivors = new_population[:SURVIVORS]
        next_gen = [copy.deepcopy(s) for s in survivors] # Elitism
        
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
