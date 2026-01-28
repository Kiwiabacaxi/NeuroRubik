#!/usr/bin/env python3
"""
Rubik's Cube Neural Network Trainer - GPU Enhanced

Train a neural network to solve Rubik's cubes using genetic algorithm
with GPU acceleration and curriculum learning.

Usage:
    python train.py --population 100 --generations 500
    python train.py --load checkpoint.json --generations 100
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cube.cube_state import MOVES, CubeState
from genetic.evolution import GeneticAlgorithm
from genetic.fitness import BatchFitnessEvaluator
from genetic.individual import Individual
from neural.network import CubeSolverNetwork, get_device_info, DEVICE
from neural.weight_export import export_training_stats, export_weights_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network to solve Rubik's cube using genetic algorithm"
    )

    # Training parameters
    parser.add_argument(
        "--population", type=int, default=100, help="Population size (default: 100)"
    )
    parser.add_argument(
        "--generations", type=int, default=500, help="Number of generations (default: 500)"
    )
    parser.add_argument(
        "--scramble-depth",
        type=int,
        default=1,
        help="Initial scramble depth for curriculum (default: 1)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=20, help="Maximum scramble depth (default: 20)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max moves to solve (default: 100)"
    )
    parser.add_argument(
        "--test-cubes",
        type=int,
        default=20,
        help="Number of test cubes per evaluation (default: 20)",
    )

    # Genetic algorithm parameters
    parser.add_argument(
        "--mutation-rate", type=float, default=0.15, help="Mutation rate (default: 0.15)"
    )
    parser.add_argument(
        "--mutation-strength", type=float, default=0.2, help="Mutation strength (default: 0.2)"
    )
    parser.add_argument(
        "--crossover-rate", type=float, default=0.7, help="Crossover rate (default: 0.7)"
    )
    parser.add_argument(
        "--elitism", type=int, default=10, help="Number of elite individuals (default: 10)"
    )

    # Network architecture
    parser.add_argument(
        "--hidden",
        type=str,
        default="512,512,256,128",
        help="Hidden layer sizes comma-separated (default: 512,512,256,128)",
    )

    # I/O
    parser.add_argument(
        "--output", type=str, default="weights", help="Output directory (default: weights)"
    )
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint from file")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N generations (default: 10)",
    )

    # Curriculum learning
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.6,
        help="Solve rate to increase difficulty (default: 0.6)",
    )

    # Other
    parser.add_argument(
        "--target-fitness", type=float, default=None, help="Stop when this fitness is reached"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    return parser.parse_args()


def create_output_dir(output_dir: str) -> str:
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_checkpoint(
    ga: GeneticAlgorithm,
    network: CubeSolverNetwork,
    output_dir: str,
    evaluator: BatchFitnessEvaluator,
    hidden_sizes: tuple,
):
    """Save current training checkpoint."""

    # Save best weights to JSON (for frontend)
    network.set_weights_flat(ga.best_ever.genome)
    export_weights_to_json(network, os.path.join(output_dir, "best_weights.json"))

    # Save training stats
    stats = ga.get_statistics()
    curriculum_status = evaluator.get_status()
    stats["evaluator"] = {
        "scramble_depth": curriculum_status["current_depth"],
        "max_steps": evaluator.evaluator.max_steps,
        "test_cubes": evaluator.evaluator.num_test_cubes,
    }
    stats["curriculum"] = curriculum_status
    export_training_stats(stats, os.path.join(output_dir, "training_stats.json"))

    # Save full checkpoint for resuming
    checkpoint = {
        "generation": ga.generation,
        "best_genome": ga.best_ever.genome.tolist(),
        "best_fitness": ga.best_ever.fitness,
        "config": {
            "population_size": ga.population_size,
            "mutation_rate": ga.mutation_rate,
            "mutation_strength": ga.mutation_strength,
            "crossover_rate": ga.crossover_rate,
            "elitism_count": ga.elitism_count,
            "hidden_sizes": list(hidden_sizes),
        },
        "curriculum": curriculum_status,
        "history": {k: v for k, v in ga.history.items()},
    }

    with open(os.path.join(output_dir, "checkpoint.json"), "w") as f:
        json.dump(checkpoint, f)


def test_best_network(
    network: CubeSolverNetwork, scramble_depth: int = 5, num_tests: int = 5, verbose: bool = True
):
    """Test the best network on some cubes."""
    if verbose:
        print("\n" + "=" * 50)
        print("Testing best network")
        print("=" * 50)

    solved = 0

    for i in range(num_tests):
        cube = CubeState()
        scramble = cube.scramble(scramble_depth)

        if verbose:
            print(f"\nTest {i + 1}:")
            print(f"  Scramble: {' '.join(scramble)}")

        moves_made = []
        visited = set()

        for step in range(100):
            if cube.is_solved():
                solved += 1
                if verbose:
                    print(f"  ✓ Solved in {step} moves: {' '.join(moves_made)}")
                break

            state_hash = cube.state.tobytes()
            if state_hash in visited:
                if verbose:
                    print(f"  ✗ Stuck in loop after {step} moves")
                break
            visited.add(state_hash)

            state = cube.to_one_hot()
            action = network.predict_action(state, deterministic=True)
            move = MOVES[action]
            cube.apply_move(move)
            moves_made.append(move)
        else:
            if verbose:
                correct = cube.count_correct_stickers()
                print(f"  ✗ Not solved (correct stickers: {correct}/54)")

    if verbose:
        print(f"\n{'='*50}")
        print(f"Results: Solved {solved}/{num_tests} cubes ({100*solved/num_tests:.0f}%)")
        print(f"{'='*50}")

    return solved


def main():
    args = parse_args()

    # Parse hidden layer sizes
    hidden_sizes = tuple(int(x) for x in args.hidden.split(","))

    # Print device info
    device_info = get_device_info()
    print("\n" + "=" * 60)
    print("🧊 RUBIK'S CUBE NEURAL SOLVER - GPU TRAINING")
    print("=" * 60)
    print(f"\n🖥️  Device: {device_info['device']}")
    if device_info.get("cuda_available"):
        print(f"   GPU: {device_info['cuda_device_name']}")
        print(f"   Memory: {device_info['cuda_memory_gb']:.1f} GB")
    else:
        print("   ⚠️  CUDA not available - using CPU")

    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"\n📁 Output: {output_dir}")

    # Create network to get genome size
    network = CubeSolverNetwork(hidden_sizes=hidden_sizes, device=DEVICE)
    genome_size = network.get_weight_count()

    print(f"\n🧠 Network Architecture:")
    print(f"   Input: 324 (54 × 6 one-hot)")
    print(f"   Hidden: {' → '.join(str(h) for h in hidden_sizes)}")
    print(f"   Output: 18 (moves)")
    print(f"   Total weights: {genome_size:,}")

    # Create genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population,
        genome_size=genome_size,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        elitism_count=args.elitism,
    )

    # Load checkpoint if provided
    if args.load:
        print(f"\n📂 Loading checkpoint from {args.load}")
        with open(args.load) as f:
            checkpoint = json.load(f)

        # Create best individual from checkpoint
        best_genome = np.array(checkpoint["best_genome"], dtype=np.float32)
        ga.best_ever = Individual(genome=best_genome)
        ga.best_ever.fitness = checkpoint["best_fitness"]
        ga.generation = checkpoint["generation"]

        print(f"   Resumed from generation {ga.generation}")
        print(f"   Best fitness: {ga.best_ever.fitness:.2f}")

        # Restore curriculum state if available
        initial_depth = checkpoint.get("curriculum", {}).get("current_depth", args.scramble_depth)
    else:
        ga.initialize_population()
        initial_depth = args.scramble_depth

    # Create fitness evaluator with curriculum learning
    evaluator = BatchFitnessEvaluator(
        num_test_cubes=args.test_cubes,
        scramble_depth=initial_depth,
        max_steps=args.max_steps,
        hidden_sizes=hidden_sizes,
        device=DEVICE,
    )

    # Set curriculum threshold
    evaluator.evaluator.depth_increase_threshold = args.curriculum_threshold

    print(f"\n⚙️  Training Configuration:")
    print(f"   Population: {args.population}")
    print(f"   Generations: {args.generations}")
    print(
        f"   Curriculum: {'Disabled' if args.no_curriculum else f'{initial_depth} → {args.max_depth} moves'}"
    )
    print(f"   Max steps: {args.max_steps}")
    print(f"   Test cubes: {args.test_cubes}")
    print(f"   Mutation: {args.mutation_rate} rate, {args.mutation_strength} strength")
    print(f"   Elitism: {args.elitism}")

    print("\n🚀 Starting training...\n")

    # Run evolution
    try:
        for gen in range(args.generations):
            # Regenerate test cubes every 5 generations to prevent overfitting
            if gen > 0 and gen % 5 == 0:
                evaluator.regenerate_test_cubes()

            ga.evolve_generation(evaluator)

            best = ga.population[0]
            avg = np.mean([ind.fitness for ind in ga.population])
            solve_rate = best.solved_count / args.test_cubes

            # Update curriculum (if not disabled)
            depth_changed = False
            if not args.no_curriculum:
                depth_changed = evaluator.update_difficulty(solve_rate)

            curriculum = evaluator.get_status()

            if not args.quiet:
                status_emoji = "🎉" if best.solved_count > 0 else "📈" if depth_changed else "🔄"
                depth_str = f"D{curriculum['current_depth']:2d}"
                print(
                    f"{status_emoji} Gen {ga.generation:4d} | {depth_str} | "
                    f"Best: {best.fitness:7.2f} | "
                    f"Solved: {best.solved_count:2d}/{args.test_cubes} ({100*solve_rate:3.0f}%) | "
                    f"Avg: {avg:6.2f}"
                )

                if depth_changed:
                    print(f"   ⬆️  Difficulty increased to depth {curriculum['current_depth']}")

            # Save checkpoint periodically
            if ga.generation % args.save_every == 0:
                save_checkpoint(ga, network, output_dir, evaluator, hidden_sizes)

            # Early stopping
            if args.target_fitness and ga.best_ever.fitness >= args.target_fitness:
                print(f"\n🎯 Target fitness {args.target_fitness} reached!")
                break

            # Check if we've mastered the max depth
            if curriculum["current_depth"] >= args.max_depth and solve_rate >= 0.9:
                print(
                    f"\n🏆 Mastered depth {args.max_depth} with {solve_rate*100:.0f}% solve rate!"
                )
                break

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    # Final save
    save_checkpoint(ga, network, output_dir, evaluator, hidden_sizes)

    curriculum = evaluator.get_status()

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Best fitness: {ga.best_ever.fitness:.2f}")
    print(f"   Generations: {ga.generation}")
    print(f"   Final depth: {curriculum['current_depth']}")
    print(f"   Weights: {output_dir}/best_weights.json")

    # Test the best network at various depths
    network.set_weights_flat(ga.best_ever.genome)

    print("\n📊 Testing at different scramble depths:")
    for test_depth in [1, 3, 5, 10, curriculum["current_depth"]]:
        if test_depth > curriculum["current_depth"] + 5:
            break
        solved = test_best_network(network, test_depth, num_tests=5, verbose=False)
        print(f"   Depth {test_depth:2d}: {solved}/5 solved ({100*solved/5:.0f}%)")


if __name__ == "__main__":
    main()
