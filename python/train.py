#!/usr/bin/env python3
"""
Rubik's Cube Neural Network Trainer

Train a neural network to solve Rubik's cubes using genetic algorithm.

Usage:
    python train.py --population 50 --generations 100 --scramble-depth 5
    python train.py --load checkpoint.json --generations 50
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cube.cube_state import MOVES, CubeState
from genetic.evolution import GeneticAlgorithm
from genetic.fitness import FitnessEvaluator
from genetic.individual import Individual
from neural.network import CubeSolverNetwork
from neural.weight_export import export_training_stats, export_weights_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network to solve Rubik's cube using genetic algorithm"
    )

    # Training parameters
    parser.add_argument("--population", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations (default: 100)"
    )
    parser.add_argument(
        "--scramble-depth", type=int, default=5, help="Number of moves to scramble (default: 5)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30, help="Max moves to solve (default: 30)"
    )
    parser.add_argument(
        "--test-cubes",
        type=int,
        default=10,
        help="Number of test cubes per evaluation (default: 10)",
    )

    # Genetic algorithm parameters
    parser.add_argument(
        "--mutation-rate", type=float, default=0.1, help="Mutation rate (default: 0.1)"
    )
    parser.add_argument(
        "--mutation-strength", type=float, default=0.3, help="Mutation strength (default: 0.3)"
    )
    parser.add_argument(
        "--crossover-rate", type=float, default=0.7, help="Crossover rate (default: 0.7)"
    )
    parser.add_argument(
        "--elitism", type=int, default=5, help="Number of elite individuals (default: 5)"
    )

    # Network architecture
    parser.add_argument(
        "--hidden1", type=int, default=256, help="First hidden layer size (default: 256)"
    )
    parser.add_argument(
        "--hidden2", type=int, default=128, help="Second hidden layer size (default: 128)"
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
    ga: GeneticAlgorithm, network: CubeSolverNetwork, output_dir: str, evaluator: FitnessEvaluator
):
    """Save current training checkpoint."""

    # Save best weights to JSON (for frontend)
    network.set_weights_flat(ga.best_ever.genome)
    export_weights_to_json(network, os.path.join(output_dir, "best_weights.json"))

    # Save training stats
    stats = ga.get_statistics()
    stats["evaluator"] = {
        "scramble_depth": evaluator.scramble_depth,
        "max_steps": evaluator.max_steps,
        "test_cubes": evaluator.num_test_cubes,
    }
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
            "hidden1": network.fc1.out_features,
            "hidden2": network.fc2.out_features,
        },
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
        for step in range(50):
            if cube.is_solved():
                solved += 1
                if verbose:
                    print(f"  Solved in {step} moves: {' '.join(moves_made)}")
                break

            state = cube.to_one_hot()
            action = network.predict_action(state, deterministic=True)
            move = MOVES[action]
            cube.apply_move(move)
            moves_made.append(move)
        else:
            if verbose:
                correct = cube.count_correct_stickers()
                print(f"  Not solved (correct stickers: {correct}/54)")

    if verbose:
        print(f"\nSolved {solved}/{num_tests} cubes")

    return solved


def main():
    args = parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"Output directory: {output_dir}")

    # Create network to get genome size
    network = CubeSolverNetwork(args.hidden1, args.hidden2)
    genome_size = network.get_weight_count()

    print("\nNetwork architecture:")
    print("  Input: 324 (54 × 6 one-hot)")
    print(f"  Hidden1: {args.hidden1}")
    print(f"  Hidden2: {args.hidden2}")
    print("  Output: 18 (moves)")
    print(f"  Total weights: {genome_size:,}")

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
        print(f"\nLoading checkpoint from {args.load}")
        with open(args.load) as f:
            checkpoint = json.load(f)

        # Create best individual from checkpoint
        best_genome = np.array(checkpoint["best_genome"], dtype=np.float32)
        ga.best_ever = Individual(genome=best_genome)
        ga.best_ever.fitness = checkpoint["best_fitness"]
        ga.generation = checkpoint["generation"]

        print(f"  Resumed from generation {ga.generation}")
        print(f"  Best fitness: {ga.best_ever.fitness:.2f}")
    else:
        ga.initialize_population()

    # Create fitness evaluator
    evaluator = FitnessEvaluator(
        num_test_cubes=args.test_cubes,
        scramble_depth=args.scramble_depth,
        max_steps=args.max_steps,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
    )

    print("\nTraining configuration:")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Scramble depth: {args.scramble_depth}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Test cubes: {args.test_cubes}")

    print("\nStarting training...\n")

    # Run evolution
    try:
        for gen in range(args.generations):
            # Regenerate test cubes every 10 generations
            if gen > 0 and gen % 10 == 0:
                evaluator.regenerate_test_cubes()

            ga.evolve_generation(evaluator)

            best = ga.population[0]
            avg = np.mean([ind.fitness for ind in ga.population])

            if not args.quiet:
                print(
                    f"Gen {ga.generation:4d} | Best: {best.fitness:6.2f} | "
                    f"Solved: {best.solved_count:2d}/{args.test_cubes} | "
                    f"Avg: {avg:6.2f}"
                )

            # Save checkpoint periodically
            if ga.generation % args.save_every == 0:
                save_checkpoint(ga, network, output_dir, evaluator)

            # Early stopping
            if args.target_fitness and ga.best_ever.fitness >= args.target_fitness:
                print(f"\nTarget fitness {args.target_fitness} reached!")
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final save
    save_checkpoint(ga, network, output_dir, evaluator)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Best fitness: {ga.best_ever.fitness:.2f}")
    print(f"Generations: {ga.generation}")
    print(f"Weights saved to: {output_dir}/best_weights.json")

    # Test the best network
    network.set_weights_flat(ga.best_ever.genome)
    test_best_network(network, args.scramble_depth, num_tests=5, verbose=not args.quiet)


if __name__ == "__main__":
    main()
