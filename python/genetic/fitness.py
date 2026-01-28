"""
Fitness Evaluation for Cube-Solving Networks

Evaluates how well a neural network can solve scrambled cubes.
"""

import os
import sys
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube.cube_state import MOVES, CubeState
from neural.network import CubeSolverNetwork


class FitnessEvaluator:
    """Evaluates fitness of individuals by testing cube solving."""

    def __init__(
        self,
        num_test_cubes: int = 10,
        scramble_depth: int = 5,
        max_steps: int = 50,
        hidden1: int = 256,
        hidden2: int = 128,
    ):
        """
        Initialize evaluator.

        Args:
            num_test_cubes: Number of cubes to test on
            scramble_depth: Moves used to scramble each cube
            max_steps: Max moves allowed to solve
            hidden1: Network hidden layer 1 size
            hidden2: Network hidden layer 2 size
        """
        self.num_test_cubes = num_test_cubes
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Pre-generate test cubes for consistent evaluation
        self.test_cubes: List[CubeState] = []
        self.regenerate_test_cubes()

    def regenerate_test_cubes(self):
        """Generate new set of test cubes."""
        self.test_cubes = []
        for _ in range(self.num_test_cubes):
            cube = CubeState()
            cube.scramble(self.scramble_depth)
            self.test_cubes.append(cube)

    def evaluate(self, genome: np.ndarray) -> Tuple[float, int, float]:
        """
        Evaluate a single genome.

        Args:
            genome: Flat array of network weights

        Returns:
            Tuple of (fitness, cubes_solved, avg_steps)
        """
        # Create network from genome
        network = CubeSolverNetwork(self.hidden1, self.hidden2)
        network.set_weights_flat(genome)
        network.eval()

        total_reward = 0.0
        cubes_solved = 0
        total_steps = 0

        for test_cube in self.test_cubes:
            cube = test_cube.clone()
            steps = 0

            for _ in range(self.max_steps):
                if cube.is_solved():
                    cubes_solved += 1
                    # Bonus for solving quickly
                    total_reward += 100 + (self.max_steps - steps) * 2
                    break

                # Get action from network
                state = cube.to_one_hot()
                action = network.predict_action(state, deterministic=True)

                # Apply move
                move = MOVES[action]
                cube.apply_move(move)
                steps += 1

            total_steps += steps

            # Partial reward based on final state
            if not cube.is_solved():
                correct = cube.count_correct_stickers()
                total_reward += (correct / 54.0) * 20

        # Calculate average
        fitness = total_reward / self.num_test_cubes
        avg_steps = total_steps / self.num_test_cubes

        return fitness, cubes_solved, avg_steps


def _evaluate_individual(args):
    """Helper function for parallel evaluation."""
    genome, evaluator_params = args

    evaluator = FitnessEvaluator(
        num_test_cubes=evaluator_params["num_test_cubes"],
        scramble_depth=evaluator_params["scramble_depth"],
        max_steps=evaluator_params["max_steps"],
        hidden1=evaluator_params["hidden1"],
        hidden2=evaluator_params["hidden2"],
    )

    # Use same test cubes for consistency
    evaluator.test_cubes = evaluator_params["test_cubes"]

    return evaluator.evaluate(genome)


class ParallelFitnessEvaluator:
    """Evaluates multiple individuals in parallel using multiprocessing."""

    def __init__(
        self,
        num_test_cubes: int = 10,
        scramble_depth: int = 5,
        max_steps: int = 50,
        hidden1: int = 256,
        hidden2: int = 128,
        num_workers: int = 4,
    ):
        self.evaluator = FitnessEvaluator(
            num_test_cubes, scramble_depth, max_steps, hidden1, hidden2
        )
        self.num_workers = num_workers

    def evaluate_population(self, genomes: List[np.ndarray]) -> List[Tuple[float, int, float]]:
        """
        Evaluate all genomes in parallel.

        Returns:
            List of (fitness, solved_count, avg_steps) for each genome
        """
        # Prepare arguments
        evaluator_params = {
            "num_test_cubes": self.evaluator.num_test_cubes,
            "scramble_depth": self.evaluator.scramble_depth,
            "max_steps": self.evaluator.max_steps,
            "hidden1": self.evaluator.hidden1,
            "hidden2": self.evaluator.hidden2,
            "test_cubes": self.evaluator.test_cubes,
        }

        args = [(genome, evaluator_params) for genome in genomes]

        # Parallel evaluation
        with Pool(self.num_workers) as pool:
            results = pool.map(_evaluate_individual, args)

        return results

    def regenerate_test_cubes(self):
        """Generate new test cubes."""
        self.evaluator.regenerate_test_cubes()
