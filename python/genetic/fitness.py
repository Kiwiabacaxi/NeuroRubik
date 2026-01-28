"""
Fitness Evaluation for Cube-Solving Networks - GPU Enhanced

Evaluates how well a neural network can solve scrambled cubes.
Supports GPU acceleration and curriculum learning.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch

from cube.cube_state import MOVES, CubeState
from neural.network import CubeSolverNetwork, DEVICE


class FitnessEvaluator:
    """Evaluates fitness of individuals by testing cube solving - GPU optimized."""

    def __init__(
        self,
        num_test_cubes: int = 20,
        scramble_depth: int = 5,
        max_steps: int = 100,
        hidden_sizes: tuple = (512, 512, 256, 128),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize evaluator.

        Args:
            num_test_cubes: Number of cubes to test on
            scramble_depth: Moves used to scramble each cube
            max_steps: Max moves allowed to solve
            hidden_sizes: Network architecture
            device: torch device to use
        """
        self.num_test_cubes = num_test_cubes
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.hidden_sizes = hidden_sizes
        self.device = device or DEVICE

        # Pre-generate test cubes for consistent evaluation
        self.test_cubes: List[CubeState] = []
        self.scramble_moves: List[List[str]] = []  # Store scramble sequences
        self.regenerate_test_cubes()

    def regenerate_test_cubes(self):
        """Generate new set of test cubes."""
        self.test_cubes = []
        self.scramble_moves = []
        for _ in range(self.num_test_cubes):
            cube = CubeState()
            moves = cube.scramble(self.scramble_depth)
            self.test_cubes.append(cube)
            self.scramble_moves.append(moves)

    def set_scramble_depth(self, depth: int):
        """Update scramble depth and regenerate cubes."""
        self.scramble_depth = depth
        self.regenerate_test_cubes()

    def evaluate(self, genome: np.ndarray) -> Tuple[float, int, float]:
        """
        Evaluate a single genome.

        Args:
            genome: Flat array of network weights

        Returns:
            Tuple of (fitness, cubes_solved, avg_correct_stickers)
        """
        # Create network from genome
        network = CubeSolverNetwork(
            hidden_sizes=self.hidden_sizes,
            dropout=0.0,  # No dropout during evaluation
            device=self.device,
        )
        network.set_weights_flat(genome)
        network.eval()

        total_reward = 0.0
        cubes_solved = 0
        total_correct = 0

        # Bonus multiplier for solving
        solve_bonus = 200 + self.scramble_depth * 20

        for i, test_cube in enumerate(self.test_cubes):
            cube = test_cube.clone()
            steps = 0
            visited_states = set()  # Prevent loops
            solved = False

            for _ in range(self.max_steps):
                if cube.is_solved():
                    cubes_solved += 1
                    solved = True
                    # Bigger bonus for solving quickly
                    efficiency_bonus = (self.max_steps - steps) / self.max_steps
                    total_reward += solve_bonus * (1 + efficiency_bonus)
                    total_correct += 54
                    break

                # Get state hash to detect loops
                state_hash = cube.state.tobytes()
                if state_hash in visited_states:
                    # Stuck in a loop - penalize and break
                    total_reward -= 5
                    break
                visited_states.add(state_hash)

                # Get action from network
                state = cube.to_one_hot()
                action = network.predict_action(state, deterministic=True)

                # Apply move
                move = MOVES[action]
                cube.apply_move(move)
                steps += 1

            # Track final correct stickers
            if not solved:
                correct = cube.count_correct_stickers()
                total_correct += correct

                # Reward based on progress (exponential to reward high correct counts)
                progress = correct / 54.0
                total_reward += progress * progress * 50

        # Calculate averages
        fitness = total_reward / self.num_test_cubes
        avg_correct = total_correct / self.num_test_cubes

        return fitness, cubes_solved, avg_correct


class CurriculumFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator with curriculum learning.

    Starts with easy scrambles and gradually increases difficulty
    as the population improves.
    """

    def __init__(
        self,
        num_test_cubes: int = 20,
        initial_depth: int = 1,
        max_depth: int = 20,
        depth_increase_threshold: float = 0.5,  # Increase when 50% solved
        max_steps: int = 100,
        hidden_sizes: tuple = (512, 512, 256, 128),
        device: Optional[torch.device] = None,
    ):
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.depth_increase_threshold = depth_increase_threshold
        self.current_depth = initial_depth
        self.generations_at_depth = 0
        self.best_solve_rate = 0.0

        super().__init__(
            num_test_cubes=num_test_cubes,
            scramble_depth=initial_depth,
            max_steps=max_steps,
            hidden_sizes=hidden_sizes,
            device=device,
        )

    def update_difficulty(self, solve_rate: float) -> bool:
        """
        Update difficulty based on performance.

        Args:
            solve_rate: Fraction of cubes solved (0-1)

        Returns:
            True if difficulty was increased
        """
        self.best_solve_rate = max(self.best_solve_rate, solve_rate)
        self.generations_at_depth += 1

        # Increase difficulty if solving well
        if solve_rate >= self.depth_increase_threshold and self.current_depth < self.max_depth:
            # Only increase if consistent performance
            if self.generations_at_depth >= 3:
                self.current_depth += 1
                self.scramble_depth = self.current_depth
                self.regenerate_test_cubes()
                self.generations_at_depth = 0
                self.best_solve_rate = 0.0
                return True

        return False

    def get_status(self) -> dict:
        """Get current curriculum status."""
        return {
            "current_depth": self.current_depth,
            "max_depth": self.max_depth,
            "generations_at_depth": self.generations_at_depth,
            "best_solve_rate": self.best_solve_rate,
        }


class BatchFitnessEvaluator:
    """
    Batch evaluator that processes multiple genomes efficiently.

    Uses shared network and batch operations for GPU efficiency.
    """

    def __init__(
        self,
        num_test_cubes: int = 20,
        scramble_depth: int = 5,
        max_steps: int = 100,
        hidden_sizes: tuple = (512, 512, 256, 128),
        device: Optional[torch.device] = None,
    ):
        self.evaluator = CurriculumFitnessEvaluator(
            num_test_cubes=num_test_cubes,
            initial_depth=scramble_depth,
            max_depth=20,
            max_steps=max_steps,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.device = device or DEVICE

    def evaluate_population(
        self, genomes: List[np.ndarray], show_progress: bool = False
    ) -> List[Tuple[float, int, float]]:
        """
        Evaluate all genomes.

        Returns:
            List of (fitness, solved_count, avg_correct) for each genome
        """
        results = []

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(genomes, desc="Evaluating", leave=False)
        else:
            iterator = genomes

        for genome in iterator:
            result = self.evaluator.evaluate(genome)
            results.append(result)

        return results

    def regenerate_test_cubes(self):
        """Generate new test cubes."""
        self.evaluator.regenerate_test_cubes()

    def update_difficulty(self, solve_rate: float) -> bool:
        """Update curriculum difficulty."""
        return self.evaluator.update_difficulty(solve_rate)

    def get_status(self) -> dict:
        """Get curriculum status."""
        return self.evaluator.get_status()

    def set_scramble_depth(self, depth: int):
        """Manually set scramble depth."""
        self.evaluator.current_depth = depth
        self.evaluator.set_scramble_depth(depth)

    def evaluate(self, genome: np.ndarray) -> tuple:
        """Evaluate a single genome (delegates to internal evaluator)."""
        return self.evaluator.evaluate(genome)
