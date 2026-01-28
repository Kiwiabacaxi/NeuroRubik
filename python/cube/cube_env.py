"""
Cube Environment for Training

Provides an environment for the genetic algorithm to evaluate
neural network solutions.
"""

from typing import List, Optional, Tuple

import numpy as np

from .cube_state import MOVES, CubeState


class CubeEnvironment:
    """Environment for training cube solvers."""

    def __init__(self, scramble_depth: int = 20, max_steps: int = 50):
        """
        Initialize environment.

        Args:
            scramble_depth: Number of moves to scramble the cube
            max_steps: Maximum moves allowed to solve
        """
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.cube: Optional[CubeState] = None
        self.initial_cube: Optional[CubeState] = None
        self.steps_taken = 0
        self.moves_history: List[str] = []

    def reset(self, scramble_moves: Optional[List[str]] = None) -> np.ndarray:
        """
        Reset environment with a new scrambled cube.

        Args:
            scramble_moves: Optional specific scramble sequence

        Returns:
            One-hot encoded state (324,)
        """
        self.cube = CubeState()

        if scramble_moves:
            self.cube.apply_moves(scramble_moves)
        else:
            self.cube.scramble(self.scramble_depth)

        self.initial_cube = self.cube.clone()
        self.steps_taken = 0
        self.moves_history = []

        return self.cube.to_one_hot()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take one step in the environment.

        Args:
            action: Move index (0-17)

        Returns:
            Tuple of (new_state, reward, done)
        """
        if self.cube is None:
            raise RuntimeError("Environment not reset")

        move = MOVES[action]
        self.cube.apply_move(move)
        self.steps_taken += 1
        self.moves_history.append(move)

        # Check if solved
        solved = self.cube.is_solved()
        done = solved or self.steps_taken >= self.max_steps

        # Calculate reward
        reward = self._calculate_reward(solved)

        return self.cube.to_one_hot(), reward, done

    def _calculate_reward(self, solved: bool) -> float:
        """
        Calculate reward for current state.

        Reward structure:
        - +100 for solving
        - Bonus based on correct stickers (0-54)
        - Small penalty per step to encourage shorter solutions
        """
        if solved:
            # Big bonus for solving, extra bonus for fewer moves
            return 100.0 + (self.max_steps - self.steps_taken) * 2

        # Partial reward based on progress
        correct = self.cube.count_correct_stickers()

        # Normalize to 0-1 range, then scale
        progress_reward = (correct / 54.0) * 10

        # Small step penalty
        step_penalty = -0.1

        return progress_reward + step_penalty

    def get_state(self) -> np.ndarray:
        """Get current state as one-hot encoding."""
        if self.cube is None:
            raise RuntimeError("Environment not reset")
        return self.cube.to_one_hot()

    def is_solved(self) -> bool:
        """Check if cube is solved."""
        return self.cube is not None and self.cube.is_solved()

    def evaluate_solution(self, actions: List[int]) -> Tuple[bool, int, float]:
        """
        Evaluate a complete solution sequence.

        Args:
            actions: List of move indices

        Returns:
            Tuple of (solved, steps_used, final_reward)
        """
        if self.initial_cube is None:
            raise RuntimeError("Environment not reset")

        # Reset to initial state
        self.cube = self.initial_cube.clone()
        self.steps_taken = 0
        self.moves_history = []

        total_reward = 0.0

        for action in actions:
            if self.steps_taken >= self.max_steps:
                break

            _, reward, done = self.step(action)
            total_reward += reward

            if done:
                break

        return self.is_solved(), self.steps_taken, total_reward


def generate_training_cubes(count: int, scramble_depth: int = 20) -> List[CubeState]:
    """
    Generate a batch of scrambled cubes for training evaluation.

    Args:
        count: Number of cubes to generate
        scramble_depth: Moves per scramble

    Returns:
        List of scrambled CubeState objects
    """
    cubes = []
    for _ in range(count):
        cube = CubeState()
        cube.scramble(scramble_depth)
        cubes.append(cube)
    return cubes
