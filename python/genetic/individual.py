"""
Individual representation for Genetic Algorithm

Each individual represents the weights of a neural network.
"""

from typing import Optional

import numpy as np


class Individual:
    """
    Represents an individual in the genetic algorithm population.

    An individual is essentially a set of neural network weights
    along with its fitness score.
    """

    def __init__(self, genome: Optional[np.ndarray] = None, genome_size: int = 0):
        """
        Initialize an individual.

        Args:
            genome: Optional pre-defined genome (weights)
            genome_size: Size of genome if creating randomly
        """
        if genome is not None:
            self.genome = genome.copy()
        elif genome_size > 0:
            # Initialize with small random values
            self.genome = np.random.randn(genome_size).astype(np.float32) * 0.1
        else:
            raise ValueError("Must provide either genome or genome_size")

        self.fitness: float = 0.0
        self.solved_count: int = 0
        self.avg_steps: float = 0.0

    def clone(self) -> "Individual":
        """Create a copy of this individual."""
        new_ind = Individual(genome=self.genome)
        new_ind.fitness = self.fitness
        new_ind.solved_count = self.solved_count
        new_ind.avg_steps = self.avg_steps
        return new_ind

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """
        Apply mutation to the genome.

        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Standard deviation of mutation
        """
        mask = np.random.random(len(self.genome)) < mutation_rate
        mutations = np.random.randn(len(self.genome)) * mutation_strength
        self.genome += mask * mutations

    @staticmethod
    def crossover(parent1: "Individual", parent2: "Individual") -> "Individual":
        """
        Create child through crossover of two parents.

        Uses uniform crossover - each gene randomly from either parent.
        """
        mask = np.random.random(len(parent1.genome)) < 0.5
        child_genome = np.where(mask, parent1.genome, parent2.genome)
        return Individual(genome=child_genome)

    @staticmethod
    def crossover_single_point(parent1: "Individual", parent2: "Individual") -> "Individual":
        """Single point crossover."""
        point = np.random.randint(0, len(parent1.genome))
        child_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
        return Individual(genome=child_genome)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.2f}, solved={self.solved_count})"
