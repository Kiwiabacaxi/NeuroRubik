"""
Genetic Algorithm Evolution Engine

Main evolution loop with selection, crossover, and mutation.
"""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .individual import Individual


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving neural network weights.
    """

    def __init__(
        self,
        population_size: int = 50,
        genome_size: int = 0,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.3,
        crossover_rate: float = 0.7,
        elitism_count: int = 5,
        tournament_size: int = 5,
    ):
        """
        Initialize genetic algorithm.

        Args:
            population_size: Number of individuals
            genome_size: Size of each genome (network weights)
            mutation_rate: Probability of mutating each gene
            mutation_strength: Standard deviation of mutations
            crossover_rate: Probability of crossover vs cloning
            elitism_count: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
        """
        self.population_size = population_size
        self.genome_size = genome_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size

        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None

        # Statistics
        self.history: Dict[str, List] = {
            "best_fitness": [],
            "avg_fitness": [],
            "solved_count": [],
            "generation_time": [],
        }

    def initialize_population(self):
        """Create initial random population."""
        self.population = [
            Individual(genome_size=self.genome_size) for _ in range(self.population_size)
        ]
        self.generation = 0
        self.best_ever = None
        self.history = {k: [] for k in self.history}

    def tournament_select(self) -> Individual:
        """Select individual using tournament selection."""
        candidates = np.random.choice(
            len(self.population),
            size=min(self.tournament_size, len(self.population)),
            replace=False,
        )
        best = self.population[candidates[0]]
        for idx in candidates[1:]:
            if self.population[idx].fitness > best.fitness:
                best = self.population[idx]
        return best

    def evolve_generation(self, evaluator: Any, progress_callback: Optional[Callable] = None):
        """
        Evolve one generation.

        Args:
            evaluator: Fitness evaluator
            progress_callback: Optional callback(generation, best, avg)
        """
        start_time = time.time()

        # Evaluate fitness
        for ind in self.population:
            fitness, solved, avg_steps = evaluator.evaluate(ind.genome)
            ind.fitness = fitness
            ind.solved_count = solved
            ind.avg_steps = avg_steps

        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best ever
        if self.best_ever is None or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0].clone()

        # Record statistics
        fitnesses = [ind.fitness for ind in self.population]
        solved_total = sum(ind.solved_count for ind in self.population)

        self.history["best_fitness"].append(self.population[0].fitness)
        self.history["avg_fitness"].append(np.mean(fitnesses))
        self.history["solved_count"].append(solved_total)
        self.history["generation_time"].append(time.time() - start_time)

        # Call progress callback
        if progress_callback:
            progress_callback(
                self.generation, self.population[0].fitness, np.mean(fitnesses), solved_total
            )

        # Create next generation
        new_population = []

        # Elitism: keep best individuals
        for i in range(self.elitism_count):
            new_population.append(self.population[i].clone())

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_select()

            if np.random.random() < self.crossover_rate:
                parent2 = self.tournament_select()
                child = Individual.crossover(parent1, parent2)
            else:
                child = parent1.clone()

            child.mutate(self.mutation_rate, self.mutation_strength)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def run(
        self,
        evaluator: Any,
        num_generations: int,
        target_fitness: Optional[float] = None,
        regenerate_cubes_every: int = 10,
        verbose: bool = True,
    ) -> Individual:
        """
        Run the genetic algorithm.

        Args:
            evaluator: Fitness evaluator
            num_generations: Maximum generations to run
            target_fitness: Stop early if this fitness is reached
            regenerate_cubes_every: Regenerate test cubes every N generations
            verbose: Show progress bar

        Returns:
            Best individual found
        """
        if not self.population:
            self.initialize_population()

        iterator = (
            tqdm(range(num_generations), desc="Evolving") if verbose else range(num_generations)
        )

        for gen in iterator:
            # Occasionally regenerate test cubes to avoid overfitting
            if gen > 0 and gen % regenerate_cubes_every == 0:
                evaluator.regenerate_test_cubes()

            self.evolve_generation(evaluator)

            if verbose:
                best = self.population[0]
                iterator.set_postfix(
                    {
                        "best": f"{best.fitness:.1f}",
                        "solved": best.solved_count,
                        "avg": f"{self.history['avg_fitness'][-1]:.1f}",
                    }
                )

            # Early stopping
            if target_fitness and self.best_ever.fitness >= target_fitness:
                if verbose:
                    print(f"\nTarget fitness {target_fitness} reached!")
                break

        return self.best_ever

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "generations": self.generation,
            "best_fitness": self.best_ever.fitness if self.best_ever else 0,
            "best_solved": self.best_ever.solved_count if self.best_ever else 0,
            "history": self.history,
            "config": {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "mutation_strength": self.mutation_strength,
                "crossover_rate": self.crossover_rate,
                "elitism_count": self.elitism_count,
            },
        }
