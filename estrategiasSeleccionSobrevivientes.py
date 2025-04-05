import abc
import numpy as np

class EstrategiaDeSeleccionDeSobrevivientes(abc.ABC):
    @abc.abstractmethod
    def reemplazar(self, poblacion: np.ndarray, fitness: np.ndarray, hijos: np.ndarray, fitness_de_los_hijos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reemplaza individuos de la poblacion actual por los hijos."""
        pass

class SteadyStateReplacement(EstrategiaDeSeleccionDeSobrevivientes):
    # [cite: 68, 84]
    def __init__(self, replacement_rate: float):
        if not 0.0 <= replacement_rate <= 1.0:
            raise ValueError("Replacement rate must be between 0 and 1.")
        self.replacement_rate = replacement_rate

    def reemplazar(self, population: np.ndarray, fitness: np.ndarray, offspring: np.ndarray, offspring_fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replaces the worst 'n%' of the population with the best 'n%' of the offspring[cite: 69]."""
        tamanio_poblacion = population.shape[0]
        num_to_replace = int(tamanio_poblacion * self.replacement_rate)

        if num_to_replace == 0 or offspring.shape[0] == 0:
            return population, fitness # No replacement possible

        # Ensure we don't try to replace more than available offspring
        num_to_replace = min(num_to_replace, offspring.shape[0])

        # Find indices of the worst individuals in the current population
        worst_indices = np.argsort(fitness)[:num_to_replace]

        # Find indices of the best individuals in the offspring
        best_offspring_indices = np.argsort(offspring_fitness)[::-1][:num_to_replace] # Use [::-1] for descending sort

        # Replace
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[worst_indices] = offspring[best_offspring_indices]
        new_fitness[worst_indices] = offspring_fitness[best_offspring_indices]

        return new_population, new_fitness