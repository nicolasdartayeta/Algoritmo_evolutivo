import abc

import numpy as np

class EstrategiaDeMutacion(abc.ABC):
    @abc.abstractmethod
    def mutar(self, hijos: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        """Aplica mutacciones a los hijos."""
        pass


class InsertionMutation(EstrategiaDeMutacion):
    def __init__(self):
        self.rng = np.random.default_rng()

    def mutar(self, offspring: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        mutated_offspring = offspring.copy()
        num_offspring, cantidad_ciudades = mutated_offspring.shape

        for i in range(num_offspring):
            if self.rng.random() < probabilidad_de_mutacion:
                # Select a city to move
                idx_to_move = self.rng.integers(0, cantidad_ciudades)
                city_to_move = mutated_offspring[i, idx_to_move]

                # Delete it from its original position
                temp_route = np.delete(mutated_offspring[i], idx_to_move)

                # Select a new position to insert
                insert_pos = self.rng.integers(0, cantidad_ciudades) # Can insert at the end

                # Insert the city
                mutated_offspring[i] = np.insert(temp_route, insert_pos, city_to_move)
        return mutated_offspring

class InversionMutation(EstrategiaDeMutacion):
    # [cite: 66]
    def __init__(self):
        self.rng = np.random.default_rng()

    def mutar(self, offspring: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        mutated_offspring = offspring.copy()
        num_offspring, cantidad_ciudades = mutated_offspring.shape

        for i in range(num_offspring):
            if self.rng.random() < probabilidad_de_mutacion:
                # Select two indices for the subarray
                idx1, idx2 = sorted(self.rng.choice(cantidad_ciudades, 2, replace=False))

                # Reverse the segment between idx1 and idx2 (inclusive)
                mutated_offspring[i, idx1:idx2+1] = mutated_offspring[i, idx1:idx2+1][::-1]

        return mutated_offspring
