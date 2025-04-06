import abc
import numpy as np

class EstrategiaDeSeleccionDeSobrevivientes(abc.ABC):
    @abc.abstractmethod
    def reemplazar(self, poblacion: np.ndarray, fitness: np.ndarray, hijos: np.ndarray, fitness_de_los_hijos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reemplaza individuos de la poblacion actual por los hijos."""
        pass

class SteadyStateReplacement(EstrategiaDeSeleccionDeSobrevivientes):
    def __init__(self, replacement_rate: float):
        if not 0.0 <= replacement_rate <= 1.0:
            raise ValueError("Replacement rate must be between 0 and 1.")
        self.replacement_rate = replacement_rate

    def reemplazar(self, poblacion: np.ndarray, fitness: np.ndarray, hijos: np.ndarray, fitness_hijos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reemplaza el n% peor de la poblacion actual con el n% mejor de los hijos creados."""
        tamanio_poblacion = poblacion.shape[0]
        cantidad_a_reemplazar = int(tamanio_poblacion * self.replacement_rate)

        if cantidad_a_reemplazar == 0 or hijos.shape[0] == 0:
            return poblacion, fitness

        cantidad_a_reemplazar = min(cantidad_a_reemplazar, hijos.shape[0])

        # Indices de las peores soluciones de la poblacion actual
        id_peores = np.argsort(fitness)[:cantidad_a_reemplazar]

        # Indices de las mejores soluciones de los hijos
        id_mejores_hijos = np.argsort(fitness_hijos)[::-1][:cantidad_a_reemplazar]

        # Reemplazar
        poblacion_nueva = poblacion.copy()
        nuevo_fitness = fitness.copy()
        poblacion_nueva[id_peores] = hijos[id_mejores_hijos]
        nuevo_fitness[id_peores] = fitness_hijos[id_mejores_hijos]

        return poblacion_nueva, nuevo_fitness