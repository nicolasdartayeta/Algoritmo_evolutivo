import abc
import numpy as np

class estrategiasSeleccionPadres(abc.ABC):
    @abc.abstractmethod
    def seleccionar(self, poblacion: np.ndarray, fitness: np.ndarray, cantidad_de_padres: int) -> np.ndarray:
        """Selecciona padres de la poblacion."""
        pass

class SeleccionPorRankingLineal(estrategiasSeleccionPadres):
    def __init__(self, sp: float):
        assert 1.0 < sp <= 2.0, f"El parametro del cruce lineal debe estar en (1.0, 2.0]"
        self.sp = sp
        self.rng = np.random.default_rng()

    def __str__(self):
        return f"Selección por ranking lineal con parametro {self.sp}"
    
    def seleccionar(self, poblacion: np.ndarray, fitness: np.ndarray, cantidad_de_padres: int) -> np.ndarray:
        """Selecciona individuos de la poblacion por ranking."""
        tamanio_poblacion = poblacion.shape[0]
        padres_seleccionados = np.empty((cantidad_de_padres, poblacion.shape[1]), dtype=poblacion.dtype)

        probabilidades = [(2 - self.sp) / tamanio_poblacion + (2 * i * (self.sp - 1)) / (tamanio_poblacion * (tamanio_poblacion - 1)) for i in range(tamanio_poblacion)]
        proabilidades_acumuladas = np.cumsum(probabilidades)

        id_peores_a_mejores = np.argsort(fitness)

        for i in range(cantidad_de_padres):
            random_number = self.rng.random()
            j = 0
            while random_number > proabilidades_acumuladas[j]:
                j += 1

            padres_seleccionados[i] = poblacion[id_peores_a_mejores[j]]

        return padres_seleccionados



class SeleccionPorTorneo(estrategiasSeleccionPadres):
    def __init__(self, tamanio_del_torneo: int):
        if tamanio_del_torneo < 2:
            raise ValueError("El tamaño del torneo debe ser por lo menos de 2.")
        self.tamanio_del_torneo = tamanio_del_torneo
        self.rng = np.random.default_rng()

    def __str__(self):
        return f"Seleeción por torneo con tamaño del torneo {self.tamanio_del_torneo}"

    def seleccionar(self, poblacion: np.ndarray, fitness: np.ndarray, cantidad_de_padres: int) -> np.ndarray:
        """Seleccion padres por torneos de tamanio_del_torneo tamaño."""
        tamanio_poblacion = poblacion.shape[0]
        padres_seleccionados = np.empty((cantidad_de_padres, poblacion.shape[1]), dtype=poblacion.dtype)

        for i in range(cantidad_de_padres):
            # Seleccionar k individuos para el torneo
            competitor_indices = self.rng.choice(tamanio_poblacion, self.tamanio_del_torneo, replace=False)
            competitors_fitness = fitness[competitor_indices]

            # Encontrar el numero de la ciudad ganadora del torneo
            id_local_ganador = np.argmax(competitors_fitness)
            id_global_ganador = competitor_indices[id_local_ganador]

            # Agregarlo a la lista de padres
            padres_seleccionados[i] = poblacion[id_global_ganador]
            # Nota: se deja a los padres volver a ser elegidos.

        return padres_seleccionados