import abc

import numpy as np

class EstrategiaDeMutacion(abc.ABC):
    @abc.abstractmethod
    def mutar(self, hijos: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        """Aplica mutacciones a los hijos."""
        pass


class MutacionPorInsersion(EstrategiaDeMutacion):
    def __init__(self):
        self.rng = np.random.default_rng()

    def mutar(self, hijos: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        hijos_mutados = hijos.copy()
        cantidad_de_hijos, cantidad_ciudades = hijos_mutados.shape

        for i in range(cantidad_de_hijos):
            if self.rng.random() < probabilidad_de_mutacion:
                # Seleccionar una ciudad al azar
                idx_ciudad_a_mover = self.rng.integers(0, cantidad_ciudades)
                ciudad_a_mover = hijos_mutados[i, idx_ciudad_a_mover]

                # Eliminarla de su posicino original
                ruta_temporal = np.delete(hijos_mutados[i], idx_ciudad_a_mover)

                # Seleccionar una posicion para insertarla
                posicion_de_insercion = self.rng.integers(0, cantidad_ciudades) # Can insert at the end

                # Insertarla
                hijos_mutados[i] = np.insert(ruta_temporal, posicion_de_insercion, ciudad_a_mover)
        return hijos_mutados

class MutacionPorInversion(EstrategiaDeMutacion):
    def __init__(self):
        self.rng = np.random.default_rng()

    def mutar(self, hijos: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        hijos_mutados = hijos.copy()
        cantidad_de_hijos, cantidad_ciudades = hijos_mutados.shape

        for i in range(cantidad_de_hijos):
            if self.rng.random() < probabilidad_de_mutacion:
                # Seleccionar dos ciudades
                idx1, idx2 = sorted(self.rng.choice(cantidad_ciudades, 2, replace=False))

                # Invertir el segmento entre las dos ciudades (incluyendo a las ciudades)
                hijos_mutados[i, idx1:idx2+1] = hijos_mutados[i, idx1:idx2+1][::-1]

        return hijos_mutados
