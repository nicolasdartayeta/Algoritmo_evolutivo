import numpy as np
import abc # Abstract Base Classes
import typing

# --- Configuration ---
# TODO: Load these from a config file or command-line arguments
DEFAULT_POPULATION_SIZE = 30 # [cite: 39] Suggests 30-50
DEFAULT_TOURNAMENT_SIZE = 3  # Common practice, adjust as needed
DEFAULT_CROSSOVER_RATE = 0.8 # [cite: 25] Suggests 0.7 or 0.8
DEFAULT_MUTATION_RATE = 0.1  # [cite: 33] Suggests 0.05, 0.1, 0.2
DEFAULT_REPLACEMENT_RATE = 0.5 # [cite: 71] Suggests 30%, 50%, 70%
DEFAULT_MAX_GENERATIONS = 500 # [cite: 45] Suggests 200, 500, 700
ARCHIVO_COSTOS = 'costos.csv'

# --- Interfaces de las estrategias (Abstract Base Classes) ---

class EstrategiaDeSeleccion(abc.ABC):
    @abc.abstractmethod
    def seleccionar(self, poblacion: np.ndarray, fitness: np.ndarray, cantidad_de_padres: int) -> np.ndarray:
        """Selecciona padres de la poblacion."""
        pass

class EstrategiaDeCruce(abc.ABC):
    @abc.abstractmethod
    def cruzar(self, padres: np.ndarray, probabilidad_de_cruce: float) -> np.ndarray:
        """Cruza pares de padres para generar hijos."""
        pass

class EstrategiaDeMutacion(abc.ABC):
    @abc.abstractmethod
    def mutar(self, hijos: np.ndarray, probabilidad_de_mutacion: float) -> np.ndarray:
        """Aplica mutacciones a los hijos."""
        pass

class EstrategiaDeEleccionDeSobrevivientes(abc.ABC):
    @abc.abstractmethod
    def reemplazar(self, poblacion: np.ndarray, fitness: np.ndarray, hijos: np.ndarray, fitness_de_los_hijos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reemplaza individuos de la poblacion actual por los hijos."""
        pass

# --- Implementaciones de las estrategias ---

class SeleccionPorTorneo(EstrategiaDeSeleccion):
    def __init__(self, tamanio_del_torneo: int):
        if tamanio_del_torneo < 2:
            raise ValueError("El tamaño del torneo debe ser por lo menos de 2.")
        self.tamanio_del_torneo = tamanio_del_torneo
        self.rng = np.random.default_rng()

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

class CrucePMX(EstrategiaDeCruce):
    def __init__(self):
        self.rng = np.random.default_rng()

    def cruzar(self, padres: np.ndarray, probabilidad_de_cruce: float) -> np.ndarray:
        cantidad_padres, cantidad_ciudades = padres.shape
        hijos = np.empty_like(padres)
        num_hijos = 0

        # Ir agarrando padres de a parejas segun fueron agregados a la poblacion de padres.
        for i in range(0, cantidad_padres, 2):
            if i + 1 >= cantidad_padres: # La cantidad de padres son impares
                hijos[num_hijos] = padres[i] # Se copia el ultimo padre como hijo directamente
                num_hijos += 1
                continue

            padre_a = padres[i]
            padre_b = padres[i+1]

            if self.rng.random() < probabilidad_de_cruce:
                # Realiza el cruce
                hijo_a = self._pmx_pair(padre_a, padre_b)
                hijo_b = self._pmx_pair(padre_b, padre_a)
                hijos[num_hijos] = hijo_a
                hijos[num_hijos + 1] = hijo_b
            else:
                # No hay cruce, se pasan los padres como hijos
                hijos[num_hijos] = padre_a.copy()
                hijos[num_hijos + 1] = padre_b.copy()

            num_hijos += 2

        return hijos[:num_hijos] # Return only the generated offspring

    def _pmx_pair(self, padre_1: np.ndarray, padre_2: np.ndarray) -> np.ndarray:
        """Funcion que realiza realiza el cruce y produce un hijo."""
        tamanio = padre_1.shape[0]
        hijo = -np.ones(tamanio, dtype=padre_1.dtype) # Initialize with -1 or another placeholder

        # 1. Elegir dos puntos de cruce aleatorios
        punto_cruce_1, punto_cruce_2 = sorted(self.rng.choice(tamanio, 2, replace=False))

        # 2. Copiar segmento del padre_1 al hijo
        hijo[punto_cruce_1:punto_cruce_2+1] = padre_1[punto_cruce_1:punto_cruce_2+1]

        # 3. Mapear elementos del padre_2 entre los puntos de cruce al hijo
        for i in range(punto_cruce_1, punto_cruce_2 + 1):
            if padre_2[i] not in hijo[punto_cruce_1:punto_cruce_2+1]:
                valor_a_copiar = padre_2[i]
                j = padre_1[i]
                posicion_hijo = np.where(padre_2 == j)[0][0]
                while hijo[posicion_hijo] != -1:
                    k = hijo[posicion_hijo]
                    posicion_hijo = np.where(padre_2 == k)[0][0]
                
                hijo[posicion_hijo] = valor_a_copiar

        # 4. Copiar elementos faltantes (si los hay)
        for i in range(tamanio):
            if hijo[i] == -1:  # Si la posición aún está vacía después del paso 3
                hijo[i] = padre_2[i]

        # Verificacion
        assert len(np.unique(hijo)) == tamanio, "PMX resulto en ciudades duplicadas"
        assert -1 not in hijo, "PMX fallo en llenar todas las posiciones"

        return hijo

# TODO: Implement EdgeRecombinationCrossover [cite: 59]
class EdgeRecombinationCrossover(EstrategiaDeCruce):
     def cruzar(self, parents: np.ndarray, probabilidad_de_cruce: float) -> np.ndarray:
         print("Warning: Edge Recombination Crossover not implemented yet. Copying parents.")
         # Placeholder: just copy parents if crossover doesn't happen
         offspring = np.empty_like(parents)
         num_offspring = 0
         for i in range(0, parents.shape[0], 2):
             if i + 1 >= parents.shape[0]:
                 offspring[num_offspring] = parents[i].copy()
                 num_offspring += 1
                 continue
             if np.random.rand() < probabilidad_de_cruce:
                # Implement Edge Recombination logic here
                # For now, just copy
                 offspring[num_offspring] = parents[i].copy()
                 offspring[num_offspring + 1] = parents[i+1].copy()
             else:
                 offspring[num_offspring] = parents[i].copy()
                 offspring[num_offspring + 1] = parents[i+1].copy()
             num_offspring += 2
         return offspring[:num_offspring]


class InsertionMutation(EstrategiaDeMutacion):
    # [cite: 64]
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


class SteadyStateReplacement(EstrategiaDeEleccionDeSobrevivientes):
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

# --- Clase del algoritmo genético ---

class AlgoritmoGeneticoTSP:
    def __init__(self,
                 matriz_costos: np.ndarray,
                 tamanio_poblacion: int,
                 estrategia_de_padres: EstrategiaDeSeleccion,
                 estrategia_de_cruce: EstrategiaDeCruce,
                 estrategia_de_mutacion: EstrategiaDeMutacion,
                 estrategia_de_sobrevivientes: EstrategiaDeEleccionDeSobrevivientes,
                 probabilidad_de_cruce: float,
                 probabilidad_de_mutacion: float,
                 cantidad_maxima_de_generaciones: int):

        self.matriz_costos = matriz_costos
        self.cantidad_ciudades = matriz_costos.shape[0]
        self.tamanio_poblacion = tamanio_poblacion
        self.estrategia_de_padres = estrategia_de_padres
        self.estrategia_de_cruce = estrategia_de_cruce
        self.estrategia_de_mutacion = estrategia_de_mutacion
        self.estrategia_de_sobrevivientes = estrategia_de_sobrevivientes
        self.probabilidad_de_cruce = probabilidad_de_cruce
        self.probabilidad_de_mutacion = probabilidad_de_mutacion
        self.cantidad_maxima_de_generaciones = cantidad_maxima_de_generaciones

        self.poblacion = np.empty((self.tamanio_poblacion, self.cantidad_ciudades), dtype=np.int32)
        self.fitness = np.empty(self.tamanio_poblacion)
        self.costos = np.empty(self.tamanio_poblacion)

        self.mejor_solucion = None
        self.mejor_fitness = -np.inf
        self.mejor_costo = np.inf
        self.rng = np.random.default_rng()

    def _inicializar_poblacion(self):
        """Se generan permutaciones aleatorias de ciudades."""
        for i in range(self.tamanio_poblacion):
            self.poblacion[i] = self.rng.permutation(self.cantidad_ciudades)
        self._calcular_fitness()

    def _calcular_fitness(self, individuos: typing.Union[np.ndarray, None] = None) -> typing.Union[tuple[np.ndarray, np.ndarray], None]:
        """Calcula costo y fitness para un individuo dado o para toda la población."""
        # Usa un individuo dado o la poblacion dependiendo de si se pasó un parametro o no.
        poblacion_objetivo = individuos if individuos is not None else self.poblacion
        cantidad_de_individuos = poblacion_objetivo.shape[0]
        costos = np.empty(cantidad_de_individuos)
        fitness = np.empty(cantidad_de_individuos)

        for i in range(cantidad_de_individuos):
            costo = 0
            ruta = poblacion_objetivo[i]
            for j in range(self.cantidad_ciudades):
                ciudad_a = ruta[j]
                # El operador de modulo aca sirve para tener en cuenta el costo de volver desde la ultima ciudad a la primera.
                ciudad_b = ruta[(j + 1) % self.cantidad_ciudades]
                costo += self.matriz_costos[ciudad_a, ciudad_b]
            costos[i] = costo
            # Fitness: 1/costo, tiene en cuenta el caso en el que el costo sea 0 (raro).
            fitness[i] = 1.0 / costo if costo != 0 else np.inf

        if individuos is None:
            # Actualizar directamente los atributos de la instancia.
            self.costos = costos
            self.fitness = fitness
            return None # Indica que se modificaron los atributos de instancia
        else:
            # Devuelve los valores calculados si un individuo fue pasado como parámetro
            return costos, fitness


    def run(self):
        """Ejecuta el algoritmo genético."""
        self._inicializar_poblacion()
        # Setea la mejor solucion hasta el momento.
        if self.tamanio_poblacion > 0:
             id_mejor_solucion = np.argmax(self.fitness)
             self.mejor_solucion = self.poblacion[id_mejor_solucion].copy()
             self.mejor_fitness = self.fitness[id_mejor_solucion]
             self.mejor_costo = self.costos[id_mejor_solucion]
             print(f"Generacion 0: Mejor costo = {self.mejor_costo:.2f}")
        else:
             print("Advertencia: No se puede correr el algoritmo con una población de tamaño 0.")
             return None, np.inf # Early return por la advertencia.


        for generacion in range(1, self.cantidad_maxima_de_generaciones + 1):
            """1. Selección de padres"""
            # Asegurar padres suficientes para el cruce
            cantidad_de_padres_a_seleccionar = self.tamanio_poblacion
            if cantidad_de_padres_a_seleccionar % 2 != 0: # Si el tamaño de la poblacion es impar
                 cantidad_de_padres_a_seleccionar +=1 # Se necesita un padre extra para poder formar parejas

            padres = self.estrategia_de_padres.seleccionar(self.poblacion, self.fitness, cantidad_de_padres_a_seleccionar)

            """2. Cruce"""
            hijos = self.estrategia_de_cruce.cruzar(padres, self.probabilidad_de_cruce)

            """3. Mutación"""
            hijos_mutados = self.estrategia_de_mutacion.mutar(hijos, self.probabilidad_de_mutacion)

            """4. Evaluate new individuals"""
            if hijos_mutados.shape[0] > 0:
                # Calcular fitness de los hijos mutados
                costos_hijos, fitness_hijos = self._calcular_fitness(hijos_mutados)
            else: # Caso en el que no se producen hijos (probabilidad de cruce 0)
                costos_hijos, fitness_hijos = np.array([]), np.array([])
                # Arreglos vacios
                hijos_mutados = np.empty((0, self.cantidad_ciudades), dtype=self.poblacion.dtype)


            """5. Seleccion de sobrevivientes"""
            # Asegurarse que haya hijos por los que reemplazar a la población actual
            if hijos_mutados.size > 0 and fitness_hijos.size > 0 :
                 self.poblacion, self.fitness = self.estrategia_de_sobrevivientes.reemplazar(
                     self.poblacion, self.fitness, hijos_mutados, fitness_hijos
                 )
                 # Actualizar las variables de instancia de costos y fitness para la nueva poblacion generada
                 self._calcular_fitness()
            # else: No se generaron hijos para reemplazar a la poblacion actual, por lo que quedda igual.


            # Actualizar mejor solución hasta el momento
            id_mejor_solucion = np.argmax(self.fitness)
            fitness_mejor_solucion = self.fitness[id_mejor_solucion]
            if fitness_mejor_solucion > self.mejor_fitness:
                self.mejor_fitness = fitness_mejor_solucion
                self.mejor_solucion = self.poblacion[id_mejor_solucion].copy()
                self.mejor_costo = self.costos[id_mejor_solucion]

            if generacion % 50 == 0 or generacion == self.cantidad_maxima_de_generaciones: # Mostrar progreso del algoritmo
                 print(f"Generación {generacion}: Mejor costo = {self.mejor_costo:.2f}")


        print("\n--- Algoritmo finalizado ---")
        print(f"Mejor solucion encontrada: {self.mejor_solucion}")
        print(f"Mejor costo: {self.mejor_costo:.2f}")
        print(f"Mejor fitness: {self.mejor_fitness:.6f}")

        return self.mejor_solucion, self.mejor_costo

# --- Main Execution ---

if __name__ == '__main__':
    # Cargar matriz de costos
    try:
        matriz_costos = np.loadtxt(ARCHIVO_COSTOS, delimiter=',')
        print(f"Matriz de costos cargada desde el archivo '{ARCHIVO_COSTOS}' ({matriz_costos.shape[0]} ciudades).")
    except FileNotFoundError:
        print(f"Error: Archivo '{ARCHIVO_COSTOS}' no encontrado.")
        exit(1)
    except Exception as e:
        print(f"Error al intentar cargar el archivo: {e}")
        exit(1)

    # --- Inicializar estrategias ---
    selection = SeleccionPorTorneo(tamanio_del_torneo=DEFAULT_TOURNAMENT_SIZE)
    crossover = CrucePMX()
    #crossover = EdgeRecombinationCrossover() # Swap to PMX easily [cite: 57, 59]
    mutation = InversionMutation() # Swap to InsertionMutation easily [cite: 64, 66]
    mutation = InsertionMutation()
    replacement = SteadyStateReplacement(replacement_rate=DEFAULT_REPLACEMENT_RATE) # [cite: 68]

    # --- Inicializar y correr el algoritmo ---
    ga = AlgoritmoGeneticoTSP(
        matriz_costos=matriz_costos,
        tamanio_poblacion=DEFAULT_POPULATION_SIZE,
        estrategia_de_padres=selection,
        estrategia_de_cruce=crossover,
        estrategia_de_mutacion=mutation,
        estrategia_de_sobrevivientes=replacement,
        probabilidad_de_cruce=DEFAULT_CROSSOVER_RATE,
        probabilidad_de_mutacion=DEFAULT_MUTATION_RATE,
        cantidad_maxima_de_generaciones=DEFAULT_MAX_GENERATIONS
    )

    best_route, best_cost = ga.run()