import numpy as np
import typing
import estrategiasCruce
import estrategiasMutacion
import estrategiasSeleccionPadres
import estrategiasSeleccionSobrevivientes

class AlgoritmoGeneticoTSP:
    def __init__(self,
                 matriz_costos: np.ndarray,
                 tamanio_poblacion: int,
                 estrategia_de_padres: estrategiasSeleccionPadres.estrategiasSeleccionPadres,
                 estrategia_de_cruce: estrategiasCruce.EstrategiaDeCruce,
                 estrategia_de_mutacion: estrategiasMutacion.EstrategiaDeMutacion,
                 estrategia_de_sobrevivientes: estrategiasSeleccionSobrevivientes.EstrategiaDeSeleccionDeSobrevivientes,
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
                 print(f"Generación {generacion}: Mejor costo = {self.mejor_costo:.2f}. Tamaño de la poblacion: {len(self.poblacion)}")


        print("\n--- Algoritmo finalizado ---")
        print(f"Mejor solucion encontrada: {self.mejor_solucion}")
        print(f"Mejor costo: {self.mejor_costo:.2f}")
        print(f"Mejor fitness: {self.mejor_fitness:.6f}")

        return self.mejor_solucion, self.mejor_costo