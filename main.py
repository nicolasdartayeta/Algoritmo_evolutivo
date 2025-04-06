import numpy as np
import tsplib95
import time
import estrategiasCruce
import estrategiasMutacion
import estrategiasSeleccionPadres
import estrategiasSeleccionSobrevivientes
import algoritmoGenericoTSP
import math

# --- Configuracion ---
# TODO: Load these from a config file or command-line arguments
DEFAULT_POPULATION_SIZE = 100 # [cite: 39] Suggests 30-50
DEFAULT_TOURNAMENT_SIZE = 3  # Common practice, adjust as needed
DEFAULT_CROSSOVER_RATE = 0.8 # [cite: 25] Suggests 0.7 or 0.8
DEFAULT_MUTATION_RATE = 0.2  # [cite: 33] Suggests 0.05, 0.1, 0.2
DEFAULT_REPLACEMENT_RATE = 0.9 # [cite: 71] Suggests 30%, 50%, 70%
DEFAULT_MAX_GENERATIONS = 500 # [cite: 45] Suggests 200, 500, 700
ARCHIVO_COSTOS = 'costos.csv'

# --- Funcion de ayuda ---
def calcular_distancia(start, end):
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1])**2)

def cargar_matriz(filename) -> np.array:
    problem = tsplib95.load(filename, special=calcular_distancia)

    if problem.is_weighted():
        if problem.edge_data_format != "FULL_MATRIX":
            matriz_costos = np.empty(shape=(problem.dimension, problem.dimension))
            for ciudad1 in range(problem.dimension):
                for ciudad2 in range(problem.dimension):
                    matriz_costos[ciudad1][ciudad2] = problem.get_weight(ciudad1 + 1, ciudad2 + 1)
        else:
            matriz_costos = np.arrayy(problem.edge_weights)
    else:
        raise ValueError("El archivo especificado no tiene caminos con costo.")

    return matriz_costos

# --- Ejecucion principal ---

if __name__ == '__main__':
    matriz_costos = cargar_matriz("ALL_tsp/berlin52.tsp")
    print(matriz_costos)
    
    # --- Inicializar estrategias ---
    selection = estrategiasSeleccionPadres.SeleccionPorTorneo(tamanio_del_torneo=DEFAULT_TOURNAMENT_SIZE)
    crossover = estrategiasCruce.CrucePMX()
    #crossover = estrategiasCruce.CruceBasadoEnArcos()
    mutation = estrategiasMutacion.MutacionPorInversion()
    #mutation = estrategiasMutacion.MutacionPorInsersion()
    replacement = estrategiasSeleccionSobrevivientes.SteadyStateReplacement(replacement_rate=DEFAULT_REPLACEMENT_RATE) # [cite: 68]

    # --- Inicializar y correr el algoritmo ---
    ga = algoritmoGenericoTSP.AlgoritmoGeneticoTSP(
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

    print("\nEmpezadno ejecucion del algoritmo genético...")
    tiempo_de_inicio = time.perf_counter()

    best_route, best_cost = ga.run()

    tiempo_de_finalizacion = time.perf_counter()
    tiempo_transcurrido = tiempo_de_finalizacion - tiempo_de_inicio

    print(f"\nEjecucion del algoritmo finalizada.")
    print(f"Tiempo de ejecución: {tiempo_transcurrido:.4f} segundos")