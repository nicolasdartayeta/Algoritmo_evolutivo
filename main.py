import numpy as np
import typing
import tsplib95
import time
import estrategiasCruce
import estrategiasMutacion
import estrategiasSeleccionPadres
import estrategiasSeleccionSobrevivientes
import algoritmoGenericoTSP

# --- Configuration ---
# TODO: Load these from a config file or command-line arguments
DEFAULT_POPULATION_SIZE = 50 # [cite: 39] Suggests 30-50
DEFAULT_TOURNAMENT_SIZE = 3  # Common practice, adjust as needed
DEFAULT_CROSSOVER_RATE = 0.8 # [cite: 25] Suggests 0.7 or 0.8
DEFAULT_MUTATION_RATE = 0.2  # [cite: 33] Suggests 0.05, 0.1, 0.2
DEFAULT_REPLACEMENT_RATE = 0.5 # [cite: 71] Suggests 30%, 50%, 70%
DEFAULT_MAX_GENERATIONS = 200 # [cite: 45] Suggests 200, 500, 700
ARCHIVO_COSTOS = 'costos.csv'

# --- Main Execution ---

if __name__ == '__main__':
    matriz_costos = tsplib95.load("ALL_tsp/bays29.tsp").edge_weights
    matriz_costos = np.array(matriz_costos)
    
    # --- Inicializar estrategias ---
    selection = estrategiasSeleccionPadres.SeleccionPorTorneo(tamanio_del_torneo=DEFAULT_TOURNAMENT_SIZE)
    crossover = estrategiasCruce.CrucePMX()
    crossover = estrategiasCruce.CruceBasadoEnArcos()
    mutation = estrategiasMutacion.InversionMutation()
    #mutation = InsertionMutation()
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

    print("\nStarting Genetic Algorithm execution...")
    start_time = time.perf_counter()

    best_route, best_cost = ga.run()

    end_time = time.perf_counter() # Get time just after running
    elapsed_time = end_time - start_time # Calculate the difference

    print(f"\nAlgorithm execution finished.")
    print(f"Total execution time: {elapsed_time:.4f} seconds")