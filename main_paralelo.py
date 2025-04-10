import numpy as np
import tsplib95
import time
import estrategiasCruce
import estrategiasMutacion
import estrategiasSeleccionPadres
import estrategiasSeleccionSobrevivientes
import algoritmoGenericoTSP
import math
import itertools # To generate combinations
import concurrent.futures # For parallel execution
import os # To suggest worker count
from datetime import datetime # To timestamp the output file

# --- Funcion de ayuda (calculating distance, loading matrix - keep as is) ---
def calcular_distancia(start, end):
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1])**2)

def cargar_matriz(filename) -> np.ndarray:
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


# --- Worker Function ---
# This function runs one full benchmark (N executions) for a single configuration
def run_single_configuration(config):
    """
    Runs the GA for a given configuration multiple times and returns avg results.
    'config' is a dictionary containing all necessary parameters.
    """
    # Unpack config (makes code clearer)
    matriz_costos = config['matriz_costos']
    tamanio_poblacion = config['tamanio_poblacion']
    probabilidad_de_cruce = config['probabilidad_de_cruce']
    probabilidad_de_mutacion = config['probabilidad_de_mutacion']
    cantidad_de_generaciones = config['cantidad_de_generaciones']
    operador_seleccion = config['operador_seleccion']
    operador_cruce = config['operador_cruce']
    operador_mutacion = config['operador_mutacion']
    operador_reemplazo = config['operador_reemplazo']
    cantidad_ejecuciones = config['cantidad_ejecuciones']

    total_tiempo = 0
    total_resultado = 0
    all_costs = [] # Store individual run costs

    # Run the GA multiple times for averaging
    for _ in range(cantidad_ejecuciones):
        # --- IMPORTANT: Instantiate the GA *inside* this loop ---
        # This ensures each run starts fresh and is independent,
        # avoiding issues with leftover state from previous runs.
        ga_run = algoritmoGenericoTSP.AlgoritmoGeneticoTSP(
            matriz_costos=matriz_costos,
            tamanio_poblacion=tamanio_poblacion,
            # Pass the instantiated operator objects directly
            estrategia_de_padres=operador_seleccion,
            estrategia_de_cruce=operador_cruce,
            estrategia_de_mutacion=operador_mutacion,
            estrategia_de_sobrevivientes=operador_reemplazo,
            probabilidad_de_cruce=probabilidad_de_cruce,
            probabilidad_de_mutacion=probabilidad_de_mutacion,
            cantidad_maxima_de_generaciones=cantidad_de_generaciones
        )
        # --- ---

        tiempo_de_inicio = time.perf_counter()
        _, best_cost = ga_run.run() # Run this instance
        tiempo_de_finalizacion = time.perf_counter()

        total_tiempo += (tiempo_de_finalizacion - tiempo_de_inicio)
        total_resultado += best_cost
        all_costs.append(best_cost) # Keep track of costs per run

    avg_tiempo = total_tiempo / cantidad_ejecuciones
    avg_resultado = total_resultado / cantidad_ejecuciones
    std_dev_resultado = np.std(all_costs) # Calculate standard deviation of costs

    # Return the original config dict and the results
    return (config, avg_tiempo, avg_resultado, std_dev_resultado)

# --- Helper to get operator description ---
def get_operator_desc(op):
    """Creates a descriptive string for an operator instance."""
    name = type(op).__name__
    params = []
    if hasattr(op, 'tournament_size'):
        params.append(f"k={op.tournament_size}")
    if hasattr(op, 'sp'): # For RankingLineal
         params.append(f"sp={op.sp}")
    if hasattr(op, 'replacement_rate'):
        params.append(f"rate={op.replacement_rate}")
    # Add other potential operator parameters here if needed
    if params:
        return f"{name}({', '.join(params)})"
    else:
        return name

# --- Ejecucion principal ---
if __name__ == '__main__':
    # --- Configuration ---
    TSP_FILE = "ALL_tsp/berlin52.tsp" # Example file
    matriz_costos = cargar_matriz(TSP_FILE)
    num_ciudades = matriz_costos.shape[0]

    # --- Parameters to benchmark ---
    TAMANIO_POBLACION = [50, 100, 200]
    TAMANIO_TORNEO = [3]
    PROBABILIDAD_DE_CRUCE = [0.8, 1.0]
    PROBABILIDAD_DE_MUTACION = [0.05, 0.1, .2]
    CANTIDAD_DE_GENERACIONES = [100, 200]
    CANTIDAD_EJECUCIONES = 10

    PARAMETRO_RANKING_LINEAL = [1.5, 2.0]
    PORCENTAJE_DE_REEMPLAZO = [0.5, 0.7]

    # --- Instantiate Operators (Do this once in the main process) ---
    # These objects will be pickled and sent to worker processes
    OPERADORES_SELECCION = [estrategiasSeleccionPadres.SeleccionPorTorneo(tamanio_del_torneo=k) for k in TAMANIO_TORNEO] \
                         + [estrategiasSeleccionPadres.SeleccionPorRankingLineal(sp=sp_val) for sp_val in PARAMETRO_RANKING_LINEAL]
    OPERADORES_CRUCE = [estrategiasCruce.CrucePMX(), estrategiasCruce.CruceBasadoEnArcos()]
    OPERADORES_MUTACION = [estrategiasMutacion.MutacionPorInversion(), estrategiasMutacion.MutacionPorInsersion()]
    OPERADORES_REEMPLAZO = [estrategiasSeleccionSobrevivientes.ReemplazoGeneracional()] \
                         + [estrategiasSeleccionSobrevivientes.ReemplazoSteadyState(replacement_rate=r) for r in PORCENTAJE_DE_REEMPLAZO]

    # --- Generate all combinations ---
    param_combinations = list(itertools.product(
        TAMANIO_POBLACION,
        PROBABILIDAD_DE_CRUCE,
        PROBABILIDAD_DE_MUTACION,
        CANTIDAD_DE_GENERACIONES,
        OPERADORES_SELECCION,
        OPERADORES_CRUCE,
        OPERADORES_MUTACION,
        OPERADORES_REEMPLAZO
    ))

    # --- Prepare tasks for parallel execution ---
    tasks = []
    for combo in param_combinations:
        # Create a dictionary for each task configuration
        config = {
            'matriz_costos': matriz_costos, # Pass the numpy array
            'tamanio_poblacion': combo[0],
            'probabilidad_de_cruce': combo[1],
            'probabilidad_de_mutacion': combo[2],
            'cantidad_de_generaciones': combo[3],
            'operador_seleccion': combo[4], # Pass operator instance
            'operador_cruce': combo[5],
            'operador_mutacion': combo[6],
            'operador_reemplazo': combo[7],
            'cantidad_ejecuciones': CANTIDAD_EJECUCIONES,
            # Store descriptions for easy logging later
            'desc_seleccion': get_operator_desc(combo[4]),
            'desc_cruce': get_operator_desc(combo[5]),
            'desc_mutacion': get_operator_desc(combo[6]),
            'desc_reemplazo': get_operator_desc(combo[7]),
        }
        tasks.append(config)

    # --- Execute in Parallel ---
    results = []
    # Use context manager for ProcessPoolExecutor
    # Defaults to os.cpu_count() workers, adjust if needed
    max_workers = os.cpu_count()
    print(f"\nStarting benchmark with {len(tasks)} combinations.")
    print(f"Running {CANTIDAD_EJECUCIONES} executions for each combination.")
    print(f"Using up to {max_workers} worker processes...")

    start_benchmark_time = time.perf_counter()

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store future objects
        future_to_config = {executor.submit(run_single_configuration, task): task for task in tasks}

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_config)):
            config_ran = future_to_config[future] # Get the config for this future
            try:
                # Get result tuple: (config_dict, avg_time, avg_cost, std_dev_cost)
                result_data = future.result()
                results.append(result_data)
                # Simple progress indicator
                print(f"Completed configuration {i + 1}/{len(tasks)}", end='\r')
            except Exception as exc:
                # Print exceptions raised in worker processes
                print(f'\nConfiguration generated an exception: {exc}')
                print(f"Problematic Config: Pop={config_ran['tamanio_poblacion']}, "
                      f"Cross={config_ran['desc_cruce']}, Mut={config_ran['desc_mutacion']}, etc.")
                # Optionally store None or the exception object
                # results.append((config_ran, None, None, None, exc))

    end_benchmark_time = time.perf_counter()
    total_benchmark_time = end_benchmark_time - start_benchmark_time
    print(f"\n\nBenchmark finished {len(results)} configurations in {total_benchmark_time:.2f} seconds.")

    # --- Write Results to File ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"benchmark_results_{timestamp}.txt"
    print(f"Writing results to {output_filename}...")

    # Sort results for consistent output (e.g., by avg cost)
    results.sort(key=lambda x: x[2]) # x[2] is avg_resultado

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"Genetic Algorithm Benchmark Results\n")
        f.write(f"TSP Problem: {TSP_FILE}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Configurations Tested: {len(results)} (out of {len(tasks)})\n")
        f.write(f"Executions per Configuration: {CANTIDAD_EJECUCIONES}\n")
        f.write(f"Total Benchmark Time: {total_benchmark_time:.2f} seconds\n")
        f.write("="*60 + "\n\n")

        for config, avg_time, avg_cost, std_dev_cost in results:
            f.write(f"Configuration:\n")
            f.write(f"  Population Size: {config['tamanio_poblacion']}\n")
            f.write(f"  Crossover Rate: {config['probabilidad_de_cruce']}\n")
            f.write(f"  Mutation Rate: {config['probabilidad_de_mutacion']}\n")
            f.write(f"  Generations: {config['cantidad_de_generaciones']}\n")
            f.write(f"  Selection: {config['desc_seleccion']}\n")
            f.write(f"  Crossover: {config['desc_cruce']}\n")
            f.write(f"  Mutation: {config['desc_mutacion']}\n")
            f.write(f"  Replacement: {config['desc_reemplazo']}\n")
            f.write(f"Results (Avg over {config['cantidad_ejecuciones']} runs):\n")
            f.write(f"  Avg Execution Time: {avg_time:.4f} seconds\n")
            f.write(f"  Avg Best Cost: {avg_cost:.4f}\n")
            f.write(f"  Std Dev Best Cost: {std_dev_cost:.4f}\n") # Added Std Dev
            f.write("-" * 60 + "\n")

    print("Results written successfully.")