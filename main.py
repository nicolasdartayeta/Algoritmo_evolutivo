import numpy as np

def print_matrices():
    print(f"Matriz de costos\n")
    print(MATRIZ_COSTOS, f'\n')
    print(f'Soluciones\n')
    print(soluciones, f'\n')
    print(f'Costos de las soluciones\n')
    print(costo_soluciones, f'\n')
    print(f'Fitness de las soluciones\n')
    print(fitness_soluciones, f'\n')

def generar_soluciones_iniciales():
    # Se general permutaciones aleatorias de ciudades
    global soluciones
    rng = np.random.default_rng()

    for i in range(len(soluciones)):
        soluciones[i] = rng.permutation(CANTIDAD_CIUDADES)

def calcular_costo_solucion(nro_solucion):
    # Se calcula sumando los costos de recorrer todas las ciudades
    costo = 0
    for i in range(CANTIDAD_CIUDADES - 1):
        ciudad_a = soluciones[nro_solucion][i]
        ciudad_b = soluciones[nro_solucion][i + 1]
        costo += MATRIZ_COSTOS[ciudad_a][ciudad_b]

    return costo

def calcular_costos_soluciones():
    # Para cada solucion se calcula su costo
    global costo_soluciones

    for i in range(len(costo_soluciones)):
        costo_soluciones[i] = calcular_costo_solucion(i)

def calcular_fitness_soluciones():
    # Se calcula como el reciproco del costo
    global fitness_soluciones

    for i in range(len(fitness_soluciones)):
        fitness_soluciones[i] = 1 / costo_soluciones[i]

def seleccion_por_torneo(nro_integrantes_torneo):
    padres = np.empty((TAMANO_POBLACION, MATRIZ_COSTOS.shape[0]), dtype=np.int32)

    # Hay que generar todos los padres. Habra tantos torneos como padres
    #for i in range(TAMANO_POBLACION):




if __name__=='__main__':
    # Definir tamaño de la poblacion.
    # TODO que el tamaño venga de un argumento del programa o archivo de configuracion
    TAMANO_POBLACION = 10

    # TODO que el archivo de costos venga de un argumento del programa o archivo de configuracion
    costos_file = 'costos.csv'

    # Cargar costros a una matriz
    MATRIZ_COSTOS = np.loadtxt(costos_file, delimiter=',')

    # Cantidad de ciudades
    CANTIDAD_CIUDADES = MATRIZ_COSTOS.shape[0]

    # Crear matrices de solucione, fitness, y costos vacias
    soluciones = np.empty((TAMANO_POBLACION, MATRIZ_COSTOS.shape[0]), dtype=np.int32)
    fitness_soluciones = np.empty(TAMANO_POBLACION)
    costo_soluciones = np.empty(TAMANO_POBLACION)

    generar_soluciones_iniciales()
    calcular_costos_soluciones()
    calcular_fitness_soluciones()
    print_matrices()
    #seleccion_de_padres()
