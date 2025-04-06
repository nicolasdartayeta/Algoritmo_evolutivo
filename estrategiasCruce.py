import abc
from collections import Counter
import random
import numpy as np

class EstrategiaDeCruce(abc.ABC):
    @abc.abstractmethod
    def cruzar(self, padres: np.ndarray, probabilidad_de_cruce: float) -> np.ndarray:
        """Cruza pares de padres para generar hijos."""
        pass

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
        """Funcion que realiza realiza el cruce y produce un hijo (version optimizada)."""
        tamanio = padre_1.shape[0]
        hijo = -np.ones(tamanio, dtype=padre_1.dtype) # Initialize with -1

        # --- Optimización: Crear mapa de posiciones para padre_2 ---
        # Esto mapea cada ciudad en padre_2 a su índice. Costo O(N) una sola vez.
        posicion_en_p2_map = {ciudad: indice for indice, ciudad in enumerate(padre_2)}
        # -----------------------------------------------------------

        # 1. Elegir dos puntos de cruce aleatorios
        punto_cruce_1, punto_cruce_2 = sorted(self.rng.choice(tamanio, 2, replace=False))

        # 2. Copiar segmento del padre_1 al hijo
        hijo[punto_cruce_1:punto_cruce_2+1] = padre_1[punto_cruce_1:punto_cruce_2+1]

        # 3. Mapear y colocar elementos conflictivos de padre_2 (dentro del segmento)
        for i in range(punto_cruce_1, punto_cruce_2 + 1):
            # Si el elemento de padre_2 NO está en el segmento ya copiado de padre_1
            if padre_2[i] not in hijo[punto_cruce_1:punto_cruce_2+1]:
                valor_a_copiar = padre_2[i] # Valor de p2 que necesita un lugar
                valor_desplazado = padre_1[i] # Valor de p1 que está en el hijo en la pos i

                # --- Optimización: Usar mapa para encontrar la posición ---
                # Encontrar dónde está el valor_desplazado en padre_2 usando el mapa. Costo O(1) avg.
                try:
                    posicion_en_p2 = posicion_en_p2_map[valor_desplazado]
                except KeyError:
                     # Seguridad: El valor de p1 debería estar en p2 si son permutaciones
                     print(f"Error Lógico: valor_desplazado {valor_desplazado} (de p1) no encontrado en el mapa de p2 {padre_2}")
                     raise ValueError("Inconsistencia entre padres durante PMX")
                # -------------------------------------------------------

                # Buscar una posición vacía (-1) en hijo siguiendo la cadena de mapeo
                posicion_hijo_destino = posicion_en_p2
                while hijo[posicion_hijo_destino] != -1:
                    valor_en_posicion_hijo = hijo[posicion_hijo_destino]

                    # --- Optimización: Usar mapa para seguir la cadena ---
                    # Encontrar dónde está este nuevo valor en padre_2 usando el mapa. Costo O(1) avg.
                    try:
                        posicion_hijo_destino = posicion_en_p2_map[valor_en_posicion_hijo]
                    except KeyError:
                         # Seguridad: El valor intermedio debería estar en p2
                         print(f"Error Lógico: valor_en_posicion_hijo {valor_en_posicion_hijo} no encontrado en el mapa de p2 {padre_2}")
                         raise ValueError("Inconsistencia en cadena PMX")
                    # -------------------------------------------------------

                # Colocar el valor_a_copiar en la posición vacía encontrada
                hijo[posicion_hijo_destino] = valor_a_copiar

        # 4. Llenar las posiciones restantes en el hijo con los valores de padre_2
        for i in range(tamanio):
            if hijo[i] == -1:  # Si la posición aún está vacía después del paso 3
                hijo[i] = padre_2[i]

        # Verificacion
        elementos_unicos = np.unique(hijo) # np.unique también ordena, útil para comparar si fuera necesario
        assert len(elementos_unicos) == tamanio, f"PMX resulto en {len(elementos_unicos)} ciudades unicas ({tamanio} esperadas). Hijo: {hijo}, Unicos: {elementos_unicos}"
        assert -1 not in hijo, f"PMX fallo en llenar todas las posiciones. Hijo: {hijo}"

        return hijo

class CruceBasadoEnArcos(EstrategiaDeCruce):
    def __init__(self):
        self.rng = np.random.default_rng() # Para la probabilidad de cruce

    def _construir_mapa_adyacencias_lista(self, p1: np.ndarray, p2: np.ndarray) -> dict[int, list]:
        """Construye el mapa de adyacencias usando LISTAS para vecinos."""
        cantidad_ciudades = len(p1)
        mapa = {}
        ciudades = np.arange(cantidad_ciudades)
        # Precalcular mapas de posiciones para eficiencia
        pos_p1_map = {ciudad: idx for idx, ciudad in enumerate(p1)}
        pos_p2_map = {ciudad: idx for idx, ciudad in enumerate(p2)}

        for ciudad in ciudades:
            # Vecinos en p1 (lista)
            idx_p1 = pos_p1_map[ciudad]
            vecinos_p1 = [p1[idx_p1 - 1], p1[(idx_p1 + 1) % cantidad_ciudades]]

            # Vecinos en p2 (lista)
            idx_p2 = pos_p2_map[ciudad]
            vecinos_p2 = [p2[idx_p2 - 1], p2[(idx_p2 + 1) % cantidad_ciudades]]

            # Obtener o inicializar la lista actual y extenderla
            lista_actual = mapa.get(ciudad, [])
            lista_actual.extend(vecinos_p1)
            lista_actual.extend(vecinos_p2)
            mapa[ciudad] = lista_actual # Guardar la lista extendida
        return mapa

    def _eliminar_referencias_lista(self, mapa_ady: dict[int, list], ciudad_a_eliminar: int):
        """Elimina TODAS las ocurrencias de una ciudad de las listas de vecinos."""
        for ciudad_key in mapa_ady:
            # Usar list comprehension para filtrar y reasignar la lista
            mapa_ady[ciudad_key] = [vecino for vecino in mapa_ady[ciudad_key] if vecino != ciudad_a_eliminar]

    def _generar_hijo_erx(self, mapa_ady_original: dict[int, list], num_ciudades: int, ciudades_disponibles: set) -> np.ndarray:
        """Genera un hijo ERX priorizando vecinos comunes (usando listas)."""
        # Copia profunda del mapa para no modificar el original entre llamadas
        mapa_ady = {k: v.copy() for k, v in mapa_ady_original.items()}
        hijo = np.full(num_ciudades, -1, dtype=int)
        visitados = set()
        idx_hijo = 0

        # 1. Elegir nodo inicial al azar
        actual = self.rng.choice(list(ciudades_disponibles)) # rng puede elegir de listas
        hijo[idx_hijo] = actual
        visitados.add(actual)
        idx_hijo += 1
        # 2. Eliminar referencias al nodo inicial
        self._eliminar_referencias_lista(mapa_ady, actual)

        # Generar todos los hijos
        while idx_hijo < num_ciudades:
            # Obtener la lista de vecinos actual (puede tener duplicados)
            vecinos_lista_actual = mapa_ady.get(actual, [])

            siguiente = -1 # Inicializar la variable para el siguiente nodo

            if not vecinos_lista_actual:
                # 4. Dead End: No hay vecinos no visitados para el nodo actual
                no_visitados_global = list(ciudades_disponibles - visitados)
                if not no_visitados_global:
                    # No deberían quedar nodos si idx_hijo < num_ciudades
                    print(f"ERROR ERX: Dead end sin nodos globales restantes en paso {idx_hijo}")
                    break # Salir del bucle si algo va muy mal
                siguiente = self.rng.choice(no_visitados_global) # Elegir al azar de los restantes
            else:
                # 3. Examinar lista de arcos y seleccionar siguiente
                # Contar ocurrencias para identificar vecinos comunes
                counts = Counter(vecinos_lista_actual)
                # Un vecino es común si aparece más de una vez en la lista original combinada
                vecinos_comunes = [ciudad for ciudad, count in counts.items() if count > 1]

                if vecinos_comunes:
                    # • Prioridad: Elegir al azar uno de los vecinos comunes
                    siguiente = random.choice(vecinos_comunes) # Usar random.choice para listas
                else:
                    # • Sin comunes: Elegir el vecino (único) con la lista de adyacencias actual más corta
                    mejor_vecino = -1
                    min_len_lista = float('inf')

                    # Considerar solo los vecinos únicos para esta elección
                    vecinos_unicos_no_visitados = list(counts.keys())
                    random.shuffle(vecinos_unicos_no_visitados) # Barajar para desempate aleatorio

                    for vecino in vecinos_unicos_no_visitados:
                        # Longitud de la lista de vecinos ACTUAL de este candidato
                        longitud_lista_vecino = len(mapa_ady.get(vecino, [])) # Longitud de su lista en el mapa *actual*

                        if longitud_lista_vecino < min_len_lista:
                            min_len_lista = longitud_lista_vecino
                            mejor_vecino = vecino
                        # El desempate es por el shuffle previo y el '<'

                    siguiente = mejor_vecino
                    if siguiente == -1 and vecinos_unicos_no_visitados:
                         # Fallback si algo raro pasa y no se eligió mejor_vecino
                         siguiente = vecinos_unicos_no_visitados[0]


            # --- Control de seguridad ---
            if siguiente == -1:
                 print(f"ERROR ERX: No se pudo determinar el siguiente nodo para {actual}. "
                       f"Visitados: {visitados}, Idx: {idx_hijo}")
                 # Forzar elección aleatoria si falla la lógica anterior
                 no_visitados_global = list(ciudades_disponibles - visitados)
                 if not no_visitados_global: break # Salir si no hay más opción
                 siguiente = self.rng.choice(no_visitados_global)

            # --- Actualizar estado ---
            actual = siguiente
            hijo[idx_hijo] = actual
            visitados.add(actual)
            idx_hijo += 1

            # 2. Eliminar referencias al nodo recién añadido de todas las listas
            self._eliminar_referencias_lista(mapa_ady, actual)


        # Verificaciones finales
        assert -1 not in hijo, f"ERX fallo en llenar todas las posiciones. Hijo: {hijo}"
        assert len(np.unique(hijo)) == num_ciudades, f"ERX produjo duplicados o faltantes ({len(np.unique(hijo))} unicos). Hijo: {hijo}"

        return hijo

    def cruzar(self, parents: np.ndarray, probabilidad_de_cruce: float) -> np.ndarray:
        num_parents, num_cities = parents.shape
        offspring = np.empty((num_parents, num_cities), dtype=int)
        num_offspring_generated = 0
        cities_available = set(range(num_cities))
        indices = np.arange(num_parents)

        for i in range(0, num_parents, 2):
            idx1 = indices[i]
            if i + 1 >= num_parents:
                if num_offspring_generated < num_parents:
                    offspring[num_offspring_generated] = parents[idx1].copy()
                    num_offspring_generated += 1
                continue
            idx2 = indices[i+1]
            padre_a = parents[idx1]
            padre_b = parents[idx2]
            hijo_a, hijo_b = None, None

            if self.rng.random() < probabilidad_de_cruce:
                tabla_adyacencias = self._construir_mapa_adyacencias_lista(padre_a, padre_b)
                hijo_a = self._generar_hijo_erx(tabla_adyacencias, num_cities, cities_available)
                hijo_b = padre_a.copy() if self.rng.choice([0,1]) == 0 else padre_b.copy()
            else:
                hijo_a = padre_a.copy()
                hijo_b = padre_b.copy()

            # Añadir hijos
            if num_offspring_generated < num_parents:
                offspring[num_offspring_generated] = hijo_a
                num_offspring_generated += 1
            if num_offspring_generated < num_parents:
                offspring[num_offspring_generated] = hijo_b
                num_offspring_generated += 1

        return offspring[:num_offspring_generated]
