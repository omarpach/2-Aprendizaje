import arboles_numericos as an
import random


def entrena_bosque_aleatorio(
    datos: list[dict[str, float | int]],
    M: int,
    target: str,
    max_profundidad: int,
    acc_nodo: int,
    min_ejemplos: int,
    variables_seleccionadas: int,
) -> list[an.NodoN]:
    subconjuntos = separar_datos(datos, M)
    bosque: list[an.NodoN] = []

    # Para cada subconjunto de datos, entrena un árbol
    for subconjunto in subconjuntos:
        arbol = an.entrena_arbol(
            subconjunto,
            target,
            max_profundidad,
            acc_nodo,
            min_ejemplos,
            variables_seleccionadas,
        )

        # Agrega el arbol al bosque
        bosque.append(arbol)
    return bosque


def separar_datos(
    datos: list[dict[str, float | int]], M: int
) -> list[list[dict[str, float | int]]]:
    subconjuntos = []

    # Repite M veces
    for _ in range(M):
        # Selecciona un subconjunto de datos con reemplazo
        subconjunto = random.choices(datos, k=len(datos))

        # Agrega el subconjunto a la lista
        subconjuntos.append(subconjunto)
    return subconjuntos
