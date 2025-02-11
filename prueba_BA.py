import bosque_aleatorio as ba
import math


def main():
    datos = [
        {"atributo1": 1, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 2, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 3, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 4, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 1, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 4, "clase": "positiva"},
    ]
    M = 5
    target = "clase"
    max_profundidad = 3
    acc_nodo = 1
    min_ejemplos = 0
    num_atributos = len(datos[0]) - 1
    variables_seleccionadas = math.floor(math.sqrt(num_atributos))
    bosque = ba.entrena_bosque_aleatorio(
        datos,
        M,
        target,
        max_profundidad,
        acc_nodo,
        min_ejemplos,
        variables_seleccionadas,
    )
    instancia = {"atributo1": 1, "atributo2": 1}
    prediccion = ba.predice_bosque_aleatorio(bosque, instancia, target)
    print(prediccion)


if __name__ == "__main__":
    main()
