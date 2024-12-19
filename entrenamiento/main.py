import os

import cv2
import weka.core.jvm as jvm
import fnmatch
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random
from weka.core.classes import serialization_read, serialization_write

def main():
    path_raiz = os.getcwd()
    path_recursos = "Resources/DatosRaw/ccnds"
    path_dataset = os.path.join(path_raiz, path_recursos)
    #print(path_dataset)

    #print(range(len(fnmatch.filter(os.listdir(f"{path_dataset}/0"), '*.png'))))

    imagenes = [[(cv2.imread(f"{path_dataset}/{numero}/{numero}_{i}.png"), numero) for i in range(1,len(fnmatch.filter(os.listdir(f"{path_dataset}/{numero}"), '*.png'))+1)] for numero in range(10)]


    ##Probamos con distintas cosas:
    ##Prueba N1: Solamente Histogramas

    #fichero_destino = "Resources/Datasets/histogramas.arff"
    #ext.extraccion(fichero_destino)
    ##Prueba N2: Hu + Ratio de Aspecto + Compacidad
    ##Prueba N3: Euler + Ratio de Aspecto + Compacidad
    ##Prueba N4: Euler + Hu
    ##Prueba N5: Todos los descriptores

def obtener_imagenes(path_recursos):
    path_raiz = os.getcwd()
    #path_recursos = "Resources/DatosRaw/ccnds"
    path_dataset = os.path.join(path_raiz, path_recursos)
    # print(path_dataset)

    # print(range(len(fnmatch.filter(os.listdir(f"{path_dataset}/0"), '*.png'))))

    imagenes = [[(cv2.imread(f"{path_dataset}/{numero}/{numero}_{i}.png"), numero) for i in
                 range(1, len(fnmatch.filter(os.listdir(f"{path_dataset}/{numero}"), '*.png')) + 1)] for numero in
                range(10)]


def training(ficheroentrenamiento):
    # Inicia la JVM para usar Weka
    jvm.start(packages=True)

    # Carga de datos desde un archivo ARFF
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("ruta_al_archivo.arff")

    # Establecer la última columna como clase objetivo
    data.class_is_last()

    # Aplicar un filtro de normalización
    normalize = Filter(classname="weka.filters.unsupervised.attribute.Normalize")
    normalize.inputformat(data)
    data_normalized = normalize.filter(data)

    # Dividir los datos en 70% entrenamiento y 30% prueba
    train, test = data.train_test_split(70.0, Random(1))

    # Crear y configurar el clasificador
    classifier = Classifier(classname="weka.classifiers.trees.RandomForest")

    # Entrenar el modelo con los datos de entrenamiento
    classifier.build_classifier(train)

    # Evaluar el modelo en el conjunto de prueba
    evaluation = Evaluation(train)
    evaluation.test_model(classifier, test)

    # Mostrar los resultados
    print(evaluation.summary())  # Resumen de precisión
    print(evaluation.class_details())  # Detalles por clase
    print(evaluation.confusion_matrix)  # Matriz de confusión


    # Guardar el modelo en un archivo
    serialization_write("modelo_identificacion.model", classifier)

    # Cargar el modelo
    loaded_classifier = serialization_read("modelo_identificacion.model")

    # Cargar nuevos datos para predicción
    new_data = loader.load_file("nuevos_datos.arff")
    new_data.class_is_last()

    # Realizar predicciones
    for index, inst in enumerate(new_data):
        pred = loaded_classifier.classify_instance(inst)
        print(f"Instancia {index + 1}: Clase predicha = {new_data.class_attribute.value(int(pred))}")

    #Finalizar JVM
    jvm.stop()