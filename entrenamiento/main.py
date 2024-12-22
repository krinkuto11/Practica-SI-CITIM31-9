import os
from datetime import datetime

import arff

import descriptores.formas
from descriptores import extraccion as ext
import pprint
import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import serialization_read, serialization_write, Random

def main():

    imagenes = ext.obtener_imagenes("Resources/DatosRaw/ccnds")
    print(f"Extraídas {len(imagenes)} imágenes.")
    ##Probamos con distintas cosas:

    ##Prueba N1: Solamente Histogramas
    print("Prueba 1: Histogramas. Opciones -> Orientaciones: 2, Píxeles/Celda: 2, Celdas/Bloque: 2")
    fichero_destino = f"Resources/Datasets/histogramas_hist_{datetime.now().strftime('%Y%m%d%H%M%S')}.arff"
    ext.extraccion(images=imagenes, opciones="histogramas",fichero_destino=fichero_destino,histoptions=[2,2,2])
    print(f'[Extracción] Generación de Dataset completada: {fichero_destino}')
    print('[Entrenamiento] Comenzando entrenamiento del modelo')
    #training(fichero_destino)
    ##Prueba N2: Hu + Ratio de Aspecto + Compacidad
    print("Prueba 2: Formas. Opciones -> Momentos Hu, Ratio de Aspecto, Compacidad")
    fichero_destino2 = f"Resources/Datasets/histogramas_form1_{datetime.now().strftime('%Y%m%d%H%M%S')}.arff"
    formas = [descriptores.formas.hu_moments, descriptores.formas.aspect_ratio, descriptores.formas.compactness]
    ext.extraccion(images=imagenes, opciones='formas',fichero_destino=fichero_destino2,formas=formas)
    print(f'[Extracción] Generación de Dataset completada: {fichero_destino2}')
    print('[Entrenamiento] Comenzando entrenamiento del modelo')
    training(fichero_destino2)


    ##Prueba N3: Euler + Ratio de Aspecto + Compacidad
    ##Prueba N4: Euler + Hu
    ##Prueba N5: Todos los descriptores


    #Entrenamiento con primer fichero
    training(fichero_destino)

def training(ficheroentrenamiento):
    # Inicia la JVM para usar Weka
    jvm.start(packages=True)

    # Carga de datos desde un archivo ARFF
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(ficheroentrenamiento)
    print('[Entrenamiento] Cargado dataset')

    # Establecer la última columna como clase objetivo
    data.class_is_last()

    # Aplicar un filtro de normalización
    normalize = Filter(classname="weka.filters.unsupervised.attribute.Normalize")
    normalize.inputformat(data)
    data_normalized = normalize.filter(data)

    # Dividir los datos en 70% entrenamiento y 30% prueba
    train, test = data_normalized.train_test_split(70.0, Random(1))

    # Crear y configurar el clasificador
    classifier = Classifier(classname="weka.classifiers.trees.RandomForest")

    # Entrenar el modelo con los datos de entrenamiento
    classifier.build_classifier(train)

    # Evaluar el modelo en el conjunto de prueba
    evaluation = Evaluation(train)
    evaluation.test_model(classifier, test)

    # Mostrar los resultados
    print(evaluation.summary())  # Resumen de precisión

    # Guardar el modelo en un archivo
    serialization_write("modelo_identificacion.model", classifier)

    # Cargar el modelo
    #loaded_classifier = serialization_read("modelo_identificacion.model")

    # Cargar nuevos datos para predicción
    #new_data = loader.load_file("nuevos_datos.arff")
    #new_data.class_is_last()

    # Realizar predicciones
    #for index, inst in enumerate(new_data):
    #    pred = loaded_classifier.classify_instance(inst)
    #   print(f"Instancia {index + 1}: Clase predicha = {new_data.class_attribute.value(int(pred))}")

    #Finalizar JVM
    jvm.stop()
