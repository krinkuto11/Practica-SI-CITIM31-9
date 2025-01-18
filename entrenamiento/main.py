from weka.classifiers import Classifier, Evaluation
from weka.core.classes import serialization_write, Random
from weka.core.converters import Loader
from weka.filters import Filter
from tqdm import tqdm
import sys
from Herramientas.General import eliminar_archivos_no_usados
from descriptores import formas
from weka.core import jvm
from descriptores import extraccion as ext
import pandas as pd
from Herramientas import General

def training(dataset, fichsalida,clasificador="weka.classifiers.trees.RandomForest",debug_level=0):
    resultados = [clasificador.split('.')[-1],dataset, fichsalida]
    # Carga de datos desde un archivo ARFF
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(dataset)
    if debug_level>0:print(f'[Entrenamiento] Cargado dataset: {dataset}')

    # Establecer la última columna como clase objetivo
    data.class_is_last()

    # Aplicar un filtro de normalización
    normalize = Filter(classname="weka.filters.unsupervised.attribute.Normalize")
    normalize.inputformat(data)
    data_normalized = normalize.filter(data)

    # Dividir los datos en 70% entrenamiento y 30% prueba
    train, test = data_normalized.train_test_split(70.0, Random(1))

    # Crear y configurar el clasificador
    classifier = Classifier(classname=clasificador)

    # Entrenar el modelo con los datos de entrenamiento
    classifier.build_classifier(train)

    # Evaluar el modelo en el conjunto de prueba
    evaluation = Evaluation(train)
    evaluation.test_model(classifier, test)


    # Mostrar los resultados
    if debug_level==2:print(evaluation.summary())

    resultados.append(evaluation.percent_correct)

    # Guardar el modelo en un archivo
    serialization_write(fichsalida, classifier)
    if debug_level>0:print(f"[Entrenamiento] Guardado modelo en: {fichsalida}")
    return resultados


def obtener_modelo(opciones,classifiers,path_recursos,raiz_proyecto):


    ########## Sección de entrenamiento ##########

    ##Extraer imágenes y sus etiquetas de la carpeta especificada
    images = ext.obtener_imagenes(raiz_proyecto+path_recursos)
    ##Especificar opciones para la generación del dataset

    ##Generación del dataset
    ##Ajustamos los outputs del terminal según el nivel de debug 0-2
    debug_level = 0
    data = ext.extraccion_batch(images,opciones,debug_level=debug_level)
    contador = 0
    output = []
    for clasificador in classifiers:
        for file in tqdm(data,desc=f"[Entrenamiento] Generando los modelos con clasificador {clasificador.split('.')[-1]}"):
            fichsalida = f"{raiz_proyecto}/Resources/Modelos/modelo-{clasificador.split('.')[-1]}-{contador}.model"
            output.append(training(file[2],fichsalida,clasificador=clasificador,debug_level=debug_level))
            output[contador] = file + output[contador]
            contador += 1

    ##Generación de la tabla con Pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)

    tabla = pd.DataFrame({
        "Tipo": [e[0] for e in output],
        "Opciones": [i[1] for i in output],
        "Dataset producido": [s[2] for s in output],
        "Clasificador": [o[3] for o in output],
        "Modelo producido": [r[5] for r in output],
        "% Correcto": [u[6] for u in output]
    })

    tabla_ordenada = tabla.sort_values(by="% Correcto", ascending=False,ignore_index=True)

    resultado = tabla_ordenada.iloc[0].tolist()
    print(tabla_ordenada)
    print(f"El modelo más apto para el uso del OCR es: {tabla_ordenada.loc[0,'Modelo producido']}")


    return resultado

