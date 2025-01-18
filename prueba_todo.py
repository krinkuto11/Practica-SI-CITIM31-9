import sys

from weka.core import jvm

from segmentacion import ejecucion_externa
from Herramientas import General
from Herramientas.General import eliminar_archivos_no_usados, procesar_imagen, traducir_predicciones
from descriptores import extraccion, formas
from entrenamiento import main
import os
from entrenamiento import *
from entrenamiento.ocr import predecir

# Limpiar directorios

General.eliminar_archivos(sys.path[1] + "/Resources/Datasets")
General.eliminar_archivos(sys.path[1] + "/Resources/Modelos")

###########Parámetros para el entrenamiento###########

opciones = [("histogramas", [8, 8, 2]), ("histogramas", [9, 8, 2]), ("histogramas", [12, 8, 2]),
            ("histogramas", [12, 16, 2]), ("histogramas", [16, 8, 2]), ("histogramas", [16, 16, 2]),
            ('formas', [formas.hu_moments]),
            ('formas', [formas.aspect_ratio]), ('formas', [formas.compactness]), ('formas', [formas.euler_number]),
            ('formas', [formas.hu_moments, formas.aspect_ratio]), ('formas', [formas.hu_moments, formas.compactness]),
            ('formas', [formas.hu_moments, formas.euler_number]), ('formas', [formas.aspect_ratio, formas.compactness]),
            ('formas', [formas.aspect_ratio, formas.euler_number]),
            ('formas', [formas.compactness, formas.euler_number]),
            ('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness]),
            ('formas', [formas.hu_moments, formas.aspect_ratio, formas.euler_number]),
            ('formas', [formas.hu_moments, formas.compactness, formas.euler_number]),
            ('formas', [formas.aspect_ratio, formas.compactness, formas.euler_number]),
            ('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness, formas.euler_number])
            ]
classifiers = [
    "weka.classifiers.trees.RandomForest",
    "weka.classifiers.trees.J48",
    "weka.classifiers.bayes.NaiveBayes"
]

raiz_proyecto = sys.path[1]
dataset = "/Resources/DatosRaw/ccnds2"
###########Parámetros para el entrenamiento###########

jvm.start(packages=True, max_heap_size="8G",logging_level=0)

############## 1.Entrenamiento del modelo #############
print("Iniciando Entrenamiento...")
resultado_entrenamiento = main.obtener_modelo(opciones,classifiers,"/Resources/DatosRaw/ccnds2",raiz_proyecto)
input("Introduce cualquier carácter y pulsa enter para continuar a la sección de segmentación")


#Eliminar todos los modelos y datasets menos el mejor de todos
eliminar_archivos_no_usados(raiz_proyecto+"/Resources/Datasets",raiz_proyecto+"/Resources/Modelos",resultado_entrenamiento)
#####################################################

path_modelo = resultado_entrenamiento[4]
path_arff = sys.path[1]+"/Resources/Temp/ocr.arff"


############## 2.Segmentación de las tarjetas #############
print("Segmentación de Tarjetas: Pulsar Flecha derecha para avanzar entre sets de tarjetas.")
ejecucion_externa()
input("Introduce cualquier carácter y pulsa enter para continuar al reconocimiento de carácteres: ")

############## 3.Reconocimiento de los carácteres #########
print("Iniciando OCR de las tarjetas...")
#Tarjeta 1
print("Tarjeta 1")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 1/char_{}.png".format(i) for i in range(1,49)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))

#Tarjeta 2
print("Tarjeta 2")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 2/char_{}.png".format(i) for i in range(1,44)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))

#Tarjeta 3
print("Tarjeta 3")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 3/char_{}.png".format(i) for i in range(1,57)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))


#Tarjeta 4
print("Tarjeta 4")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 4/char_{}.png".format(i) for i in range(1,38)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))


#Tarjeta 5
print("Tarjeta 5")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 5/char_{}.png".format(i) for i in range(1,6)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))

#Tarjeta 6
print("Tarjeta 6")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 6/char_{}.png".format(i) for i in range(1,39)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))

#Tarjeta 7
print("Tarjeta 7")
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 7/char_{}.png".format(i) for i in range(1,33)]
img_cv = [procesar_imagen(image) for image in images]
extraccion.extraccion_ocr(img_cv,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
prediccion_nums = predecir(path_modelo, path_arff)
print(prediccion_nums)
print(traducir_predicciones(prediccion_nums))

########## Apagar JVM
jvm.stop()