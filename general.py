import sys

from Herramientas import General
from Herramientas.General import eliminar_archivos_no_usados
from descriptores import extraccion, formas
from entrenamiento import *
import os
from entrenamiento import entrenamiento

# Limpiar directorios

General.eliminar_archivos(sys.path[1] + "/Resources/Datasets")
General.eliminar_archivos(sys.path[1] + "/Resources/Modelos")

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
dataset = raiz_proyecto+"/Resources/DatosRaw/ccnds2"

resultado_entrenamiento = entrenamiento.main.obtener_modelo(opciones,classifiers,"/Resources/DatosRaw/ccnds2",raiz_proyecto)
eliminar_archivos_no_usados(raiz_proyecto+"/Resources/Datasets",raiz_proyecto+"/Resources/Modelos",resultado_entrenamiento)
images = [sys.path[1]+"/Resources/Segmentación/Output/IMG 1/char_{}.png".format(i) for i in range(1,50)]
path_modelo = sys.path[1]+resultado_entrenamiento[4]
path_arff = sys.path[1]+"/Resources/Temp/ocr.arff"
extraccion.extraccion_ocr(images,resultado_entrenamiento[0],path_arff,histoptions=resultado_entrenamiento[1])
print(os.path.exists(sys.path[1]+"/Resources/Modelos/modelo-NaiveBayes-42.model"))
