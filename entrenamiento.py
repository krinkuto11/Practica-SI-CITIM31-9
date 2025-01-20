import sys
from weka.core import jvm
import entrenamiento.main
from Herramientas import General
from descriptores import formas
from entrenamiento import *
from definitions import ROOT_DIR

jvm.start(packages=True, max_heap_size="8G")
##########Limpieza de resultados anteriores###########
General.eliminar_archivos(ROOT_DIR + "/Resources/Datasets")
General.eliminar_archivos(ROOT_DIR + "/Resources/Modelos")
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
raiz_proyecto = ROOT_DIR
dataset = "/Resources/DatosRaw/ccnds2"
###########Parámetros para el entrenamiento###########

#Ejecuta el entrenamiento del modelo
entrenamiento.main.obtener_modelo(opciones,classifiers,"/Resources/DatosRaw/ccnds2",raiz_proyecto)


####### Apagar JVM ######
jvm.stop()