import sys
from weka.core import jvm
from segmentacion import ejecucion_externa
from Herramientas.General import eliminar_archivos_no_usados, procesar_imagen, traducir_predicciones, eliminar_archivos, \
    contar_carpetas, obtener_pngs
from descriptores import extraccion, formas
from entrenamiento import main
from entrenamiento import *
from entrenamiento.ocr import predecir
from definitions import ROOT_DIR

# Limpiar directorios

eliminar_archivos(ROOT_DIR + "/Resources/Datasets")
eliminar_archivos(ROOT_DIR + "/Resources/Modelos")

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

jvm.start(packages=True, max_heap_size="8G",logging_level=0)

############## 1.Entrenamiento del modelo #############
print("Iniciando Entrenamiento...")
resultado_entrenamiento = main.obtener_modelo(opciones,classifiers,"/Resources/DatosRaw/ccnds2",raiz_proyecto)
input("Introduce cualquier carácter y pulsa enter para continuar a la sección de segmentación")


#Eliminar todos los modelos y datasets menos el mejor de todos
eliminar_archivos_no_usados(raiz_proyecto+"/Resources/Datasets",raiz_proyecto+"/Resources/Modelos",resultado_entrenamiento)
#####################################################

path_modelo = resultado_entrenamiento[4]
path_arff = ROOT_DIR+"/Resources/Temp/ocr.arff"


############## 2.Segmentación de las tarjetas #############
print("Segmentación de Tarjetas: Pulsar Flecha derecha para avanzar entre sets de tarjetas.")
ejecucion_externa()
input("Introduce cualquier carácter y pulsa enter para continuar al reconocimiento de carácteres: ")

############## 3.Reconocimiento de los carácteres #########
print("Iniciando OCR de las tarjetas...")
#Tarjeta 1
print("Tarjeta 1")

for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 1")):
    if len(obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 1/{reg}"))>0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 1/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 1/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))


#Tarjeta 2
print("Tarjeta 2")

for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 2")):
    if len(obtener_pngs(ROOT_DIR + f"/Resources/Segmentación/Output/IMG 2/{reg}")) > 0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 2/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 2/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))


#Tarjeta 3
print("Tarjeta 3")

for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 3")):
    if len(obtener_pngs(ROOT_DIR + f"/Resources/Segmentación/Output/IMG 3/{reg}")) > 0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 3/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 3/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))



#Tarjeta 4
print("Tarjeta 4")

for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 4")):
    if len(obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 4/{reg}"))>0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 4/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 4/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))



#Tarjeta 5
print("Tarjeta 5")
for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 5")):
    if len(obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 5/{reg}"))>0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 5/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 5/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))

#Tarjeta 6
print("Tarjeta 6")
for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 6")):
    if len(obtener_pngs(ROOT_DIR + f"/Resources/Segmentación/Output/IMG 6/{reg}")) > 0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 6/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 6/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))

#Tarjeta 7
print("Tarjeta 7")
for reg in range(0,contar_carpetas(ROOT_DIR+"/Resources/Segmentación/Output/IMG 7")):
    if len(obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 7/{reg}"))>0:
        print(f"Región {reg}")
        images = [ROOT_DIR + f"/Resources/Segmentación/Output/IMG 7/{reg}/{imagen}" for imagen in obtener_pngs(ROOT_DIR+f"/Resources/Segmentación/Output/IMG 7/{reg}")]
        img_cv = [procesar_imagen(image) for image in images]
        extraccion.extraccion_ocr(img_cv, resultado_entrenamiento[0], path_arff, histoptions=resultado_entrenamiento[1])
        prediccion_nums = predecir(path_modelo, path_arff)
        print(prediccion_nums)
        print(traducir_predicciones(prediccion_nums))

########## Apagar JVM
jvm.stop()