from datetime import datetime

from weka.core import jvm

from descriptores import extraccion as ext
import pandas as pd
from entrenamiento.main import training

#Aquí se deberá realizar una demostración de todas las funciones de
#realizadas en el proyecto.


#Sección segmentación


#Sección de entrenamiento
jvm.start(packages=True)
images = ext.obtener_imagenes("Resources/DatosRaw/ccnds2")
opciones = [("histogramas",[8,8,2]),("histogramas",[9,8,2]),("histogramas",[12,8,2]),("histogramas",[12,16,2]),("histogramas",[16,8,2]),("histogramas",[16,16,2])]
stats = ext.extraccion_batch(images,opciones)
contador = 0
correlations = []
for stat in stats:
    fichsalida = f"Resources/Modelos/modelo{contador}.model"
    correlations.append(training(stat[2],fichsalida))
    contador += 1

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
tabla = pd.DataFrame({
    "Tipo": [e[0] for e in stats],
    "Opciones": [i[1] for i in stats],
    "Dataset producido": [o[0] for o in correlations],
    "Modelo producido": [r[1] for r in correlations],
    "Coeficiente de correlación": [u[2] for u in correlations],
})

tabla_ordenada = tabla.sort_values(by="Coeficiente de correlación", ascending=True)

print(tabla)
jvm.stop()

#1. Cargar Dataset
#   Estructura del dataset: lista->[path,número correspondiente a la imagen]
#   1.1. Conversión a archivo ARFF -> Función: Descriptores/extraccion: extraccion(imagen, opciones)



