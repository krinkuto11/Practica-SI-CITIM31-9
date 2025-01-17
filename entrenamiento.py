from tqdm import tqdm
from descriptores import formas
from weka.core import jvm
from descriptores import extraccion as ext
import pandas as pd
from entrenamiento.main import training

#Aquí se deberá realizar una demostración de todas las funciones de
#realizadas en el proyecto.


#Sección segmentación


########## Sección de entrenamiento ##########
##Iniciar la máquina virtual de WEKA
jvm.start(packages=True,max_heap_size="8G")
##Extraer imágenes y sus etiquetas de la carpeta especificada
images = ext.obtener_imagenes("Resources/DatosRaw/ccnds2")
##Especificar opciones para la generación del dataset
opciones = [("histogramas",[8,8,2]),("histogramas",[9,8,2]),("histogramas",[12,8,2]),("histogramas",[12,16,2]),("histogramas",[16,8,2]),("histogramas",[16,16,2]),('formas', [formas.hu_moments]),
            ('formas', [formas.aspect_ratio]),('formas', [formas.compactness]),('formas', [formas.euler_number]),('formas', [formas.hu_moments, formas.aspect_ratio]),('formas', [formas.hu_moments, formas.compactness]),
            ('formas', [formas.hu_moments, formas.euler_number]),('formas', [formas.aspect_ratio, formas.compactness]),('formas', [formas.aspect_ratio, formas.euler_number]),('formas', [formas.compactness, formas.euler_number]),
            ('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness]),('formas', [formas.hu_moments, formas.aspect_ratio, formas.euler_number]),('formas', [formas.hu_moments, formas.compactness, formas.euler_number]),
            ('formas', [formas.aspect_ratio, formas.compactness, formas.euler_number]),('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness, formas.euler_number])
]
classifiers = [
    "weka.classifiers.trees.RandomForest",
    "weka.classifiers.trees.J48",
    "weka.classifiers.bayes.NaiveBayes"
]
##Generación del dataset
##Ajustamos los outputs del terminal según el nivel de debug 0-2
debug_level = 0
data = ext.extraccion_batch(images,opciones,debug_level=debug_level)
contador = 0
output = []
for clasificador in classifiers:
    for file in tqdm(data,desc=f"[Entrenamiento] Generando los modelos con clasificador {clasificador.split('.')[-1]}"):
        fichsalida = f"Resources/Modelos/modelo-{clasificador.split('.')[-1]}-{contador}.model"
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

print(tabla_ordenada)

print(f"El modelo más apto para el uso del OCR es: {tabla_ordenada.loc[0,'Modelo producido']}")

##Paramos la máquina virtual de WEKA
jvm.stop()

