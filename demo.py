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
jvm.start(packages=True)
##Extraer imágenes y sus etiquetas de la carpeta especificada
images = ext.obtener_imagenes("Resources/DatosRaw/ccnds2")
##Especificar opciones para la generación del dataset
opciones = [("histogramas",[8,8,2]),("histogramas",[9,8,2]),("histogramas",[12,8,2]),("histogramas",[12,16,2]),("histogramas",[16,8,2]),("histogramas",[16,16,2]),('formas', [formas.hu_moments]),
            ('formas', [formas.aspect_ratio]),('formas', [formas.compactness]),('formas', [formas.euler_number]),('formas', [formas.hu_moments, formas.aspect_ratio]),('formas', [formas.hu_moments, formas.compactness]),
            ('formas', [formas.hu_moments, formas.euler_number]),('formas', [formas.aspect_ratio, formas.compactness]),('formas', [formas.aspect_ratio, formas.euler_number]),('formas', [formas.compactness, formas.euler_number]),
            ('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness]),('formas', [formas.hu_moments, formas.aspect_ratio, formas.euler_number]),('formas', [formas.hu_moments, formas.compactness, formas.euler_number]),
            ('formas', [formas.aspect_ratio, formas.compactness, formas.euler_number]),('formas', [formas.hu_moments, formas.aspect_ratio, formas.compactness, formas.euler_number])
]
##Generación del dataset
data = ext.extraccion_batch(images,opciones)
contador = 0
output = []
for file in tqdm(data,desc="[Entrenamiento] Generando los modelos"):
    fichsalida = f"Resources/Modelos/modelo{contador}.model"
    output.append(training(file[2],fichsalida))
    contador += 1

##Generación de la tabla con Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

tabla = pd.DataFrame({
    "Tipo": [e[0] for e in data],
    "Opciones": [i[1] for i in data],
    "Dataset producido": [o[0] for o in output],
    "Modelo producido": [r[1] for r in output],
    "Coeficiente de correlación": [u[2] for u in output],
})

tabla_ordenada = tabla.sort_values(by="Coeficiente de correlación", ascending=False)

print(tabla_ordenada)

print(f"El modelo más apto para el uso del OCR es: {tabla_ordenada['Modelo producido'].max()}")

##Paramos la máquina virtual de WEKA
jvm.stop()

