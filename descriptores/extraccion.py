import itertools
from datetime import datetime

from tqdm import tqdm
import fnmatch
import os
import arff
import cv2
import descriptores.histogramasgo as hist
from Herramientas.General import tqdm_condicional


def obtener_imagenes(path_recursos):
    path_raiz = os.getcwd()
    path_dataset = os.path.join(path_raiz, path_recursos)
    print(f'[Extracción] Obteniendo imágenes del directorio {path_raiz}')
    imagenes = [
        (cv2.adaptiveThreshold(cv2.imread(os.path.join(path_dataset, str(numero), archivo),cv2.IMREAD_GRAYSCALE), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2), numero)
        for numero in range(36)
        for archivo in fnmatch.filter(os.listdir(os.path.join(path_dataset, str(numero))), f"{numero}_*.png")
        if cv2.imread(os.path.join(path_dataset, str(numero), archivo),cv2.IMREAD_GRAYSCALE) is not None
    ]
    return imagenes

def extraccion_batch(images, opciones, debug_level=0):
    resultado_total = []
    for opc in tqdm(opciones,desc="[Extraccion] Generando datasets por cada opción"):
        fich_dest = f"Resources/Datasets/histogramas_form1_{datetime.now().strftime('%Y%m%d%H%M%S')}.arff"
        resultado_local = [opc[0]]
        if opc[0] == "histogramas":
            resultado_local.append(opc[1])
            extraccion(images, opc[0], fich_dest, histoptions=opc[1], debug_level=debug_level)
        else:
            resultado_local.append([h.__name__ for h in opc[1]])
            extraccion(images, opc[0], fich_dest, formas=opc[1], debug_level=debug_level)
        resultado_local.append(fich_dest)
        resultado_total.append(resultado_local)
    return resultado_total

def extraccion(images, opciones, fichero_destino,debug_level=0, **kwargs):
    label_range = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
    # Devuelve ARFF
    # Caso 1: Histogramas
    if opciones == "histogramas":
        histoptions = kwargs["histoptions"]

        # Añadir barra de progreso
        props = [
            hist.extraer_histogramas(images[e][0], histoptions[0], histoptions[1], histoptions[2])
            for e in tqdm_condicional(range(len(images)), desc="[Extracción] Extrayendo histogramas",debug_level=debug_level)
        ]

        arff_data = {
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props[0]))] + [("label", label_range)],
            "data": [props[e] + [int(images[e][1])] for e in tqdm_condicional(range(len(images)),desc='[Extracción] Generando archivo ARFF',debug_level=debug_level)],
            "description": "Descriptores HOG de una imagen",
            "relation": "hog_features",
        }

        if debug_level>0:print('[Extracción] Escribiendo archivo ARFF')
        with open(fichero_destino, 'w+') as f:
            arff.dump(arff_data, f)




    # Caso 2: Formas
    elif opciones == "formas":
        formas = kwargs["formas"]

        # Añadir barra de progreso
        props = []
        for imagen in tqdm_condicional(images, desc="[Extracción] Procesando formas",debug_level=debug_level):
            prop_img = []
            for forma in formas:
                prop_img.append(forma(imagen[0]))
            prop_img = list(itertools.chain.from_iterable(prop_img))
            props.append(prop_img)




        arff_data = {
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props[0]))] + [("label", label_range)],
            "data": [props[e] + [int(images[e][1])] for e in tqdm_condicional(range(len(images)),desc='[Extracción] Generando archivo ARFF',debug_level=debug_level)],
            "description": "Descriptores formas de una imagen",
            "relation": "shape_features",
        }

        if debug_level>0:print('[Extracción] Escribiendo archivo ARFF')
        with open(fichero_destino, 'w+') as f:
            arff.dump(arff_data, f)



