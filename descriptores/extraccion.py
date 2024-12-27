import itertools
from datetime import datetime

from tqdm import tqdm
import fnmatch
import os
import arff
import cv2
import descriptores.histogramasgo as hist


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

def extraccion_batch(imagenes,opciones):
    resultado_total = []
    for opc in opciones:
        fich_dest = f"Resources/Datasets/histogramas_form1_{datetime.now().strftime('%Y%m%d%H%M%S')}.arff"
        resultado_local = [opc[0]]
        if opc[0] == "histogramas":
            resultado_local.append(opc[1])
            extraccion(imagenes, opc[0], fich_dest, histoptions=opc[1])
        else:
            resultado_local.append([h.__name__ for h in opc[1]])
            extraccion(imagenes, opc[0], fich_dest, formas=opc[1])

        resultado_local.append(fich_dest)
        resultado_total.append(resultado_local)
    return resultado_total

def extraccion(images, opciones, fichero_destino, **kwargs):  # Devuelve ARFF
    # Caso 1: Histogramas
    if opciones == "histogramas":
        histoptions = kwargs["histoptions"]

        # Añadir barra de progreso
        props = [
            hist.extraer_histogramas(images[e][0], histoptions[0], histoptions[1], histoptions[2])
            for e in tqdm(range(len(images)), desc="[Extracción] Extrayendo histogramas")
        ]

        arff_data = {
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props[0]))] + [("label", "NUMERIC")],
            "data": [props[e] + [int(images[e][1])] for e in tqdm(range(len(images)),desc='[Extracción] Generando archivo ARFF')],
            "description": "Descriptores HOG de una imagen",
            "relation": "hog_features",
        }

        print('[Extracción] Escribiendo archivo ARFF')
        with open(fichero_destino, 'w+') as f:
            arff.dump(arff_data, f)




    # Caso 2: Formas
    elif opciones == "formas":
        formas = kwargs["formas"]

        # Añadir barra de progreso
        props = []
        for imagen in tqdm(images, desc="[Extracción] Procesando formas"):
            prop_img = []
            for forma in formas:
                prop_img.append(forma(imagen[0]))
            prop_img = list(itertools.chain.from_iterable(prop_img))
            props.append(prop_img)




        arff_data = {
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props[0]))] + [("label", "NUMERIC")],
            "data": [props[e] + [int(images[e][1])] for e in tqdm(range(len(images)),desc='[Extracción] Generando archivo ARFF')],
            "description": "Descriptores formas de una imagen",
            "relation": "shape_features",
        }

        print('[Extracción] Escribiendo archivo ARFF')
        with open(fichero_destino, 'w+') as f:
            arff.dump(arff_data, f)



