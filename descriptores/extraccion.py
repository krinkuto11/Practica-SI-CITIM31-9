import csv
import fnmatch
import os
import arff
import cv2
import descriptores.histogramasgo as hist

def obtener_imagenes(path_recursos):
    path_raiz = os.getcwd()
    #path_recursos = "Resources/DatosRaw/ccnds"
    path_dataset = os.path.join(path_raiz, path_recursos)
    imagenes = [
        (cv2.imread(os.path.join(path_dataset, str(numero), archivo)), numero)
        for numero in range(10)
        for archivo in fnmatch.filter(os.listdir(os.path.join(path_dataset, str(numero))), f"{numero}_*.png")
        if cv2.imread(os.path.join(path_dataset, str(numero), archivo)) is not None
    ]
    return imagenes

def extraccion(images,opciones,fichero_destino,**kwargs): #Devuelve ARFF

    #Caso 1:
    if opciones == "histogramas":
        histoptions = kwargs["histoptions"]

        props = [hist.extraer_histogramas(images[e][0],histoptions[0],histoptions[1],histoptions[2]) for e in range(len(images))]
        arff_data = {
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props[0]))] + [("label", "NUMERIC")],
            "data": [props[e] + [int(images[e][1])] for e in range(len(images))],
            "description": "Descriptores HOG de una imagen",
            "relation": "hog_features",
        }
        print(arff.dumps(arff_data))
        with open(fichero_destino, 'w+') as f:
           arff.dump(arff_data, f)

    elif opciones == "formas":
        formas = kwargs["formas"]
        props = [[forma(imagen).flatten() for forma in formas] for imagen in images]
        arff_data = {
            "description": "Descriptores de formas de una imagen",
            "relation": "shape_features",
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props))],
            "data": [(props[e]) for e in range(len(images))],
        }
        with open(fichero_destino, 'w') as f:
            arff.dump(arff_data, f)

