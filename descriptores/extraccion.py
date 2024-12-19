import csv

import arff

import histogramasgo

def extraccion(images,opciones,**kwargs): #Devuelve ARFF

    #Caso 1:
    if opciones == "histogramas":
        histoptions = kwargs["histoptions"]
        props = [(histogramasgo.extraer_histogramas(images[e],histoptions[0],histoptions[1],histoptions[2],histoptions[3]) for e in range(len(images)))]

        arff_data = {
            "description": "Descriptores HOG de una imagen",
            "relation": "hog_features",
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props))],
            "data": [(props[e]) for e in range(len(images))],
        }

        with open('hog_features.arff', 'w') as f:
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

