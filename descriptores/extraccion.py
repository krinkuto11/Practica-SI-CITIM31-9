import csv

import arff

import histogramasgo

def extraccion(image,opciones,**kwargs): #Devuelve ARFF

    #Caso 1:
    if opciones == "histogramas":
        histoptions = kwargs["histoptions"]
        props = histogramasgo.extraer_histogramas(image,histoptions[0],histoptions[1],histoptions[2],histoptions[3])

        arff_data = {
            "description": "Descriptores HOG de una imagen",
            "relation": "hog_features",
            "attributes": [(f"feature{i + 1}", "REAL") for i in range(len(props))],
            "data": [props.tolist()],
        }

        with open('hog_features.arff', 'w') as f:
            arff.dump(arff_data, f)


