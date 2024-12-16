from weka.core.classes import serialization_read
from weka.core.converters import Loader, Saver
from descriptores import *


def ocr(image,modelo):

    loader = Loader(classname="weka.core.converters.ArffLoader")
    loaded_classifier = serialization_read(modelo)

    new_data = loader.load_file("nuevos_datos.arff")
    new_data.class_is_last()

    # Realizar predicciones
    for index, inst in enumerate(new_data):
        pred = loaded_classifier.classify_instance(inst)
        print(f"Instancia {index + 1}: Clase predicha = {new_data.class_attribute.value(int(pred))}")
