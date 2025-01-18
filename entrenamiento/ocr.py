import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
import math  # Para detectar NaN

def predecir(modelo_path, arff_path):

    # Iniciar JVM de WEKA

    try:
        # Cargar el modelo entrenado
        deserialized = Classifier.deserialize(modelo_path)

        # Verificar si es una tupla y extraer el primer elemento
        classifier = deserialized[0] if isinstance(deserialized, tuple) else deserialized


        # Cargar los datos ARFF
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(arff_path)
        data.class_is_last()  # Asegurar que la última columna sea la clase

        # Verificar que el archivo ARFF tenga instancias
        if data.num_instances == 0:
            raise ValueError("El archivo ARFF no contiene instancias para predecir.")

        # Lista para almacenar predicciones
        predicciones = []

        # Iterar sobre todas las instancias
        for i, instance in enumerate(data):
            prediction = classifier.classify_instance(instance)  # Predicción numérica

            # Manejar posibles NaN en la predicción
            if math.isnan(prediction):
                class_value = "ERROR: NaN en predicción"
            else:
                class_value = data.class_attribute.value(int(prediction))  # Convertir índice a nombre de clase

            predicciones.append(class_value)
            print(f"Instancia {i}: Predicción = {class_value}")

        return predicciones

    finally:
      print("OCR Completo")

# Ejemplo de uso:
# resultado = predecir("modelo.model", "imagenes.arff")
# print("Todas las predicciones:", resultado)
