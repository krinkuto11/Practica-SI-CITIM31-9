import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier


def predecir(modelo_path, arff_path):
    """
    Carga un modelo WEKA entrenado y predice la clase de la primera instancia en un archivo ARFF.

    :param modelo_path: Ruta al archivo del modelo (.model)
    :param arff_path: Ruta al archivo ARFF con los datos a predecir
    :return: Predicción de la primera instancia en el archivo ARFF
    """
    # Iniciar JVM de WEKA
    jvm.start()

    try:
        # Cargar el modelo entrenado
        classifier = Classifier.deserialize(modelo_path)

        # Cargar los datos ARFF
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(arff_path)
        data.class_is_last()  # Asegurar que la última columna sea la clase

        # Predecir la primera instancia
        prediction = classifier.classify_instance()

        # Obtener el nombre de la clase predicha
        class_value = data.class_attribute.value(int(prediction))

        return class_value

    finally:
        # Detener la JVM al finalizar
        jvm.stop()

# Ejemplo de uso:
# resultado = predecir("modelo.model", "imagen.arff")
# print(f"Predicción: {resultado}")