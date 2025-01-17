import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import glob

def eliminar_archivos_no_usados(carpeta_datasets, carpeta_modelos, resultado):
    # Extraer los archivos que deben conservarse
    arff_permitido = os.path.abspath(resultado[2])
    modelo_permitido = os.path.abspath(resultado[4])

    # Eliminar archivos .arff no permitidos
    for archivo in os.listdir(carpeta_datasets):
        ruta_completa = os.path.abspath(os.path.join(carpeta_datasets, archivo))
        if archivo.endswith(".arff") and ruta_completa != arff_permitido:
            os.remove(ruta_completa)
            print(f"Eliminado: {ruta_completa}")

    # Eliminar archivos .model no permitidos
    for archivo in os.listdir(carpeta_modelos):
        ruta_completa = os.path.abspath(os.path.join(carpeta_modelos, archivo))
        if archivo.endswith(".model") and ruta_completa != modelo_permitido:
            os.remove(ruta_completa)
            print(f"Eliminado: {ruta_completa}")

def eliminar_archivos(ruta_carpeta):
    archivos = glob.glob(os.path.join(ruta_carpeta, "*"))
    for archivo in archivos:
        if os.path.isfile(archivo):  # Verifica que sea un archivo
            os.remove(archivo)

def procesar_imagen(imagen):
    imagen_gris = cv2.imread(imagen, cv2.IMREAD_GRAYSCALE)
    imagen_bin = cv2.adaptiveThreshold(
        imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return imagen_bin


def mostrar_imagen(imagen):
    imagen_rgb = cv2.cvtColor(imagen[0], cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))  # Tamaño de la figura
    plt.imshow(imagen_rgb)
    plt.axis('off')  # Oculta los ejes
    plt.title(imagen[1], fontsize=12, color='blue')  # Pie de imagen
    plt.show()


def tqdm_condicional(iterable,desc, debug_level):
    if debug_level > 0:  # Ajusta el valor según el nivel en el que quieras mostrar la barra
        return tqdm(iterable,desc)  # Muestra la barra de progreso
    else:
        return iterable

