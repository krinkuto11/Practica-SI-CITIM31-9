import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


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