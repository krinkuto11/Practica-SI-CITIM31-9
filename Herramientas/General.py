import cv2
from matplotlib import pyplot as plt


def mostrar_imagen(imagen):
    imagen_rgb = cv2.cvtColor(imagen[0], cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))  # Tamaño de la figura
    plt.imshow(imagen_rgb)
    plt.axis('off')  # Oculta los ejes
    plt.title(imagen[1], fontsize=12, color='blue')  # Pie de imagen
    plt.show()


