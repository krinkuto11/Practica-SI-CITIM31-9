# Módulo con los métodos relacionados con la extracción de
# los Histogramas de Gradientes Orientados.
# Más en: https://www.geeksforgeeks.org/hog-feature-visualization-in-python-using-skimage
import cv2
from skimage.feature import hog


def extraer_histogramas(imagen,orientaciones, pixelspercell,cellsperblock,tamaño=(50,66)):
    imagen_redimensionada = cv2.resize(imagen, tamaño)
    imagen_redimensionada = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_RGB2GRAY)
    features = hog(imagen_redimensionada, orientations=orientaciones, pixels_per_cell=(pixelspercell, pixelspercell),
                              cells_per_block=(cellsperblock, cellsperblock),feature_vector=True)
    return features.tolist()