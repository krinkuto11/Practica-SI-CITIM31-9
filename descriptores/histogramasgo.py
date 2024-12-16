# Módulo con los métodos relacionados con la extracción de
# los Histogramas de Gradientes Orientados.
# Más en: https://www.geeksforgeeks.org/hog-feature-visualization-in-python-using-skimage
from skimage import color
from skimage.feature import hog
from skimage import data, exposure, io

def extraer_histogramas(imagen,orientaciones, pixelspercell,cellsperblock,multichannel):
    image = imagen.grayscale()
    features = hog(image, orientations=orientaciones, pixels_per_cell=(pixelspercell, pixelspercell),
                              cells_per_block=(cellsperblock, cellsperblock), multichannel=multichannel)
    