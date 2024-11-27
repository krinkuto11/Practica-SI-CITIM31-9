# Módulo con los métodos relacionados con la extracción de descriptores
# de formas

#Imports
import cv2
from skimage.measure import label, regionprops

#Función combinada
def extract_shape_features(image):
    binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    ar = aspect_ratio(binary_image)
    comp = compactness(binary_image)
    hu = hu_moments(binary_image)
    euler = euler_number(binary_image)
    feature_vector = [ar, comp] + hu + [euler]
    return feature_vector
#Funciones individuales
def aspect_ratio(image):
    x, y, w, h = cv2.boundingRect(image)
    return w / h if h != 0 else 0

def compactness(image):
    area = cv2.countNonZero(image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    return (perimeter**2) / area if area != 0 else 0

def euler_number(image):
    labeled_image = label(image)
    euler = sum([prop.euler_number for prop in regionprops(labeled_image)])
    return euler

def hu_moments(image):
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments).flatten()
    return hu.tolist()


