import cv2
import numpy as np

def obtener_kernel(tipo_kernel, tamaño):
    """
    Devuelve el kernel según el tipo y tamaño especificados
    """
    if tipo_kernel == "cuadrado":
        return np.ones((tamaño, tamaño), np.uint8)
    elif tipo_kernel == "eliptico":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamaño, tamaño))
    elif tipo_kernel == "cruz":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamaño, tamaño))
    else:
        return np.ones((3, 3), np.uint8)  # Por defecto

def erosion(imagen, tipo_kernel="cuadrado", tamaño=3):
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.erode(imagen, kernel, iterations=1)

def dilatacion(imagen, tipo_kernel="cuadrado", tamaño=3):
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.dilate(imagen, kernel, iterations=1)

def apertura(imagen, tipo_kernel="cuadrado", tamaño=3):
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

def cierre(imagen, tipo_kernel="cuadrado", tamaño=3):
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)

def gradiente_morfologico(imagen, tipo_kernel="cuadrado", tamaño=3):
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)