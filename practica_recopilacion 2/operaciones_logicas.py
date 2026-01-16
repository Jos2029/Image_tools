import cv2
import numpy as np

def operacion_and(imagen1, imagen2):
    return cv2.bitwise_and(imagen1, imagen2)

def operacion_or(imagen1, imagen2):
    return cv2.bitwise_or(imagen1, imagen2)

def operacion_xor(imagen1, imagen2):
    return cv2.bitwise_xor(imagen1, imagen2)

def operacion_not(imagen):
    return cv2.bitwise_not(imagen)