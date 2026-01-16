# morfologia_binaria.py
# Morfología matemática BINARIA completa
# ESCOM - Procesamiento Digital de Imágenes

import cv2
import numpy as np


def obtener_kernel(tipo_kernel, tamaño):
    if tipo_kernel == "cuadrado":
        return np.ones((tamaño, tamaño), np.uint8)
    elif tipo_kernel == "eliptico":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamaño, tamaño))
    elif tipo_kernel == "cruz":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamaño, tamaño))
    else:
        return np.ones((3, 3), np.uint8)


def binarizar(imagen, umbral=128):
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return binaria


def erosion(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.erode(img, kernel, iterations=1)


def dilatacion(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.dilate(img, kernel, iterations=1)


def apertura(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def cierre(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def gradiente(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)



def frontera(imagen, tipo_kernel="cuadrado", tamaño=3):
    img = binarizar(imagen)
    kernel = obtener_kernel(tipo_kernel, tamaño)
    erosionada = cv2.erode(img, kernel)
    return cv2.subtract(img, erosionada)


def hit_or_miss(imagen, kernel_hit=None, kernel_miss=None):
    if kernel_hit is None:
        kernel_hit = np.array([[1,1],
                               [1,1]], dtype=np.uint8)
    if kernel_miss is None:
        kernel_miss = np.array([[0,0],
                                [0,0]], dtype=np.uint8)

    img_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape)==3 else imagen
    return cv2.morphologyEx(img_bin, cv2.MORPH_HITMISS, kernel_hit)



def adelgazamiento(imagen):
    img = binarizar(imagen)
    esqueleto = np.zeros(img.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        erosionada = cv2.erode(img, kernel)
        dilatada = cv2.dilate(erosionada, kernel)
        temp = cv2.subtract(img, dilatada)
        esqueleto = cv2.bitwise_or(esqueleto, temp)
        img = erosionada.copy()

        if cv2.countNonZero(img) == 0:
            break

    return esqueleto


def esqueleto(imagen):
    img = binarizar(imagen)
    skel = np.zeros(img.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        erosionada = cv2.erode(img, kernel)
        apertura_img = cv2.morphologyEx(erosionada, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(erosionada, apertura_img)
        skel = cv2.bitwise_or(skel, temp)
        img = erosionada.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel

def aislamiento(imagen, tamaño_kernel=3):
    img = binarizar(imagen)
    kernel = np.ones((tamaño_kernel, tamaño_kernel), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
