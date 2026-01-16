# ruido_y_filtros.py
# Ruido y filtrado en imágenes
# ESCOM - Procesamiento Digital de Imágenes

import cv2
import numpy as np

# =========================
# AGREGAR RUIDO
# =========================

def ruido_sal_pimienta(imagen, cantidad=0.02):
    """
    Agrega ruido sal y pimienta a una imagen en escala de grises
    """
    salida = imagen.copy()

    if len(imagen.shape) == 3:
        salida = cv2.cvtColor(salida, cv2.COLOR_BGR2GRAY)

    filas, columnas = salida.shape
    total_pixeles = int(cantidad * filas * columnas)

    for _ in range(total_pixeles):
        x = np.random.randint(0, filas)
        y = np.random.randint(0, columnas)
        salida[x, y] = 255 if np.random.rand() < 0.5 else 0

    return salida


def ruido_gaussiano(imagen, media=0, sigma=25):
    """
    Agrega ruido gaussiano a una imagen
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    gauss = np.random.normal(media, sigma, imagen.shape)
    ruido = imagen.astype(np.float32) + gauss
    ruido = np.clip(ruido, 0, 255)

    return ruido.astype(np.uint8)


# =========================
# FILTROS LINEALES
# =========================

def filtro_promediador(imagen, ksize=5):
    return cv2.blur(imagen, (ksize, ksize))


def filtro_promediador_pesado(imagen, ksize=5):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(imagen, -1, kernel)


def filtro_gaussiano(imagen, ksize=5):
    return cv2.GaussianBlur(imagen, (ksize, ksize), 0)


def filtro_laplaciano(imagen):
    lap = cv2.Laplacian(imagen, cv2.CV_64F)
    return np.uint8(np.absolute(lap))


def filtro_sobel(imagen):
    sx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sx, sy)


def filtro_prewitt(imagen):
    kx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

    ky = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    px = cv2.filter2D(imagen, -1, kx)
    py = cv2.filter2D(imagen, -1, ky)

    return px + py


def filtro_roberts(imagen):
    krx = np.array([[1, 0],
                    [0, -1]])

    kry = np.array([[0, 1],
                    [-1, 0]])

    rx = cv2.filter2D(imagen, -1, krx)
    ry = cv2.filter2D(imagen, -1, kry)

    return rx + ry


def filtro_canny(imagen, th1=100, th2=200):
    return cv2.Canny(imagen, th1, th2)


# =========================
# FILTROS NO LINEALES
# =========================

def filtro_mediana(imagen, ksize=5):
    return cv2.medianBlur(imagen, ksize)


def filtro_moda(imagen, k=3):
    salida = np.zeros_like(imagen)
    pad = k // 2
    img_pad = np.pad(imagen, pad_width=pad, mode="edge")

    for i in range(salida.shape[0]):
        for j in range(salida.shape[1]):
            ventana = img_pad[i:i+k, j:j+k].flatten()
            valores, conteos = np.unique(ventana, return_counts=True)
            salida[i, j] = valores[np.argmax(conteos)]

    return salida


def filtro_maximo(imagen, k=3):
    salida = imagen.copy()
    pad = k // 2

    for i in range(pad, imagen.shape[0] - pad):
        for j in range(pad, imagen.shape[1] - pad):
            salida[i, j] = np.max(imagen[i-pad:i+pad+1, j-pad:j+pad+1])

    return salida


def filtro_minimo(imagen, k=3):
    salida = imagen.copy()
    pad = k // 2

    for i in range(pad, imagen.shape[0] - pad):
        for j in range(pad, imagen.shape[1] - pad):
            salida[i, j] = np.min(imagen[i-pad:i+pad+1, j-pad:j+pad+1])

    return salida

def filtro_bilateral(imagen, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(imagen, d, sigmaColor, sigmaSpace)

def filtro_paso_altas(imagen):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(imagen, -1, kernel)
