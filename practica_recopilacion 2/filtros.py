import cv2
import numpy as np

# =========================
# AGREGAR RUIDO
# =========================

def ruido_sal_pimienta(imagen, cantidad=0.02): 
    """Agrega ruido sal y pimienta a la imagen"""
    salida = imagen.copy()
    filas, columnas = imagen.shape
    total_pixeles = int(cantidad * filas * columnas)

    for _ in range(total_pixeles):
        x = np.random.randint(0, filas)
        y = np.random.randint(0, columnas)
        salida[x, y] = 255 if np.random.rand() < 0.5 else 0
    return salida


def ruido_gaussiano(imagen, media=0, sigma=25):
    """Agrega ruido gaussiano a la imagen"""
    gauss = np.random.normal(media, sigma, imagen.shape)
    ruido = imagen + gauss
    ruido = np.clip(ruido, 0, 255)
    return ruido.astype(np.uint8)


# =========================
# FILTROS LINEALES
# =========================

def filtro_promediador(imagen, ksize=5):
    """Filtro promediador (blur simple)"""
    return cv2.blur(imagen, (ksize, ksize))


def filtro_promediador_pesado(imagen, ksize=5):
    """Filtro promediador con kernel personalizado"""
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(imagen, -1, kernel)


def filtro_gaussiano(imagen, ksize=5):
    """Filtro Gaussiano para suavizado"""
    return cv2.GaussianBlur(imagen, (ksize, ksize), 0)


def filtro_laplaciano(imagen):
    """Filtro Laplaciano para detección de bordes"""
    lap = cv2.Laplacian(imagen, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap


# =========================
# FILTROS NO LINEALES
# =========================

def filtro_mediana(imagen, ksize=5):
    """Filtro de mediana - excelente para ruido sal y pimienta"""
    return cv2.medianBlur(imagen, ksize)


def filtro_moda(imagen, k=3):
    """Filtro de moda - valor más frecuente en la ventana"""
    salida = np.zeros_like(imagen)
    pad = k // 2
    img_pad = np.pad(imagen, pad_width=pad, mode="edge")

    for i in range(salida.shape[0]):
        for j in range(salida.shape[1]):
            ventana = img_pad[i:i+k, j:j+k].flatten()
            valores, conteos = np.unique(ventana, return_counts=True)
            moda = valores[np.argmax(conteos)]
            salida[i, j] = moda

    return salida


def filtro_maximo(imagen, k=3):
    """Filtro de máximo - dilata las áreas claras"""
    salida = imagen.copy()
    pad = k // 2
    for i in range(pad, imagen.shape[0] - pad):
        for j in range(pad, imagen.shape[1] - pad):
            ventana = imagen[i-pad:i+pad+1, j-pad:j+pad+1]
            salida[i, j] = np.max(ventana)
    return salida


def filtro_minimo(imagen, k=3):
    """Filtro de mínimo - dilata las áreas oscuras"""
    salida = imagen.copy()
    pad = k // 2
    for i in range(pad, imagen.shape[0] - pad):
        for j in range(pad, imagen.shape[1] - pad):
            ventana = imagen[i-pad:i+pad+1, j-pad:j+pad+1]
            salida[i, j] = np.min(ventana)
    return salida