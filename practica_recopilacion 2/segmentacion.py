import cv2
import numpy as np

# =========================
# HISTOGRAMA
# =========================
def calcular_histograma(imagen_gris):
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0,256])
    return hist.flatten()

# =========================
# UMBRAL OTSU
# =========================
def umbral_otsu(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

# =========================
# UMBRAL MEDIA
# =========================
def umbral_media(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    umbral = np.mean(gris)
    binaria = (gris >= umbral).astype(np.uint8) * 255
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

# =========================
# MÉTODO DE KAPUR
# =========================
def umbral_kapur(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gris, bins=256, range=(0,256))
    total = gris.size

    max_entropia = -1
    umbral_optimo = 0

    for t in range(1, 255):
        p1 = np.sum(hist[:t]) / total
        p2 = np.sum(hist[t:]) / total
        if p1 == 0 or p2 == 0:
            continue

        h1 = hist[:t] / np.sum(hist[:t])
        h2 = hist[t:] / np.sum(hist[t:])

        e1 = -np.sum(h1 * np.log(h1 + 1e-10))
        e2 = -np.sum(h2 * np.log(h2 + 1e-10))

        entropia = p1 * e1 + p2 * e2
        if entropia > max_entropia:
            max_entropia = entropia
            umbral_optimo = t

    binaria = (gris > umbral_optimo).astype(np.uint8) * 255
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

# =========================
# AJUSTE DE BRILLO
# =========================
def ecualizacion_uniforme(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gris)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def ecualizacion_exponencial(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    exp = np.uint8(255 * (1 - np.exp(-gris / 255)))
    return cv2.cvtColor(exp, cv2.COLOR_GRAY2BGR)

def ecualizacion_rayleigh(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ray = np.uint8(255 * np.sqrt(gris / 255))
    return cv2.cvtColor(ray, cv2.COLOR_GRAY2BGR)

def ecualizacion_hipercubica(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    hip = np.uint8(255 * (gris / 255) ** 4)
    return cv2.cvtColor(hip, cv2.COLOR_GRAY2BGR)

def ecualizacion_logaritmica(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    log = np.uint8(255 * np.log1p(gris) / np.log1p(255))
    return cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)

def correccion_gamma(imagen, gamma=1.5):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gamma_img = np.power(gris / 255.0, gamma) * 255
    return cv2.cvtColor(gamma_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)



import cv2
import numpy as np

def watershed_segmentacion(imagen):
    # Asegurar escala de grises
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # Umbralización (Otsu)
    _, binaria = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Eliminación de ruido
    kernel = np.ones((3, 3), np.uint8)
    apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fondo seguro
    fondo = cv2.dilate(apertura, kernel, iterations=3)

    # Transformada de distancia
    dist = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)

    # Objetos seguros
    _, objetos = cv2.threshold(
        dist,
        0.5 * dist.max(),
        255,
        0
    )
    objetos = np.uint8(objetos)

    # Región desconocida
    desconocido = cv2.subtract(fondo, objetos)

    # Marcadores
    _, marcadores = cv2.connectedComponents(objetos)
    marcadores = marcadores + 1
    marcadores[desconocido == 255] = 0

    # Watershed
    imagen_color = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    marcadores = cv2.watershed(imagen_color, marcadores)

    # Imagen binaria final
    resultado = np.zeros_like(gris)
    resultado[marcadores > 1] = 255

    return resultado
