import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# UTILIDADES
# =========================
def a_grises(img):
    """Convierte a escala de grises si es necesario"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# =========================
# HISTOGRAMA
# =========================
def calcular_histograma(imagen_gris):
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])
    return hist.flatten()

def histograma(img, titulo="Histograma"):
    img = a_grises(img)
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(0,256))
    plt.title(titulo)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.show()

def comparacion_histogramas(img_original, img_segmentada):
    img_original = a_grises(img_original)
    img_segmentada = a_grises(img_segmentada)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(img_original.ravel(), bins=256, range=(0,256))
    plt.title("Histograma original")
    plt.subplot(1,2,2)
    plt.hist(img_segmentada.ravel(), bins=256, range=(0,256))
    plt.title("Histograma segmentada")
    plt.tight_layout()
    plt.show()

# =========================
# MÉTODOS DE SEGMENTACIÓN
# =========================
def umbral_otsu(imagen):
    gris = a_grises(imagen)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

def umbral_media(imagen):
    gris = a_grises(imagen)
    umbral = np.mean(gris)
    binaria = (gris >= umbral).astype(np.uint8) * 255
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

def umbral_kapur(imagen):
    gris = a_grises(imagen)
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
        h1 = h1[h1>0]
        h2 = h2[h2>0]
        entropia = -np.sum(h1*np.log(h1)) - np.sum(h2*np.log(h2))
        if entropia > max_entropia:
            max_entropia = entropia
            umbral_optimo = t
    _, binaria = cv2.threshold(gris, umbral_optimo, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

def watershed_segmentacion(imagen):
    gris = a_grises(imagen)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
    fondo = cv2.dilate(apertura, kernel, iterations=3)
    dist = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
    _, objetos = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    objetos = np.uint8(objetos)
    desconocido = cv2.subtract(fondo, objetos)
    _, marcadores = cv2.connectedComponents(objetos)
    marcadores = marcadores + 1
    marcadores[desconocido==255] = 0
    imagen_color = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    marcadores = cv2.watershed(imagen_color, marcadores)
    resultado = np.zeros_like(gris)
    resultado[marcadores > 1] = 255
    return resultado

# =========================
# AJUSTE DE BRILLO / GAMMA
# =========================
def ecualizacion_uniforme(imagen):
    gris = a_grises(imagen)
    eq = cv2.equalizeHist(gris)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def ecualizacion_exponencial(imagen):
    gris = a_grises(imagen)
    exp = np.uint8(255 * (1 - np.exp(-gris / 255)))
    return cv2.cvtColor(exp, cv2.COLOR_GRAY2BGR)

def ecualizacion_rayleigh(imagen):
    gris = a_grises(imagen)
    ray = np.uint8(255 * np.sqrt(gris / 255))
    return cv2.cvtColor(ray, cv2.COLOR_GRAY2BGR)

def ecualizacion_hipercubica(imagen):
    gris = a_grises(imagen)
    hip = np.uint8(255 * (gris / 255)**4)
    return cv2.cvtColor(hip, cv2.COLOR_GRAY2BGR)

def ecualizacion_logaritmica(imagen):
    gris = a_grises(imagen)
    log = np.uint8(255 * np.log1p(gris) / np.log1p(255))
    return cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)

def correccion_gamma(imagen, gamma=1.5):
    gris = a_grises(imagen)
    gamma_img = np.power(gris/255.0, gamma) * 255
    return cv2.cvtColor(gamma_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# =========================
# DETECCIÓN DE FIGURAS GEOMÉTRICAS
# =========================
def detectar_figuras(img_input):
    """
    Detecta figuras geométricas en una imagen.
    Devuelve la imagen con contornos dibujados y un diccionario con el conteo.
    """

    # --- Asegurar imagen binaria ---
    if len(img_input.shape) == 3:
        gris = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    else:
        gris = img_input.copy()

    # Aplicar umbral Otsu si no está binarizada
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Encontrar contornos ---
    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

    conteo = {"Triangulo":0,"Cuadrado":0,"Rectangulo":0,"Pentagono":0,
              "Hexagono":0,"Heptagono":0,"Octagono":0,"Circulo":0,"Elipse":0,"Estrella":0}

    h, w = binaria.shape

    for cnt in contornos:
        area = cv2.contourArea(cnt)

        # --- Descarta contorno del borde completo ---
        if area < 50 or cv2.boundingRect(cnt)[2] >= w-5 or cv2.boundingRect(cnt)[3] >= h-5:
            continue

        peri = cv2.arcLength(cnt, True)
        epsilon = 0.03 * peri
        aprox = cv2.approxPolyDP(cnt, epsilon, True)
        v = len(aprox)

        figura = None
        if v == 3:
            figura = "Triangulo"
        elif v == 4:
            x, y, w_rect, h_rect = cv2.boundingRect(aprox)
            ar = w_rect / h_rect
            figura = "Cuadrado" if 0.9 <= ar <= 1.1 else "Rectangulo"
        elif v == 5:
            figura = "Pentagono"
        elif v == 6:
            figura = "Hexagono"
        elif v == 7:
            figura = "Heptagono"
        elif v == 8:
            figura = "Octagono"
        elif v > 8:
            circularidad = 4 * np.pi * area / (peri * peri)
            if circularidad > 0.8:
                figura = "Circulo"
            elif circularidad > 0.65:
                figura = "Elipse"
            else:
                figura = "Estrella"

        if figura:
            conteo[figura] += 1

            # --- Dibujar contorno ---
            cv2.drawContours(resultado, [aprox], -1, (0,255,0), 2)

            # --- Escribir nombre en el centroide ---
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.putText(resultado, figura, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return resultado, conteo


# =========================
# COMPARACIÓN DE SEGMENTACIONES
# =========================
def comparar_segmentaciones(img):
    """
    Devuelve un diccionario con resultados de cada segmentación
    y también genera la visualización comparativa.
    """
    img = a_grises(img)
    segs = {
        "Otsu": umbral_otsu(img),
        "Kapur": umbral_kapur(img),
        "Media": umbral_media(img)
    }

    plt.figure(figsize=(12,8))
    for i, (nombre, binaria) in enumerate(segs.items()):
        plt.subplot(2,2,i+1)
        plt.imshow(binaria, cmap="gray")
        plt.title(nombre)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return segs
