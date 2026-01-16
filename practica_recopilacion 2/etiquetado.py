import cv2
import numpy as np

def extraer_regiones_umbral(imagen, umbral_min, umbral_max):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral_min, umbral_max, cv2.THRESH_BINARY)
    return binaria


def etiquetar_regiones(imagen_binaria):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagen_binaria, connectivity=8
    )

    output = np.zeros(
        (imagen_binaria.shape[0], imagen_binaria.shape[1], 3),
        dtype=np.uint8
    )

    for i in range(1, num_labels):
        mask = labels == i
        output[mask] = np.random.randint(0, 255, size=3)

    return output
def etiquetar_patron(imagen_binaria, patron_binario, umbral_similitud=0.2):
    """
    Reconoce patrones por similitud de forma y dibuja la silueta
    sobre las regiones que coinciden.
    """

    # Asegurar escala de grises
    if len(imagen_binaria.shape) == 3:
        gris = cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_binaria.copy()

    salida = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    # Contorno del patrón
    contornos_patron, _ = cv2.findContours(
        patron_binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos_patron:
        return salida

    contorno_patron = max(contornos_patron, key=cv2.contourArea)
    hu_patron = cv2.HuMoments(cv2.moments(contorno_patron)).flatten()

    # Contornos de la imagen
    contornos, _ = cv2.findContours(
        gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        hu_cnt = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # Comparación de similitud
        distancia = cv2.matchShapes(
            contorno_patron, cnt, cv2.CONTOURS_MATCH_I1, 0
        )

        if distancia < umbral_similitud:
            # Dibujar silueta del patrón reconocido
            cv2.drawContours(salida, [cnt], -1, (0, 255, 0), 2)

    return salida
