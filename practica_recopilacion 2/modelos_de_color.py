import cv2
import numpy as np

# ---------------------------------
# PREPROCESAMIENTO BÁSICO
# ---------------------------------

def binarizar_imagen(imagen, umbral=128):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

def binarizar_imagen_umbral(imagen, umbral):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

def escala_grises(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

# ---------------------------------
# MODELOS DE COLOR
# ---------------------------------

def rgb_a_hsv(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

def rgb_a_cmyk(imagen):
    rgb = imagen.astype(float) / 255.0
    k = 1 - np.max(rgb, axis=2)

    c = (1 - rgb[:, :, 2] - k) / (1 - k + 1e-7)
    m = (1 - rgb[:, :, 1] - k) / (1 - k + 1e-7)
    y = (1 - rgb[:, :, 0] - k) / (1 - k + 1e-7)

    cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)
    return cmyk

# ---------------------------------
# SEPARACIÓN DE CANALES RGB
# ---------------------------------

def mostrar_canales_rgb(imagen):
    """
    Muestra tres ventanas independientes:
    Canal Rojo, Verde y Azul, cada uno en su color real
    """

    # Separar canales (OpenCV usa BGR)
    b, g, r = cv2.split(imagen)

    # Crear imágenes en color para cada canal
    canal_rojo = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    canal_verde = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    canal_azul = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])

    # Mostrar las tres ventanas
    cv2.imshow("Canal Rojo (R)", canal_rojo)
    cv2.imshow("Canal Verde (G)", canal_verde)
    cv2.imshow("Canal Azul (B)", canal_azul)

    # NO usar destroyAllWindows
    cv2.waitKey(1)


# ---------------------------------
# PSEUDOCOLOR (MAPA PASTEL)
# ---------------------------------

def pseudocolor_pastel(imagen):
    """
    Aplica un mapa de color pastel a una imagen en escala de grises
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    pastel_lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for i in range(256):
        if i < 64:
            pastel_lut[i] = [255, 204, 229]   # rosa claro
        elif i < 128:
            pastel_lut[i] = [204, 255, 204]   # verde menta
        elif i < 192:
            pastel_lut[i] = [204, 229, 255]   # azul lavanda
        else:
            pastel_lut[i] = [255, 255, 204]   # amarillo suave

    imagen_color = cv2.applyColorMap(imagen, pastel_lut)
    return imagen_color
def pseudocolor_personalizado(imagen, colores_rgb_norm):
    """
    Aplica un mapa de color personalizado a una imagen en escala de grises.
    colores_rgb_norm: lista de colores RGB normalizados (0–1)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Crear LUT
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    n = len(colores_rgb_norm)
    segmentos = np.linspace(0, 255, n).astype(int)

    for i in range(n - 1):
        c1 = np.array(colores_rgb_norm[i]) * 255
        c2 = np.array(colores_rgb_norm[i + 1]) * 255

        for j in range(segmentos[i], segmentos[i + 1]):
            alpha = (j - segmentos[i]) / (segmentos[i + 1] - segmentos[i])
            lut[j] = (1 - alpha) * c1 + alpha * c2

    lut[255] = np.array(colores_rgb_norm[-1]) * 255

    return cv2.applyColorMap(imagen, lut)

def pseudocolor_tierra(imagen):
    """
    Aplica un mapa de color tipo TIERRA (cafés, verdes, amarillos)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    tierra_lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for i in range(256):
        if i < 64:
            tierra_lut[i] = [50, 80, 120]      # café oscuro
        elif i < 128:
            tierra_lut[i] = [60, 120, 60]      # verde oliva
        elif i < 192:
            tierra_lut[i] = [80, 160, 120]     # verde claro
        else:
            tierra_lut[i] = [120, 200, 180]    # arena / beige

    return cv2.applyColorMap(imagen, tierra_lut)

def pseudocolor_frios(imagen):
    """
    Aplica un mapa de color de tonos fríos (azules, cian, morado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    frios_lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for i in range(256):
        if i < 64:
            frios_lut[i] = [80, 0, 80]        # morado oscuro
        elif i < 128:
            frios_lut[i] = [150, 50, 0]       # azul oscuro
        elif i < 192:
            frios_lut[i] = [200, 150, 50]     # azul claro
        else:
            frios_lut[i] = [255, 255, 150]    # cian claro

    return cv2.applyColorMap(imagen, frios_lut)


# ---------------------------------
# SELECTOR GENERAL DE MODELO DE COLOR
# ---------------------------------

def aplicar_modelo_color(imagen, modelo):
    """
    Aplica el modelo de color seleccionado
    """
    if modelo == "ESCALA_GRISES":
        return escala_grises(imagen)

    elif modelo == "BINARIA":
        return binarizar_imagen(imagen)

    elif modelo == "HSV":
        return rgb_a_hsv(imagen)

    elif modelo == "CMYK":
        return rgb_a_cmyk(imagen)

    elif modelo == "RGB_CANALES":
        mostrar_canales_rgb(imagen)
        return imagen

    elif modelo == "PSEUDOCOLOR_PASTEL":
        return pseudocolor_pastel(imagen)
    
    elif modelo == "PSEUDOCOLOR_TIERRA":
        return pseudocolor_tierra(imagen)
    
    elif modelo == "PSEUDOCOLOR_FRIOS":
        return pseudocolor_frios(imagen)


    else:
        return imagen
