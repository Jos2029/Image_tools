import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# DOMINIO DE FRECUENCIA
# =========================
def compute_fft(img):
    """Calcula la FFT y devuelve la magnitud y fase"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)  # Magnitud logarítmica
    phase = np.angle(fshift)                 # Fase


    return magnitude, phase

def mostrar_fft(img, title="FFT"):
    """Muestra magnitud y fase"""
    fshift, magnitude, phase = compute_fft(img)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(magnitude, cmap="gray")
    plt.title(f"{title} - Magnitud")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(phase, cmap="gray")
    plt.title(f"{title} - Fase")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# =========================
# FILTROS
# =========================
def low_pass_filter(img, radius):
    """Filtro pasa-bajo"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Aplicar máscara
    fshift_filtered = fshift * mask

    # Reconstrucción
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    return img_back

def high_pass_filter(img, radius):
    """Filtro pasa-alto"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    fshift_filtered = fshift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    return img_back

# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    img = cv2.imread("tu_imagen.jpg")  # Cambia por tu ruta
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mostrar FFT original
    mostrar_fft(img_gray, "Imagen Original")

    # Aplicar filtros
    img_low = low_pass_filter(img_gray, radius=30)
    img_high = high_pass_filter(img_gray, radius=30)

    # Mostrar resultados
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_low, cmap="gray")
    plt.title("Filtro Pasa-Bajo")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_high, cmap="gray")
    plt.title("Filtro Pasa-Alto")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
