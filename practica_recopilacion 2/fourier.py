import numpy as np
import cv2

def compute_fft(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    return magnitude

def low_pass_filter(img, radius):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    return img_back

def high_pass_filter(img, radius):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    return img_back
