import cv2
import numpy as np

def sobel(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)

def prewitt(img):
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    gx = cv2.filter2D(img, -1, kernelx)
    gy = cv2.filter2D(img, -1, kernely)
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

def roberts(img):
    kernelx = np.array([[1,0],[0,-1]])
    kernely = np.array([[0,1],[-1,0]])
    gx = cv2.filter2D(img, -1, kernelx)
    gy = cv2.filter2D(img, -1, kernely)
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

def canny(img, t1, t2):
    return cv2.Canny(img, t1, t2)
