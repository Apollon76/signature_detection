import numpy as np
import cv2

img = cv2.imread('s1200.jpeg')

blur = cv2.GaussianBlur(img, (9,9), 0)
cv2.imshow('blur', blur)

edges = cv2.Canny(blur, 100, 200)

cv2.imshow('edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
