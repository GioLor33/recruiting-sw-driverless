import cv2
import numpy as np

min = 50
max = 200

img = cv2.imread('corrupted.png')
rows, cols, _ = img.shape
height = rows//2

# STEP 1: Solve the rolling shift

top_image = img[height:rows, 0:cols]
M = np.float32([[1,0,0],[0,1,height]])
img = cv2.warpAffine(img,M,(cols,rows))
img[0:height, 0:cols] = top_image

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()