import cv2
import numpy as np
import os

def create_original(name):
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

    # STEP 2: Solve the color shift.

    # Since the corrupted.png file is an image which is normalized but has been 'corrupted', meaning it has 
    # a (probably different) offset in the three BGR channels, we need to first find the offset for each part of 
    # the image (lower and upper part). This can be done by comparing the color of the corrupted image with the 
    # color of the original one (which needs to be normalized).

    # Y=541 X=128: [250,158,3] -> due to the y value, this is in the lower part of the image
    bgr_lowerPart = np.array([250,158,3])
    color_lowerPart = img[541,128]
    offset_lowerPart = np.array([0,0,0])

    # Y=267 X=564: [40,195,240] -> due to the y value, this is in the upper part of the image
    bgr_upperPart = np.array([40,195,240])
    color_upperPart = img[267,564]
    offset_upperPart = np.array([0,0,0])

    for a in range(3):
        offset_upperPart[a] = color_upperPart[a] - min - (max-min)*(bgr_upperPart[a])/(255)
        offset_lowerPart[a] = color_lowerPart[a] - min - (max-min)*(bgr_lowerPart[a])/(255)

    # To compute the BGR values of the original image, I used the following formulas on each one of the BGR channels:

    # original_img_channel * (max-min) / 255 + min = normalized_img_channel
    # &
    # corrupted_img_channel = normalized_img_channel + offset => normalized_img_channel = corrupted_img_channel - offset

    # => original_img_channel * (max-min) / 255 + min = corrupted_img_channel - offset
    # => original_img_channel = (corrupted_img_channel - offset - min) * 255 / (max-min)

    # where corrupted_img_channel is given, the offset has been computed above while original_img_channel is our target.

    for i in range(rows):
        offset = offset_lowerPart
        if i < height:
            offset = offset_upperPart

        for j in range(cols):
            for a in range(3):
                img[i,j][a] = (img[i,j][a] - offset[a] - min)*255/(max-min)

    print(img[267,564])
    print(img[541,128])

    # As we can see, the BGR values obtained for the two points whose values where noted during the calibration 
    # procedure are (more or less) correct. In the image some artifcats appears, but they are probably due to the
    # normalization process applied on the image (compression)

    cv2.imwrite(name,img)



name = 'original.png'
if not os.path.exists(name):
    create_original(name)

# STEP 3: Detect the three cones in the picture

# Use HSV color space to separate the foreground from the background. The range in the HSV color space used
# was empirically found.
img = cv2.imread(name)
hsv = img.copy()
hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
binary_img = cv2.inRange(hsv, (0, 180, 50), (179, 255, 255))

# Added some blur to remove some noise from the image to improve the thresholding
binary_img = cv2.medianBlur(binary_img,9)

# Threshold the image in order to get a "mask" in binary image with only the three cones
_, binary_image = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

# From the binary image, it is easy to extract the contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('binary after hsv thresholding',binary_image)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()