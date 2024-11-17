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

# Since from the HSV the white and black lines in the cone are not visible, we need to combine their bboxes 
# to get the final bounding boxes of each cone. Since the cones have a particular shape such that the bbox of
# the base has a width which is bigger than the bboxes of the upper parts of the cone, the idea is to find one
# base at a time and check all the others bbox to get which ones belong to such base. 
# To do so, I filtered the contours found by the y-values in descending order (from bbox which start at the 
# bottom to top); this assures to find the base  of a cone (or at least the bottom-visible part of it).
# For each base found, I checked all the other boxes to see if they are positioned above the base and inside
# its width.  More conditions are added to be sure to detect only one cone, the comments on that can be found
# in the code below.

#print(len(contours)) -> we can see that too many contours where found, some with a very small area. We need to filer them
contour_filtered = [
    cv2.boundingRect(contour)
    for contour in contours
    if cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] > 20
]
contour_sorted = sorted(contour_filtered, key=lambda x: (-x[1]))

bounding_boxes = []
colors = []
cone_in_focus = []
while len(contour_sorted) > 0:
    base = contour_sorted[0]
    x, y, w, h = base
    cone_in_focus = []
    cone_in_focus.append(base)
    i = 0
    contour_sorted.remove(base)

    # Find the color of the base
    x_center = x + w//2 + 25    # the '+25' is needed due to the artifacts in the image, mainly in the center cone
    y_center = y + h//2
    colors.append(img[y_center, x_center])  #in BGR
    cv2.circle(img, (x_center, y_center), 1, (0,255,0), 1)

    for info in contour_sorted[:]:
        x, y, w, h = info
        
        if x > cone_in_focus[0][0] and x + w < cone_in_focus[0][0] + cone_in_focus[0][2]:
            if w < cone_in_focus[i][2]: 
                cone_in_focus.append(info)
                contour_sorted.remove(info)
                i += 1
            else:
                # If the width of the bbox is bigger than the last bbox added to the cone, it means we found
                # the base of another cone. We can break the loop.
                break
    
    x = cone_in_focus[0][0]
    y = cone_in_focus[i][1]
    w = cone_in_focus[0][2]
    h = cone_in_focus[0][1] + cone_in_focus[0][3] - cone_in_focus[i][1]

    bounding_boxes.append((x,y,w,h))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# STEP 4: Create the .txt file with the cones+bboxes info
# To be more efficient, I saved the color of the cone already in the previous loop. In fact, I considered the
# center of the base bbox to be sure to get the right color. Here, therefore, I simply write the information
# on the .txt file.

i = 0
with open('bboxes.txt', 'w') as f:
    for bbox in bounding_boxes:
        hsv = cv2.cvtColor(np.uint8([[colors[i]]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = hsv[0]

        # 0–15	    Red-Orange
        # 15–85	    Yellow-Green
        # 85–125	Green-Cyan
        # 125–170	Cyan-Blue
        # 170–180	Magenta-Purple

        if 0 <= hue < 15:
            color = 'orange'
        elif 15 <= hue < 85:
            color = 'yellow'
        elif 85 <= hue < 170:
            color = 'blue'
        else:
            color = 'unknown'

        x, y, w, h = bbox

        f.write(f'{color}: ({x}, {y}, {x+w}, {y+h})\n')
        i += 1

cv2.imshow('binary after hsv thresholding',binary_image)
cv2.imshow('img',img)
cv2.imwrite('bboxes.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()