import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv.imread('car.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

bfil = cv.bilateralFilter(gray_img,11,17,17,17)
edged = cv.Canny(bfil, 30, 200)

keypoints = cv.findContours(edged.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea,reverse=True)[:10]
location = None
for contour in contours:
  approx = cv.approxPolyDP(contour,10,True)
  if len(approx) == 4:
    location = approx
    break
mask = np.zeros(gray_img.shape,np.uint8)
new_image = cv.drawContours(mask,[location],0,255,-1)
new_image = cv.bitwise_and(img,img,mask=mask)

#crop image
(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
cropped_image = gray_img[x1:x2+1,y1:y2+1]
# extract text
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)