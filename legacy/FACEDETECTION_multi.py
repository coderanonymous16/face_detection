#here we import face data from cv library
#importing open cv data

import cv2 
from random import randrange as r

#loading dataset
trainedData=cv2.CascadeClassifier('facedata.xml')

#selection of image
img=cv2.imread('multip.jpg')

cv2.imshow('multip.jpg',img)  #//to check whether img is inserted or not

#conversion to grayscale(black and white)
grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



#detection of face
faceCoordinates=trainedData.detectMultiScale(grayimg)


for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)

#show the image
cv2.imshow('mvl',img)

#displaying image
#cv2.imshow('tonystark.jpg',grayimg)
#here waitkey used to pause the execution of program until any key is pressed
cv2.waitKey()


print('end of program')