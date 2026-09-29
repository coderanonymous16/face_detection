#here we import face data from cv library
#importing open cv data

import cv2 as cv

#loading dataset
trainedData=cv.CascadeClassifier('facedata.xml')

#selection of image
img=cv.imread('tonystark.jpg')

#conversion to grayscale(black and white)
grayimg=cv.cvtColor(img,cv.COLOR_BGR2BGRAY)

#detection of face
faceCoordinates=trainedData.detectMultiScale(grayimg)

#displaying image
cv.imshow('tonystark.jpg',grayimg)
#here waitkey used to pause the execution of program until any key is pressed
cv.waitKey()