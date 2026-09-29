#here we import face data from cv library
#importing open cv data

import cv2 

#loading dataset
trainedData=cv2.CascadeClassifier('facedata.xml')

#selection of image
img=cv2.imread('tonystark.jpg')

#conversion to grayscale(black and white)
grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('tonystark.jpg',img)//to check whether img is inserted or not

#detection of face
faceCoordinates=trainedData.detectMultiScale(grayimg)

#print(faceCoordinates)
#[[278  32  82  82]
# [544  32  79  79]]

x,y,w,h=faceCoordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#show the image
cv2.imshow('tony',img)

#displaying image
#cv2.imshow('tonystark.jpg',grayimg)
#here waitkey used to pause the execution of program until any key is pressed
cv2.waitKey()

print('end of program')