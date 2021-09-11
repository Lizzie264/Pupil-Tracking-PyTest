import cv2
import numpy as np

#Stuff we'll need
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('stockimage.jpeg')

#Let's start by detecting a face
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make picture gray
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5) #Detect objects in image and return rectangles

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) #Draw rectangles on our image with (255, 255, 0)RGB and contour thickness of 2 pixels

gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
face = img[y:y+h, x:x+w] # cut the face frame out
eyes = eye_cascade.detectMultiScale(gray_face)

for (ex,ey,ew,eh) in eyes: 
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

