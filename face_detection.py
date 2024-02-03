import cv2
facecd = cv2.CascadeClassifier("files/frontalface.xml")
cap = cv2.VideoCapture(0)

while True:
    lg,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facecd.detectMultiScale(img,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)
        cv2.imshow('img',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break