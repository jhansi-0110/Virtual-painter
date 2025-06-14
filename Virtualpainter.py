import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

brushthickness = 15
eraserthickness=100

folderPath="Header"  #where we stored images folder
myList=os.listdir(folderPath)  #used os to get files of images

overLayList=[]
fingers=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')#returns the image as a matrix (NumPy array) where each element represents a pixel in the image.
    overLayList.append(image)

header=overLayList[0]
draw_color=(255,0,255)

#run webcam
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=htm.HandTracker(detectionCon=0.85)

xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3),np.uint8)
while True:
    #1. import image
    success,img=cap.read()
    img = cv2.flip(img, 1)

    #2.find hand landmarks
    img = detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    # tip of index finger and middle finger[8][1:0]  slicing [8,123,344]
    if len(lmList)!=0:

        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
    #3.check which fingers are up  (index finger up draw 2 fingers up select)
        fingers = detector.get_fingers_up(lmList)


    #4.if selection mode-2 fingers are up select
        if len(fingers)==2 and fingers[0]==2 and fingers[1]==3 :
            cv2.rectangle(img,(x1,y1-30),(x2,y2+30),draw_color,cv2.FILLED)
            xp, yp = 0, 0
            if y1<125:
                if 250<x1<400:
                    draw_color=(255,0,255)
                    header=overLayList[0]
                elif 550<=x1<650:
                    draw_color=(255,0,0)
                    header=overLayList[1]
                elif 750<=x1<900:
                    draw_color = (0, 255, 0)
                    header=overLayList[2]
                elif 1050<=x1<1200:
                    draw_color = (0, 0, 0)
                    header=overLayList[3]


    #5.if drawing mode-index finger up
        if len(fingers)==1 and fingers[0]==2:
            cv2.circle(img,(x1,y1),15,draw_color,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
        #image wont be there permanently so we need to create canvas to draw
            if draw_color==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraserthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, eraserthickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),draw_color,brushthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, brushthickness)
            xp,yp=x1,y1
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)


    #laying out image we have to slice and specify height and width
    img[0:125,0:1280]=header
    img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

# from HandTrackingModule import HandTracker
#
#
# # Initialize HandTracker object
# tracker = HandTracker()
#
# # Capture video
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
#
#     # Find hands and draw them on the frame
#     img = tracker.findHands(img)
#
#     # Get the position of a specific landmark (e.g., id 8 for index fingertip)
#     lmList = tracker.findPosition(img, draw=True, idList=[8])
#
#     # Optional: Print the position of landmark 8 if it's available
#     if lmList:
#         print(lmList[8])
#
#     # Calculate and display FPS
#     img = tracker.getFPS(img)
#
#     # Show the image
#     cv2.imshow("Hand Tracking", img)
#     cv2.waitKey(1)