#Import needed modules
import cv2
import mediapipe as mp
import time

#This is to use the video capture device of the PC
cap = cv2.VideoCapture(1)

#Calling the hand module in mediapipe.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

#Opens the VideoCapture window with hand detection
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

#Plotting points and connections on the parts of the hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id, cx, cy)
                #if id == 0:
                cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#Setting up the fps
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

#Displaying the fps in the img
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
    (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)