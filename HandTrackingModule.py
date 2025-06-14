import mediapipe as mp
import cv2
import time


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=self.detectionCon,
                      min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.ctime = 0
        self.ptime = 0

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, idList=[]):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if id in idList and draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

    def getFPS(self, img):
        self.ctime = time.time()
        fps = 1 / (self.ctime - self.ptime)
        self.ptime = self.ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        return img

    def get_fingers_up(self,lmList):
        """Returns a list of fingers that are up (extended) based on landmark positions.

        Args:
            lmList (list): List of landmarks, where each entry is [id, x, y].

        Returns:
            list: A list of finger IDs that are currently up.
        """
        fingers = []

        # Thumb (id 4)
        if lmList[4][2] > lmList[3][2]:  # Compare y-coordinates
            fingers.append(1)  # Thumb is up

        # Index finger (id 8)
        if lmList[8][2] < lmList[6][2]:  # Compare y-coordinates
            fingers.append(2)  # Index finger is up

        # Middle finger (id 12)
        if lmList[12][2] < lmList[10][2]:  # Compare y-coordinates
            fingers.append(3)  # Middle finger is up

        # Ring finger (id 16)
        if lmList[16][2] < lmList[14][2]:  # Compare y-coordinates
            fingers.append(4)  # Ring finger is up

        # Pinky (id 20)
        if lmList[20][2] < lmList[18][2]:  # Compare y-coordinates
            fingers.append(5)  # Pinky is up

        return fingers

