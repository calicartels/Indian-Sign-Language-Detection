import cv2
import numpy as np
from scipy import signal
import os
import mediapipe as mp

mp_hands = mp.solutions.hands

mp_model = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.75
)

# Load the video
video = cv2.VideoCapture("video.mp4")
currentframe=1000
k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5,5])

# Read the first frame to get the video's width and height
success, frame = video.read()
if not success:
    print("Failed to read the video")
    exit()

# Create the output folder
if not os.path.exists('datax11'):
    os.makedirs('datax11')


# Loop through all the frames in the video
while True:
    # Read the next frame
    success, frame = video.read()

    # If the frame was not read correctly, break out of the loop
    if not success:
        break
    
    results = mp_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_height, img_width, c = frame.shape
    
    xlist,ylist = [],[]
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = img_width
            y_min = img_height
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                xlist.append(x)
                ylist.append(y)
            
    if len(xlist)>0 and len(ylist)>0:
        minx,maxx,miny,maxy = max(0,min(xlist)-15),min(img_width,max(xlist)+15),max(0,min(ylist)-15),min(img_height,max(ylist)+15)
        # minx,maxx,miny,maxy = min(xlist)-15,max(xlist)+15,min(ylist)-15,max(ylist)+15
       
        frame = frame[miny:maxy,minx:maxx]
        frame = cv2.resize(frame, (224,224))
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to the frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        xMask = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
        yMask = xMask.T.copy()
        fx = signal.convolve2d(blur, xMask, boundary='symm', mode='same')
        fy = signal.convolve2d(blur, yMask, boundary='symm', mode='same')
        Gm = (fx**2 + fy**2)**(1/2)
        imSharp = blur + 3*Gm

        res = cv2.morphologyEx(imSharp, cv2.MORPH_TOPHAT, k_ellipse)
        
        res = cv2.merge([res, res, res])
        
        cv2.imshow("res", res)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        
        fliph = cv2.flip(res, 0)
        flipv = cv2.flip(res, 1)
        flipb = cv2.flip(res, -1)
        
        cv2.imwrite("datax11/" + str(currentframe) + '.jpg' , res)
        currentframe += 1
        cv2.imwrite("datax11/" + str(currentframe) + '.jpg' , fliph)
        currentframe += 1
        cv2.imwrite("datax11/" + str(currentframe) + '.jpg' , flipv)
        currentframe += 1
        cv2.imwrite("datax11/" + str(currentframe) + '.jpg' , flipb)
        currentframe += 1
