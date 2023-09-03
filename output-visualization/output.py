import cv2
import numpy as np
import tensorflow as tf
from scipy import signal
import mediapipe as mp
from gtts import gTTS
from io import BytesIO
import pygame
import math

def speak(text):
    mp3_fp = BytesIO()
    tts = gTTS(text, lang='en')
    tts.write_to_fp(mp3_fp)
    return mp3_fp

pygame.init()
pygame.mixer.init()

listx = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
counter = 1
tmp = "ax"

def func(letter, coun):
    listx[coun-1] = letter
    if len(np.unique(np.array(listx))) == 1:
        audiox(listx[0])
    
def audiox(letter):
    global tmp
    if letter == tmp:
        return 0
    else:
        tmp = letter
        playx(letter)
    
def playx(letter):
    sound = speak(letter)
    sound.seek(0)
    pygame.mixer.music.load(sound, 'mp3')
    pygame.mixer.music.play()

list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

mp_hands = mp.solutions.hands

mp_model = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.75
)

video = cv2.VideoCapture("video.mp4")
modelx = tf.keras.models.load_model("sign_model_with_1_optuna")
k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5,5])

success, frame = video.read()
if not success:
    print("Failed to read the video")
    exit()

while True:
    # Read the next frame
    success, frame = video.read()
    
    raw_frame = frame

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
       
        frame = frame[miny:maxy,minx:maxx]
        frame = cv2.resize(frame, (224,224))
    
        # print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to the frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        xMask = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
        yMask = xMask.T.copy()
        fx = signal.convolve2d(blur, xMask, boundary='symm', mode='same')
        fy = signal.convolve2d(blur, yMask, boundary='symm', mode='same')
        Gm = (fx**2 + fy**2)**(1/2)
        imSharp = blur + Gm

        res = cv2.morphologyEx(imSharp, cv2.MORPH_TOPHAT, k_ellipse)
        
        res = cv2.resize(res, (224,224))
        
        # fxx = res.copy()
        
        res = cv2.merge([res, res, res])
        
        resx = res.copy()
        frame = np.array(resx, dtype=np.int32)
        frame = np.invert(frame)
        
        framex = np.expand_dims(frame,axis=0)

        predict = modelx.predict(framex)
        preval = np.argmax(predict)
        
        frame = raw_frame
        predFrame = cv2.putText(np.zeros((300,300)), list[preval], (100,200), cv2.FONT_HERSHEY_COMPLEX, 6, (255,0,255), 2, cv2.LINE_AA)
        func(list[preval], counter)
        if counter>9:
            counter = math.floor(counter%10)
        counter += 1
        
        cv2.imshow("framex", frame)
        cv2.imshow("Predictions", predFrame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
