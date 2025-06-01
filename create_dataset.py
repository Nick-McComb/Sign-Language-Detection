import os
import mediapipe as mp
import cv2
import pickle
import matplotlib.pyplot as plt
#data path
DATA_DIR = './data'

#objects to detect landmarks and draw landmarks over image
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #temp 1
        #read image from letter directory
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        #convert image from bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #process rgb images using hand model
        results = hands.process(img_rgb)
        
        data_aux = []

        #check for multihands landmakrs
        if results.multi_hand_landmarks:
            #loop over # hands 
            for hand in results.multi_hand_landmarks:
                '''mp_drawing.draw_landmarks(
                    img_rgb,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())'''
                #loop over landmarks per hand
                for landmark in hand.landmark:
                    #collect x and y of landmark data
                    x = landmark.x
                    y = landmark.y
                    #store x and y of landmark data
                    data_aux.append(x)
                    data_aux.append(y)
            '''plt.figure()
            plt.imshow(img_rgb)'''
            #store specific class landmark data
            data.append(data_aux)
            #store labels 
            labels.append(dir_)
'''plt.show()'''
#create file storing landmark data/labels  
f = open('data.pickle', 'wb')
pickle.dump({"data": data, "labels": labels}, f)
f.close()