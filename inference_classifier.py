import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
    cv2.imshow('Camera', frame_rgb)
    cv2.waitKey(1)
        



# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()
