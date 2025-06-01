import cv2
import os

DATA_DIR = './data'

#creates data dir if none
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#letters
number_of_classes = 3
#num of pictures
dataset_size = 100

#video capture
cam = cv2.VideoCapture(0)

#loops over # of classes
for i in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR ,chr(i+ 97))):
        os.makedirs(os.path.join(DATA_DIR, chr(i+97))) #makes dir for each letter

    done = True
    while True:
        #collects frames
        ret, frame = cam.read()
        # Display the captured frame
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)

        # Press 'q' to start collecting data
        if key == ord('q'):
            print("Collecting data for class {}" .format(chr(i+97)))
            done = False
            j = 0

        if not done:
            # Write the frame to the output file
            filename = os.path.join(DATA_DIR, chr(i+97), str(j) + ".jpg")
            cv2.imwrite(filename, frame)
            j+=1
            #break after reached data size
            if j == dataset_size:
                print("Finished data from class {}" .format(chr(i+97)))
                break

        # Press 'z' to exit the loop
        if key == ord('z'):
            print("Breaking from class {}" .format(chr(i+97)))
            break

# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()





