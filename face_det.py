import cv2
import numpy as np

#init web cam
cap = cv2.VideoCapture(0)

# face dectection
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_data = []
dataset_path = "./data/"
file_name = input("enter the name of the person : ")


while True:
    ret, frame = cap.read()

    if ret == False :
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f:f[2]*f[3])

    #pick the last face(because it has the largest area)

    for face in faces[-1:]:

        #draw bounding box or the rectangle
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #extract
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)
        print(len(face_section))


    # print the op screen
    cv2.imshow("frame",frame)
    #cv2.imshow("gray", gray)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert face data list into numpy array

face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy', face_data)
print("data save successfully")


cap.release()
cv2.destroyAllWindows()