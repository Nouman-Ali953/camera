import os
import pickle
import numpy as np
import cv2
import face_recognition

# Replace 'your_camera_url' with the actual URL of your IP camera
camera_url = 'http://192.168.0.102:8080/video'

while True:
    try:
        # Attempt to open the camera
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            raise Exception("Error: Unable to open the camera.")

        cap.set(3, 640)
        cap.set(4, 480)

        # Load the encoding file
        print("Loading Encode File ...")
        with open('EncodeFile.p', 'rb') as file:
            encodeListKnownWithIds = pickle.load(file)
        encodeListKnown, studentIds = encodeListKnownWithIds
        print("Encode File Loaded")

        break  # Break out of the loop if the camera is successfully opened

    except Exception as e:
        print(f"Error: {e}")
        cap = None  # Set cap to None to indicate that the camera is not available
        print("Retrying in 2 seconds...")
        cv2.waitKey(2000)  # Wait for 2 seconds before retrying

while True:
    if cap is not None:
        success, img = cap.read()

        if not success:
            print("Failed to read a frame from the camera.")
            break  # Exit the loop or handle the error accordingly

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(
                    encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)

                # Assuming studentIds is a list of names corresponding to encodeListKnown
                name = studentIds[matchIndex]
                print(f"Recognized: {name}")

                # Draw a rectangle around the face
                top, right, bottom, left = faceLoc
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

                # Display the name near the face
                cv2.putText(img, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Camera Stream", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if cap is not None:
    cap.release()

cv2.destroyAllWindows()
