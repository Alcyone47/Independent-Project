import cv2

# Load the face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the nose point classifier
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Load the video capture device (use 0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes in the face
        eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])

        # Loop over the eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Detect the nose point in the face
            nose = nose_cascade.detectMultiScale(gray[y:y + h, x:x + w], 1.3, 5)

            # Loop over the nose points
            for (nx, ny, nw, nh) in nose:
                # Calculate the position of the eye relative to the position of the nose
                eye_pos = (x + ex + ew / 2, y + ey + eh / 2)
                nose_pos = (x + nx + nw / 2, y + ny + nh / 2)
                rel_pos = (eye_pos[0] - nose_pos[0], eye_pos[1] - nose_pos[1])

                # Draw a line from the nose to the eye
                cv2.line(frame, nose_pos, eye_pos, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
