import cv2
import sys

mouthCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth = mouthCascade.detectMultiScale(gray, 1.3, 5)
    faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Draw a rectangle around the faces
        roi_gray_mouth = gray[y+(int(h/2)):y+h, x:x+w]
        roi_color_mouth = frame[y+(int(h/2)):y+h, x:x+w]

        roi_gray_eye = gray[y-(int(h/2)):y+h, x:x+w]
        roi_color_eye = frame[y-(int(h/2)):y+h, x:x+w]

        mouth = mouthCascade.detectMultiScale(roi_gray_mouth)
        eyes = eyeCascade.detectMultiScale(roi_gray_eye)
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(roi_color_mouth, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        for (eex,eey,eew,eeh) in eyes:
            d = int(eew / 2)
            cv2.circle(roi_color_eye, (int(eex + eew / 4) + int(d / 2), int(eey + eeh / 4) + int(d / 2)), int(d) ,(0,0,255),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()