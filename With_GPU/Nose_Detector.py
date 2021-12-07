import cv2
def Nose_Detection(noseCascade, face_bbox_frame):
  flag = 0
  gray = cv2.cvtColor(face_bbox_frame, cv2.COLOR_BGR2GRAY)
  nose = noseCascade.detectMultiScale(gray)
  print(len(nose))
  if len(nose) > 0:
    return 1
  else:
    return 0