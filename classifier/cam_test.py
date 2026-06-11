import cv2

cap = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = cap.read()
    print(i, ret)

cap.release()