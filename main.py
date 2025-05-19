import cv2
import numpy as np


img = cv2.imread('parking_lot.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
display = img.copy()


REGION_W, REGION_H = 140, 220


lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])


lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
car_mask = red_mask | yellow_mask

def check_parking_spot(x, y):
    global display
    x1, y1 = max(0, x - REGION_W//2), max(0, y - REGION_H//2)
    x2, y2 = min(img.shape[1], x1 + REGION_W), min(img.shape[0], y1 + REGION_H)
    roi_mask = car_mask[y1:y2, x1:x2]

    car_pixels = cv2.countNonZero(roi_mask)

    if car_pixels > 200:  
        status = "Occupied"
        color = (0, 0, 255)
    else:
        status = "Empty"
        color = (0, 255, 0)

    display = img.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        check_parking_spot(x, y)

cv2.namedWindow("Color-Based Parking Detection")
cv2.setMouseCallback("Color-Based Parking Detection", on_mouse)

while True:
    cv2.imshow("Color-Based Parking Detection", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()