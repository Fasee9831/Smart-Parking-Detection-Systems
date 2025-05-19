import cv2
import numpy as np

# Function to calculate the convex hull and defects
def get_hull_defects(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    return max_contour, defects

# Function to count fingers
def count_fingers(defects, threshold_angle=80):
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / np.pi
        
        if angle <= threshold_angle and a > 20 and b > 20 and c > 20:
            count += 1
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Background subtraction setup (simplified)
_, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    
    # Calculate the absolute difference to detect motion
    diff = cv2.absdiff(bg_gray, frame_gray)
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Get the hand mask
    max_contour, defects = get_hull_defects(thresh)
    
    # Count fingers and display
    if max_contour is not None:
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        finger_count = count_fingers(defects)
        cv2.putText(frame, str(finger_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()