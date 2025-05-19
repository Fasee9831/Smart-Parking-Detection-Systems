import cv2
import numpy as np

# Load images
ref_img = cv2.imread("currency_note.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread("test_note.jpg", cv2.IMREAD_GRAYSCALE)

if ref_img is None or test_img is None:
    print("Error loading images.")
    exit()

# ORB detector
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(ref_img, None)
kp2, des2 = orb.detectAndCompute(test_img, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"Good Matches after ratio test: {len(good)}")

note_status = "Counterfeit Note"
result_img = None

if len(good) > 10:
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is not None:
        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)

        print(f"Inlier Matches after Homography: {inliers}")

        if inliers > 15:
            note_status = "Genuine Note"
        else:
            note_status = "Counterfeit Note"

        print(f"Currency Note is  {note_status}")

        # Draw matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matches_mask,
                           flags=2)
        result_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, good, None, **draw_params)

else:
    print("Not enough good matches - Currency Note is NOT Authentic")
    note_status = "Counterfeit Note"
    print(f"Currency Note is {note_status}")
    # Still draw the matches without inlier filtering
    result_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, good, None, matchColor=(0, 0, 255), flags=2)

# Save and show labeled reference image
ref_labeled = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
cv2.putText(ref_labeled, "Genuine Note", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite("reference_labeled.jpg", ref_labeled)
cv2.imshow("Reference Image", ref_labeled)

# Save and show labeled test image
test_labeled = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
color = (0, 255, 0) if note_status == "Genuine Note" else (0, 0, 255)
cv2.putText(test_labeled, note_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
cv2.imwrite("test_labeled.jpg", test_labeled)
cv2.imshow("Test Image", test_labeled)

# Add label and show match result image
if result_img is not None:
    height, width = result_img.shape[:2]
    label_position = (int(width * 0.6), 30)
    cv2.putText(result_img, note_status, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow("Matching Result", result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()