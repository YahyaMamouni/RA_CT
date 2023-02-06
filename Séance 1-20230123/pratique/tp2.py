import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Image de la webcam:

    ret, img2 = cap.read()
    if ret == False:
        break
    # Load images
    img1 = cv2.imread("images/test.jpg")

    # Detect keypoints and compute descriptors using ORB
    #SIFT

    #ORB 
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute-force matching

    #ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)


    good_matches = []
    for m in matches:
        if m.distance < 30:
            good_matches.append(m)
    # Draw first 10 matches
    #img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show results
    cv2.imshow("Matches", img_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Quand tout est terminé on ferme les fenêtres et on libère le flux webcam
cap.release()
cv2.destroyAllWindows()