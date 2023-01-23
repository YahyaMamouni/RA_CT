import numpy as np
import cv2
import pdb

cap = cv2.VideoCapture(0)


while(True):
    # Image de la webcam:
    ret, frame = cap.read()
    if ret == False:
      break

    # pour charger une image enregistrée sur le disque :  
    #frame = cv2.imread('../images/cham1.jpg')

    
    # conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 


    # Traitements à compléter ici : détection des points d'intérêt
    # Harris corner, SIFT, FAST, SURF...

        
    """
    # SIFT
    # Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Detect and describe the features
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw the keypoints on the image
    frame = cv2.drawKeypoints(gray, keypoints, frame)
    """
    
    
    # Harris

    corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 10)
    corners = np.int0(corners)
 
    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    """    
    #FAST
    # Create a FAST object
    fast = cv2.FastFeatureDetector_create()
    
    # Detect corners
    keypoints = fast.detect(gray, None)
    
    # Draw corners
    frame = cv2.drawKeypoints(gray, keypoints, frame)
   """
   
    # Affichage de l'image
    cv2.imshow('gray frame',gray)
    cv2.imshow('color frame',frame)
    #cv2.imshow('harris frame', harris)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quand tout est terminé on ferme les fenêtres et on libère le flux webcam
cap.release()
cv2.destroyAllWindows()
