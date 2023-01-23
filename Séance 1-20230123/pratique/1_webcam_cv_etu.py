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

   
    # Affichage de l'image
    cv2.imshow('gray frame',gray)
    cv2.imshow('color frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quand tout est terminé on ferme les fenêtres et on libère le flux webcam
cap.release()
cv2.destroyAllWindows()
