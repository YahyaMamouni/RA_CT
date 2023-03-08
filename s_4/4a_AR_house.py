import numpy as np
import cv2
import pdb


SIFT = 0
MIN_MATCHES = 6

# Pour tester si un point est à l'intérieur d'une image
def inside_image(x0,y0,frame):
  return ((x0>0) & (y0>0)&(x0< frame.shape[1]) & (y0< frame.shape[0]))


def main():
  #---------------------------------
  # Accès Webcam
  #---------------------------------
  cap = cv2.VideoCapture(0)
  imgref = cv2.imread('test.jpg')

  size = imgref.shape
  h, w = imgref.shape[:2]
  #------------------------------------------
  # Définition des paramètres de calibration
  #------------------------------------------

  # A completer
  focal_length = size[1]
  center = (size[1]/2, size[0]/2)
  K = np.array([[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
                )
  dist_coef = np.zeros((4,1))


  #------------------------------------
  # Traitement de l'image de reference
  #------------------------------------
  
  

  # Détecteur de point d'intérêts et matcher
  if SIFT:
    detector = cv2.SIFT_create(nfeatures=1000)
    bf = cv2.BFMatcher(crossCheck=True)
  else:
    detector = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Calcul des points d'intérêt et descripteurs sur l'image de référence
  kp_ref, ds_ref = detector.detectAndCompute(imgref,None)
      

  #------------------------------------
  # Définition du modèle filaire
  #------------------------------------

  # Coordonnées de sommets :
  ar_verts = np.float32([[0, 0, 0], [w, 0, 0],[w, h, 0], [0, h, 0] ,[0, 0, -500], [w, 0, -500], [w, h, -500], [0, h, -500]]) # A completer
  #ar_verts = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
  # Arêtes :
  #ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
  ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),(0, 4),(1 ,5),(2, 6),(3, 7) , (4,5) ,(5, 6), (6, 7), (7, 4)]
         

  #------------------------------------
  # Boucle principale
  #------------------------------------
  while(True):
    # Lecture de l'image
    ret, frame = cap.read()
    if ret == False:
      print("Impossible de récupérer l'image de la webcam")
      break    
    # conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #---------------------------------
    # Keypoints and descriptors
    #--------------------------------- 
    # Compute interest points and descriptors on the current frame
    kp_cur, ds_cur = detector.detectAndCompute(frame,None)

    #---------------------------------
    # Matching
    #---------------------------------

    # Calcul des appariements 
    matches = bf.match(ds_ref,ds_cur)
    # Classement des appariements
    matches = sorted(matches, key = lambda x:x.distance)
    th = 3 * matches[0].distance  # seuil de conservation des appariements
    n_matchs = 0
    for m in matches:
      if m.distance <= th:
        n_matchs = n_matchs +1
      else:
        break
    #---------------------------------
    # Calcul de l'homographie
    #---------------------------------
    if n_matchs > MIN_MATCHES: 
      src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches[0:n_matchs]]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in matches[0:n_matchs]]).reshape(-1, 1, 2)
      H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

      #---------------------------------
      # Affichage rectangle
      #---------------------------------
      src_rect = np.array([[0, 0],
                           [imgref.shape[1],0],
                           [imgref.shape[1],imgref.shape[0]],
                           [0,imgref.shape[0]]])
      cur_rect = cv2.perspectiveTransform(src_rect.reshape(-1,1,2).astype(np.float32), H)

      frame = cv2.polylines(frame, [np.int32(cur_rect)],True,255,3, cv2.LINE_AA)

      #---------------------------------
      # Calcul de la projection 3D-2D
      #---------------------------------
      # A completer
      quad_3d = np.float32([[0, 0 , 0],
                           [imgref.shape[1],0 , 0],
                           [imgref.shape[1],imgref.shape[0] , 0],
                           [0,imgref.shape[0], 0]])
      quad_2d = cur_rect

      _ret, rvec, tvec = cv2.solvePnP(quad_3d, quad_2d, K, dist_coef)

      #---------------------------------
      # Projection du modèle filaire
      #---------------------------------
      verts,_ = cv2.projectPoints(ar_verts, rvec, tvec, K, dist_coef)
      #---------------------------------
      # Affichage
      #---------------------------------
      # Affichage des arêtes avec cv2.line
      for i, j in ar_edges:
        (x0, y0), (x1, y1) = verts[i][0], verts[j][0]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
      
    # Affichage des appariements
    imgmatching = cv2.drawMatches(imgref,kp_ref,frame,kp_cur,matches[0:n_matchs],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Affichage du résultat
    cv2.imshow('Resultat',frame)
    cv2.imshow('Appariements',imgmatching)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

  return 0

if __name__ == '__main__':
    main()