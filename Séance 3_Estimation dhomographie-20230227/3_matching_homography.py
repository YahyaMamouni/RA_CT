import numpy as np
import cv2


cap = cv2.VideoCapture('isima1.mp4')
#cap = cv2.VideoCapture(0)

# Load reference image
imgref = cv2.imread('ref_isima.jpg')
#imgref = cv2.imread("images/test.jpg")


# Load logo image
logo = cv2.imread('INP_texte.png', cv2.IMREAD_UNCHANGED)


# Create Detector and descriptor, sift par exemple
detector = cv2.ORB_create(nfeatures=300)

# Compute interest points and descriptors on the first image
kp_ref, ds_ref = detector.detectAndCompute(imgref,None)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
      break
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #---------------------------------
    # Create Keypoints and descriptors
    #---------------------------------    
    # Compute interest points and descriptors on the current frame
    kp_cur, ds_cur = detector.detectAndCompute(gray,None)
    
    #---------------------------------
    # Matching
    #---------------------------------
    # BFMatcher with default params
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ds_ref,ds_cur)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    n_matchs = 15
    
    #---------------------------------
    # Compute homography
    #---------------------------------
    if n_matchs > 4: #at least 4 matches for homography estimation 
      src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
      dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
       
      H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
      print('H = ',H)
      
    #---------------------------------
    # Compute rectangle
    #---------------------------------
      #h,w = imgref.shape

      src_rect = np.array([[0, 0], [0, imgref.shape[0] - 1], [imgref.shape[1] - 1, imgref.shape[0] - 1], [imgref.shape[1] - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
      #src_rect = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

      cur_rect = cv2.perspectiveTransform(src_rect, H)     

    #---------------------------------
    # Display 
    #---------------------------------
    # Rectangle : 
    frame = cv2.polylines(frame,[np.int32(cur_rect)],True,(0,255,255),3)
    
    # Matching results
    imgmatching = cv2.drawMatches(imgref,kp_ref,frame,kp_cur,matches[0:n_matchs],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Add logo to frame
    logo_resized = cv2.resize(logo, (cur_rect[2][0][0], cur_rect[2][0][1]))
    alpha_channel = logo[:, :, 3] / 255.0
    logo_image = logo_resized[:, :, 0:3]
    cur_rect_int = cur_rect.astype(int)
    bg_roi = frame[cur_rect_int[0][0][1]:cur_rect_int[2][0][1], cur_rect_int[0][0][0]:cur_rect_int[2][0][0]]
    bg_roi = bg_roi.astype(float)
    
    # Apply the alpha channel
    alpha_channel = cv2.resize(alpha_channel, (bg_roi.shape[1], bg_roi.shape[0]))
    alpha_channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
    logo_image = cv2.multiply(alpha_channel, logo_image.astype(float))
    
    # Apply the logo
    bg_roi = cv2.multiply(1.0 - alpha_channel, bg_roi)
    bg_roi = cv2.add(bg_roi, logo_image.astype(float))
    bg_roi = bg_roi.astype(np.uint8)
    frame[cur_rect[0][0][1]:cur_rect[2][0][1], cur_rect[0][0][0]:cur_rect[2][0][0]] = bg_roi

    # Display the resulting frame
    cv2.imshow('ref image',imgref)
    #cv2.imshow('color frame',frame)
    cv2.imshow('Matching result',imgmatching)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
