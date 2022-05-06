# Add the MaskFormer directory to PATH
import os 
import copy 
import numpy as np
import cv2 

def circle_finder(mask_used, img=None):
    mask = copy.deepcopy(mask_used)
    if mask.shape[-1] == 3:
        mask_gray = cv2.cvtColor(src=mask, code=cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = copy.deepcopy(mask)
    mask_blur = cv2.GaussianBlur(src=mask_gray.astype(float), ksize=(11,11), sigmaX=1, sigmaY=1).astype(np.uint8)
    detected_circles = cv2.HoughCircles(mask_blur.astype(np.uint8), method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, minRadius=10, maxRadius=35, param1=55, param2=10)
    num_circles = 0
    col = (0, 0, 255)
    if detected_circles is not None:
        num_circles = len(detected_circles)
        detected_circles = np.uint16(detected_circles)
        for pt in detected_circles[0,:]:
            a, b, r = pt[0], pt[1], pt[2]
            if img is None:
                mask = cv2.circle(img=mask, center=(a, b), radius=r, color=col, thickness=2)
            else:
                img = cv2.circle(img=img, center=(a, b), radius=r, color=col, thickness=2)
    if img is None:
        return mask, num_circles
    else:
        return img, num_circles

# Getting paths 
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets", "Vitrolife_dataset", "masks")
mask_files = [x for x in os.listdir(dataset_dir) if x.endswith(".png")]
mask_used = cv2.imread(os.path.join(dataset_dir, mask_files[0]), cv2.IMREAD_COLOR)

mask, num_circles = circle_finder(mask_used=mask_used, img=mask_used)

cv2.imshow("Mask with open-cv circle finder", mask_used)
cv2.waitKey(0)
cv2.destroyAllWindows()



