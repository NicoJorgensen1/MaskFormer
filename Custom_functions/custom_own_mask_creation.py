import os
import numpy as np 
from matplotlib import pyplot as plt 
import pickle 
import cv2

hist_dict_file = [x for x in os.listdir(os.getcwd()) if x.endswith("pkl") and "ypred" in x.lower()][0]
with open(hist_dict_file, "rb") as file :
    hist_file = pickle.load(file)

im_idx = 0
y_pred_col = hist_file["y_pred"][im_idx]
y_true_col = hist_file["y_true"][im_idx]

y_pred = (cv2.cvtColor(y_pred_col, cv2.COLOR_RGB2GRAY)).astype(np.uint8)
y_true = (cv2.cvtColor(y_true_col, cv2.COLOR_RGB2GRAY)).astype(np.uint8)

y_pred = (y_pred==29).astype(int) #+ (y_pred==226).astype(int) + (y_true==179).astype(int)
y_true = (y_true==29).astype(int) #+ (y_true==226).astype(int) + (y_true==179).astype(int)

y_pred_final = np.zeros_like(y_pred_col)
y_pred = y_pred.astype(bool)
y_true = y_true.astype(bool)
y_pred_final[y_pred] = (255,255,255)
y_pred_final[y_true] = (125,125,125)
y_intersection = np.logical_or(y_pred, y_true).astype(int)
y_union = np.logical_and(y_pred, y_true).astype(int)
y_union = y_union[105:340,95:330]
y_union = cv2.resize(src=y_union, dsize=y_pred.shape, interpolation=cv2.INTER_NEAREST).astype(bool)
union_y = np.ones_like(y_pred_col)
union_y[y_union==1] = (100,0,230)
union_y[y_union==0] = (255,255,255)

y_intersection[y_pred] = 125
y_intersection[y_true] = 255
intersection_y = np.ones_like(y_pred_col)
intersection_y[np.logical_or(y_pred, y_true)] = (200, 145, 245)
intersection_y[np.logical_and(y_pred, y_true)] = (100,0,230)
intersection_y[np.logical_or(y_pred, y_true)==0] = (255,255,255)
intersection_y = intersection_y[20:380,20:380]
intersection_y = cv2.resize(src=intersection_y, dsize=y_pred.shape, interpolation=cv2.INTER_NEAREST)


num_rows, num_cols, ax_count = 2, (3,3), 0
im_list = [y_true_col, y_pred_col, intersection_y, y_true, y_pred, union_y]
titles = ["Ground truth", "Prediction", "Intersection", "True cell mask", "Predicted cell mask", "Union"]
fig = plt.figure()#(figsize=(8, 6))
for row in range(num_rows):
    for col in range(num_cols[row]):
        plt.subplot(num_rows, num_cols[row], 1+row*num_cols[row]+col)
        plt.imshow(im_list[ax_count], cmap="gray")
        plt.axis("off")
        loc = "left" if any([titles[ax_count].lower() in x for x in ["intersection", "union"]]) else "center"
        plt.title(titles[ax_count], loc="center")
        ax_count += 1

fig.tight_layout()
save_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Billeder eller figurer")
fig.savefig(os.path.join(save_dir, "IoU_self_test.jpg"), bbox_inches="tight")
plt.show(block=False)



