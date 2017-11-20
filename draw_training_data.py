import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

full_labels = pd.read_csv('labels/all_labels.csv')

def draw_boxes(image_name):
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread('images/{}.JPG'.format(image_name))
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255,0,0), 5)
    return img

row = col = 10
fig, axs = plt.subplots(row,col, figsize=(20,20))
axs = axs.flatten()
im_ids = np.random.choice(full_labels.filename.unique(), row*col, replace=False)

for i, im in enumerate(im_ids):
    axs[i].imshow(draw_boxes(im))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    axs[i].set_aspect('equal')
plt.savefig('training_data.png')
