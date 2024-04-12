import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('D:\pythonProject1\elephant.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
original_img = img
plt.show()
print(img.shape)

#flatten img

all_pixels = img.reshape((330*500, 3))
print(all_pixels.shape)

dominant_colors = 4
km = KMeans(n_clusters = dominant_colors)
km.fit(all_pixels)

centers = km.cluster_centers_
centers = np.array(centers, dtype='uint8')
print(centers)

i = 1
colors = []
for each_color in centers:
    plt.subplot(1, 4, i)
    plt.axis("off")
    i += 1
    colors.append(each_color)

    a = np.zeros((100, 100, 3), dtype='uint8')
    a[:, :, :, ] = each_color
    plt.imshow(a)
    plt.show()

new_img = np.zeros((330*500, 3 ) , dtype='uint8')
print(new_img.shape)

a = km.labels_
print(len(a))

for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]

new_img = new_img.reshape(original_img.shape)
plt.imshow(new_img)
plt.show()






