from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt

img = load_img('./teacher\down/final_0.jpg',target_size=(100,100))
data = img_to_array(img)

print(data)
print(type(data))   # <class 'numpy.ndarray'>
print(data.shape)   # (300, 300, 3)

plt.subplot(1,2,1)
plt.imshow(img)


img2 = load_img('teacher/down/final_7.jpg')
data2 = img_to_array(img2)

print(data2)
print(type(data2))   # <class 'numpy.ndarray'>
print(data2.shape)   # (300, 300, 3)

plt.subplot(1,2,2)
plt.imshow(img2)


plt.show()