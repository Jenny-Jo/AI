import pandas as pd
from skimage import io
data = pd.read_csv('./teacher/a943287.csv')
print(data.head())
print(data.shape)
# print(data.columns())
print(data.describe())
print(list(data))
print(len(list(data)))

data_male = data[data['please_select_the_gender_of_the_person_in_the_picture']=='male']
data_female = data[data['please_select_the_gender_of_the_person_in_the_picture']=='female']
final_data = pd.concat([data_male[:1000], data_female[:1000]], axis=0).reset_index(drop=True)
print(final_data.shape)     # (2000, 10)
print("==========================================")
# print(final_data.loc['image_url'])
print(final_data.iloc[0])
print("==========================================")
print(final_data.iloc[1])
print("==========================================")
print(final_data.iloc[2])
print("==========================================")
# print(final_data[0])
print(type(final_data))
print("==========================================")
print(final_data.loc[0]['image_url'])
print(final_data.loc[0][0])
print(final_data.loc[0][7])
print("==========================================")
print(final_data)
print("==========================================")
print(final_data.iloc[0][7])
print(final_data.iloc[0]['image_url'])
print(final_data.iloc[0].loc['image_url'])
print(final_data.iloc[0].iloc[7])
print("==========================================")
print(final_data.loc[0]['please_select_the_gender_of_the_person_in_the_picture'])

from skimage import io
from matplotlib import pyplot as plt
img = io.imread(final_data.loc[0]['image_url'])
print(img.shape)        # (300, 300, 3)
io.imshow(img)

# plt.savefig('img')
# plt.show()

x = []
y = []
path = './teacher\down/'
# filename = path + "final_" + 3 + ".jpg"
# print(filename)

for i in range(final_data.shape[0]):        # 2000번 돌려라.
    try:
        image = io.imread(final_data.loc[i]['image_url'])
        if(image.shape==(300, 300, 3)):
            x.append(image)
            y.append(final_data.loc[i]['please_select_the_gender_of_the_person_in_the_picture'])
            filename = path + "final_" + str(i) + ".jpg"
            print(filename)
            io.imsave(filename, image)      # 한번만 다운받아서 저장하자꾸나.
            print("정상저장_", i)
    except:
        continue

# !pip install opencv-python
import cv2
import numpy as np
x2 = []
y2 = []
print(x)
for i in range(len(x)):
    img2 = cv2.resize(x[i], (50, 50))
    x2.append(img2)
    img_label = np.where(y[i]=='male', 1, 0)
    y2.append(img_label)

# print('여기까지 왔다.')






