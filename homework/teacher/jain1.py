import pandas as pd
from skimage import io
import os

data = pd.read_csv('teacher/a943287.csv')


male_urls = data[data['please_select_the_gender_of_the_person_in_the_picture']=='male']['image_url']
female_urls = data[data['please_select_the_gender_of_the_person_in_the_picture']=='female']['image_url']


total_len = len(male_urls) + len(female_urls)
print(total_len)
# plt.savefig('img')
# plt.show()


# path = './teacher\down/'
path = './'
# filename = path + "final_" + 3 + ".jpg"
# print(filename)

ids = male_urls.keys()
print(ids)

def image_save(image_urls, gender):
    ids = image_urls.keys()
    print(ids)
    os.makedirs(path + gender, exist_ok= True)
    for id, url in zip(ids, image_urls):
        # print(url)
        # print(id)
        try:
            image = io.imread(url)
                                                    # 이 부분에서 resize해서 이미지 저장해도 됌! 
            # print(image)
            if(image.shape==(300, 300, 3)):
                print('########')
                filename = "final_" + str(id) + ".jpg"
                save_path = os.path.join(path, gender , filename)       # 이미지 저장 경로 ' path/male/이미지.jpg
                # print(save_path)
                io.imsave(save_path, image)      # 한번만 다운받아서 저장하자꾸나.
                print("정상저장_", id)
        except:
            continue
        


for url, gender in zip([male_urls, female_urls], ['male', 'female']):
    image_save(url[:2000], gender)

'''
        for url in urls
            

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

'''