import pandas as pd
from skimage import io
data = pd.read_csv('F:\Study\homework/teacher/a943287.csv')
print(data.head()) # 데이타 위에서 다섯번째 줄까지만 출력
print(data.shape)  # 데이터 shape 출력 (64084,10)
# print(data.columns())
print(data.describe()) # 데이터 평균, 표준편차,분산, 등 통계학적 수치를 보여주는 내적 함수
print(list(data)) # 데이터 열의 제목들을 리스트로 보여준다
print(len(list(data))) # 제목의 갯수
'''
data_male = data[data['please_select_the_gender_of_the_person_in_the_picture']=='male'] # 데이터 중 남자의 데이터만 가져옴 
data_female = data[data['please_select_the_gender_of_the_person_in_the_picture']=='female'] # 데이터 중 여자의 데이터만 가져옴
final_data = pd.concat([data_male[:1000], data_female[:1000]], axis=0).reset_index(drop=True) # 여자, 남자의 데이터를  천 개씩 가져와 위 아래로 붙이고, 인덱스는 다시 오름차순으로 바꿈
print(final_data.shape)     # (2000, 10)
print("=====================1=====================")
# print(final_data.loc['image_url'])
print(final_data.iloc[0]) # final_data의 첫번째 행(인덱스)의 데이터를 보여줌
print("=====================2=====================")
print(final_data.iloc[1])
print("======================3====================")
print(final_data.iloc[2])
print("======================4====================")
# print(final_data[0])
print(type(final_data)) # final_data의 데이터 타입이 판다스임을 알려줌. pandas.core.frame.DataFrame
print("======================5====================")
print(final_data.loc[0]['image_url']) # 0번째 행의 'image_url'의 열의 위치의 정보를 알려줌  
print(final_data.loc[0][0]) # 0번째 행의 0번째 열의 정보를 알려줌 (1023132475)
print(final_data.loc[0][7])# 0번째 행의 7번째 의 열의 위치의 정보를 알려줌  = 'image_url' # 'https://d1qb2nb5cznatu.cloudfront.net/users/40-large'
print("=====================6=====================")
print(final_data) # final data를 간략히 보여줌
print("=====================7=====================")
print(final_data.iloc[0][7])
print(final_data.iloc[0]['image_url'])
print(final_data.iloc[0].loc['image_url'])
print(final_data.iloc[0].iloc[7])
# 전부다 url의 위치를 찾아 url을 보여줌

# 판다스에서 iloc, loc 는 행으로 자료의 위치를 찾고 iloc는 위치정수를 기반으로 인덱싱하고, loc는 레이블을 기반으로 인덱싱한다

print("=====================8=====================")
print(final_data.loc[0]['please_select_the_gender_of_the_person_in_the_picture'])
# 첫번째 행의 성별을 찾아줌 : male

from skimage import io # 이미지를 읽는 방법 중 하나 / Image reading via the ImageIO Library
from matplotlib import pyplot as plt
img = io.imread(final_data.loc[0]['image_url']) # 첫째 행의 url로 이미지 읽어와서 넘파이로 변환
print(img.shape)        # (300, 300, 3)
io.imshow(img)   # 읽어온 이미지를 보여준다

# plt.savefig('img')
# plt.show()


x = []
y = []

path = './teacher\down/' # path를 지정해서 만들기
for i in range(final_data.shape[0]):        # 2000번의 이미지를 다 돌리기
    try:
        image = io.imread(final_data.loc[i]['image_url']) # i 번째 행의 이미지 url을 넘파이로 변환
        if(image.shape==(300, 300, 3)): # shape이 맞다면 
            x.append(image) # x 리스트에는 이미지를 추가시키고
            y.append(final_data.loc[i]['please_select_the_gender_of_the_person_in_the_picture']) # y list에는 성별을 추가한다
            filename = path + "final_" + str(i) + ".jpg" # file 이름을 경로와 같이 지정해주고 
            print(filename)
            io.imsave(filename, image)      # 한번만 다운받아서 저장
            print("정상저장_", i)
    except: # 예외가 있을 경우 그냥 계속 진행한다
        continue


import glob
import cv2 
path = glob.glob('homework/teacher/down/*.jpg')
x = []
y = []
for img in path:
    n = cv2.imread(img)
    x.append(n)
    y.append(final_data.loc[i]['please_select_the_gender_of_the_person_in_the_picture'])


# !pip install opencv-python
import cv2
import numpy as np

x2 = []
y2 = []

for i in range(len(x)): # x 리스트 길이 2000번만큼 돌린다
    img2 = cv2.resize(x[i], (50, 50)) # x 의 i번 째 그림을 (50,50,3) 으로 사이즈 변경하여 x2 리스트에 붙여줌
    x2.append(img2) 
    img_label = np.where(y[i]=='male', 1, 0) # y list에서 남자를 찾아 1로 바꾸고, 나머진 0으로 바꾸기
    y2.append(img_label) # 그걸 y2에다 붙여줌

print('여기까지 왔다.')




'''